"""Detection pass: players, ball, hoop regions, made-basket events, warm-ups.

One decode of the chosen time range at the analysis frame rate:
  * every Nth frame: YOLO (COCO person + sports ball) on overlapping tiles of
    the court band of the 360 panorama (or the flat frame), conf 0.1, as in
    the v2 auto-zoom pipeline. These detections drive the virtual camera.
  * every frame: a close-up "rim view" of each hoop. Frame differencing
    gates YOLO ball detection there; candidates are verified by a downward
    through-rim / net check (roll-outs rejected). Net motion spikes add
    confidence. Warm-up, camera-setup, lone-shooter, and occlusion rules
    mark false candidates for the review step.
Detections are cached (npz) so clips can be re-planned without re-detecting.
See docs/HIGHLIGHT_STRATEGY.md.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from fhh_core import JobController, Progress, format_ts, models_dir
from fhh_source import (
    FLAT_HFOV, DecodePlan, FrameReader, Panorama, SceneRenderer, SourceInfo,
    decode_plan, flat_to_px, flat_to_scene, render_view, wrap180,
)

COCO_PERSON = 0
COCO_BALL = 32

QUALITY = {
    # model, analysis fps, run full-court detection every Nth analysis frame,
    # full-court tile inference size, rim-view size and inference size
    "fast":     dict(model="yolov8n.pt", fps=8.0,  every=4, tile_imgsz=640,  roi=320, roi_imgsz=480),
    "balanced": dict(model="yolov8s.pt", fps=10.0, every=4, tile_imgsz=640,  roi=384, roi_imgsz=640),
    "accurate": dict(model="yolov8m.pt", fps=12.0, every=3, tile_imgsz=960,  roi=448, roi_imgsz=640),
    "max":      dict(model="yolov8x.pt", fps=15.0, every=3, tile_imgsz=1280, roi=512, roi_imgsz=800),
}
QUALITY_LABELS = {
    "fast": "Fast (YOLOv8n, lightest)",
    "balanced": "Balanced (YOLOv8s)",
    "accurate": "Accurate (YOLOv8m)",
    "max": "Max (YOLOv8x, like the v2 edits; very slow on CPU)",
}
WEIGHT_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}",
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/{name}",
]

COURT_BAND = (-35.0, 25.0)     # latitude band (deg) searched for players/ball
ANALYSIS_BAND = (-40.0, 50.0)  # rows stitched during analysis (court + rims)


@dataclass
class Hoop:
    name: str
    yaw: float
    pitch: float
    width: float  # apparent rim width in degrees (or pseudo-degrees for flat)
    source: str = "manual"

    def to_dict(self) -> dict:
        return dict(name=self.name, yaw=round(self.yaw, 3), pitch=round(self.pitch, 3),
                    width=round(self.width, 3), source=self.source)

    @staticmethod
    def from_dict(d: dict) -> "Hoop":
        return Hoop(str(d.get("name", "A")), float(d["yaw"]), float(d["pitch"]),
                    float(d.get("width", 2.0)), str(d.get("source", "manual")))


def estimate_rim_width(pitch: float) -> float:
    """Guess the rim's apparent width from its elevation (camera ~1.6 m high).

    Uses a slightly conservative diameter so the through-rim zone stays tight;
    an overestimated width lets roll-outs and near-misses look like makes.
    """
    p = max(2.0, float(pitch))
    dist = 1.45 / math.tan(math.radians(p))
    # 0.40 m ≈ rim diameter with a bit of foreshortening; cap lower than before.
    return float(np.clip(math.degrees(0.40 / max(dist, 1.0)), 0.7, 3.2))


# ---------------------------------------------------------------------------
# YOLO wrapper
# ---------------------------------------------------------------------------

def ensure_weights(name: str, log: Callable[[str], None]) -> Path:
    dest = models_dir() / name
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    for tmpl in WEIGHT_URLS:
        url = tmpl.format(name=name)
        log(f"Downloading detection model {name} (first run only)...")
        tmp = dest.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as fp:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                last = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fp.write(chunk)
                    got += len(chunk)
                    if total and got - last > total / 5:
                        last = got
                        log(f"  {got / total * 100:.0f}% of {total / 1e6:.0f} MB")
            if tmp.stat().st_size > 1_000_000:
                tmp.replace(dest)
                return dest
        except Exception as exc:  # try the next mirror
            log(f"  download failed from {url}: {exc}")
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    raise RuntimeError(
        f"Could not download {name}. Connect to the internet, or put {name} into the models folder."
    )


class Detector:
    def __init__(self, weights: Path, threads: int, log: Callable[[str], None], device: str = "auto"):
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The detection libraries are not installed. Run install.bat first "
                f"(missing: {exc.name})."
            ) from exc
        try:
            torch.set_num_threads(max(1, threads))
        except Exception:
            pass
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        # half= only on CUDA; passing half=False still triggers Ultralytics warnings.
        self.half = str(device).startswith("cuda") and torch.cuda.is_available()
        self.model = YOLO(str(weights))
        self.names = getattr(self.model, "names", {}) or {}
        log(f"Detector: {Path(weights).name} on {'GPU (CUDA)' if self.half else 'CPU'}")
        if not self.half:
            log("No CUDA GPU found: detection runs on the CPU and will be slow. "
                "Use Detection quality 'Fast' to speed it up.")

    def __call__(self, images: list[np.ndarray], imgsz: int, conf: float, classes=None):
        if not images:
            return []
        kwargs = dict(imgsz=imgsz, conf=conf, classes=classes, verbose=False, device=self.device)
        if self.half:
            kwargs["half"] = True
        res = self.model.predict(images, **kwargs)
        out = []
        for r in res:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 6), np.float32))
                continue
            xyxy = b.xyxy.cpu().numpy()
            cf = b.conf.cpu().numpy()[:, None]
            cl = b.cls.cpu().numpy()[:, None]
            out.append(np.hstack([xyxy, cf, cl]).astype(np.float32))
        return out


# ---------------------------------------------------------------------------
# Tiling and rim views
# ---------------------------------------------------------------------------

def band_tiles(pano: Panorama, band: tuple[float, float]) -> list[tuple[np.ndarray, int, int]]:
    """Square tiles over the court band, wrapping around 360. -> (img, x0, y0)."""
    _, y_top = pano.to_px(0, band[1])
    _, y_bot = pano.to_px(0, band[0])
    y0 = int(max(0, math.floor(float(y_top))))
    y1 = int(min(pano.img.shape[0], math.ceil(float(y_bot))))
    size = max(64, y1 - y0)
    stride = int(size * 0.8)
    W = pano.W
    ext = np.concatenate([pano.img[y0:y1], pano.img[y0:y1, :size]], axis=1)
    tiles = []
    x = 0
    while x < W:
        tiles.append((np.ascontiguousarray(ext[:, x:x + size]), x, y0))
        x += stride
    return tiles


def flat_tiles(img: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    h, w = img.shape[:2]
    if w <= h * 1.3:
        return [(img, 0, 0)]
    n = int(math.ceil((w - h) / (h * 0.8))) + 1
    xs = np.linspace(0, w - h, n).astype(int)
    return [(np.ascontiguousarray(img[:, x:x + h]), int(x), 0) for x in xs]


@dataclass
class RimView:
    hoop: Hoop
    size: int
    fov: float
    center_pitch: float
    rim_x: float
    rim_y: float
    rim_px: float

    @staticmethod
    def make(hoop: Hoop, size: int, flat: bool) -> "RimView":
        # Tighter FOV than before so the through-rim cylinder stays useful.
        fov = float(np.clip(hoop.width * 9.0, 7.0, 26.0))
        off = fov * 0.12  # rim sits a bit below centre: more room for the arc above
        f = (size / 2) / math.tan(math.radians(fov) / 2)
        if flat:
            ppd = size / fov
            return RimView(hoop, size, fov, hoop.pitch + off, size / 2, size / 2 + off * ppd, hoop.width * ppd)
        return RimView(hoop, size, fov, hoop.pitch + off, size / 2,
                       size / 2 + f * math.tan(math.radians(off)), f * math.radians(hoop.width))

    def render(self, frame_or_pano, is_flat: bool, pano_full_h: int = 0, row0: int = 0) -> np.ndarray:
        import cv2
        if is_flat:
            img = frame_or_pano
            h, w = img.shape[:2]
            cx, cy = flat_to_px(self.hoop.yaw, self.center_pitch, w, h)
            side = w * self.fov / FLAT_HFOV
            s = self.size / side
            M = np.array([[s, 0, -(cx - side / 2) * s], [0, s, -(cy - side / 2) * s]], np.float32)
            return cv2.warpAffine(img, M, (self.size, self.size), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
        return render_view(frame_or_pano, self.hoop.yaw, self.center_pitch, self.fov,
                           self.size, self.size, pano_full_h, row0)

    def masks(self):
        if not hasattr(self, "_masks"):
            ys, xs = np.mgrid[0:self.size, 0:self.size]
            dx = (xs - self.rim_x) / self.rim_px
            dy = (ys - self.rim_y) / self.rim_px
            zone = (np.abs(dx) < 3) & (dy > -5) & (dy < 2.5)
            net = (np.abs(dx) < 0.6) & (dy > 0.2) & (dy < 1.6)
            self._masks = (zone, net)
        return self._masks

    def to_scene(self, x: float, y: float) -> tuple[float, float]:
        if self.size <= 0:
            return self.hoop.yaw, self.hoop.pitch
        ppd = self.size / self.fov
        return (self.hoop.yaw + (x - self.rim_x) / ppd / max(0.2, math.cos(math.radians(self.hoop.pitch))),
                self.hoop.pitch - (y - self.rim_y) / ppd)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@dataclass
class AnalysisSettings:
    quality: str = "balanced"
    start: float = 0.0
    end: Optional[float] = None
    sensitivity: float = 0.5          # 0 = strict, 1 = catch more
    lens_model: str = "x4"
    device: str = "auto"
    threads: int = 4


@dataclass
class Detections:
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))          # global frame times
    persons: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))   # t, yaw, pitch, h_deg, conf
    balls: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))     # t, yaw, pitch, conf
    rim_cands: np.ndarray = field(default_factory=lambda: np.zeros((0, 7))) # t, hoop, dx, dy, conf, src, size
    rim_net: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))   # t, hoop, net_motion, zone_motion
    rim_times: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, times=self.times, persons=self.persons, balls=self.balls,
                            rim_cands=self.rim_cands, rim_net=self.rim_net, rim_times=self.rim_times)

    @staticmethod
    def load(path: Path) -> "Detections":
        z = np.load(path)
        return Detections(z["times"], z["persons"], z["balls"], z["rim_cands"], z["rim_net"], z["rim_times"])


def _boxes_to_scene(boxes: np.ndarray, x0: int, y0: int, scale: float, to_scene, t: float,
                    persons: list, balls: list, wrap_w: int = 0) -> None:
    for x1, y1, x2, y2, cf, cl in boxes:
        cx = x0 + (x1 + x2) / 2 * scale
        cy = y0 + (y1 + y2) / 2 * scale
        if wrap_w:
            cx = cx % wrap_w
        yaw, pitch = to_scene(cx, cy)
        _, ptop = to_scene(cx, y0 + y1 * scale)
        _, pbot = to_scene(cx, y0 + y2 * scale)
        if int(cl) == COCO_PERSON:
            persons.append((t, float(yaw), float(pitch), float(abs(ptop - pbot)), float(cf)))
        elif int(cl) == COCO_BALL:
            balls.append((t, float(yaw), float(pitch), float(cf)))


def _dedupe_points(rows: list, t_index: int = 0, dist: float = 1.0) -> list:
    """Remove duplicates from overlapping tiles (same time, within dist degrees)."""
    out: list = []
    for r in sorted(rows, key=lambda r: -r[-1]):
        if any(o[0] == r[0] and abs(wrap180(o[1] - r[1])) < dist and abs(o[2] - r[2]) < dist for o in out):
            continue
        out.append(r)
    out.sort(key=lambda r: r[0])
    return out


def run_analysis(ffmpeg: str, info: SourceInfo, hoops: list[Hoop], st: AnalysisSettings,
                 ctl: JobController, prog: Progress, log: Callable[[str], None],
                 auto_hoops: bool = False) -> tuple[Detections, list[Hoop]]:
    import cv2
    q = QUALITY.get(st.quality, QUALITY["balanced"])
    weights = ensure_weights(q["model"], log)
    threads = ctl.thread_count(st.threads)
    det = Detector(weights, threads, log, st.device)
    cv2.setNumThreads(threads)

    start = max(0.0, st.start)
    end = min(info.duration, st.end) if st.end else info.duration
    if end - start < 2:
        raise ValueError("The time range is too short (needs at least a few seconds).")
    is_flat = info.kind == "flat"
    plan = decode_plan(info, max_lens=1920, max_width=1920 if is_flat else 3840)
    scene = SceneRenderer(info, plan, st.lens_model, eq_width=None if not info.is_fisheye else 2 * plan.lens_size,
                          lat_range=ANALYSIS_BAND if info.is_fisheye else None)

    if auto_hoops or not hoops:
        prog.begin("Finding hoops")
        hoops = scout_hoops(ffmpeg, info, plan, scene, det, q, start, end, st, ctl, prog, log)
        if not hoops:
            log("Could not find the hoops automatically. Use 'Pick hoops' and click both rims.")
            raise RuntimeError("No hoops found. Pick the hoops on a frame, then run again.")

    rimviews = [RimView.make(h, q["roi"], is_flat) for h in hoops]
    for rv in rimviews:
        log(f"Hoop {rv.hoop.name}: yaw {rv.hoop.yaw:.1f}, pitch {rv.hoop.pitch:.1f}, rim {rv.hoop.width:.2f} deg "
            f"({rv.hoop.source}); rim view {rv.fov:.0f} deg")

    prog.begin("Detecting plays")
    fps = q["fps"]
    reader = FrameReader(ffmpeg, info, start, end - start, plan, ctl, fps=fps,
                         threads=max(1, ctl.thread_count(st.threads) // 2))
    persons: list = []
    balls: list = []
    times: list = []
    rim_cands: list = []
    rim_net: list = []
    rim_times: list = []
    prev_gray: list[Optional[np.ndarray]] = [None] * len(rimviews)
    total = max(1, reader.expected_frames)
    sens = float(np.clip(st.sensitivity, 0, 1))
    conf_ball = 0.18 - 0.1 * sens
    i = 0
    for t, frame in reader:
        if is_flat:
            pano_img, full_h, row0 = frame, frame.shape[0], 0
        else:
            pano_img = scene.panorama(frame, t)
            full_h, row0 = scene.eq_h, scene.row0
        # --- full-court detection (players + ball) for the virtual camera
        if i % q["every"] == 0:
            times.append(t)
            p_rows: list = []
            b_rows: list = []
            if is_flat:
                tiles = flat_tiles(pano_img)
                h, w = pano_img.shape[:2]
                outs = det([tl for tl, _, _ in tiles], q["tile_imgsz"], 0.1, [COCO_PERSON, COCO_BALL])
                for (tl, x0, y0), boxes in zip(tiles, outs):
                    _boxes_to_scene(boxes, x0, y0, 1.0, lambda x, y: flat_to_scene(x, y, w, h), t, p_rows, b_rows)
            else:
                pano = Panorama(pano_img, full_h, row0)
                tiles = band_tiles(pano, COURT_BAND)
                outs = det([tl for tl, _, _ in tiles], q["tile_imgsz"], 0.1, [COCO_PERSON, COCO_BALL])
                for (tl, x0, y0), boxes in zip(tiles, outs):
                    _boxes_to_scene(boxes, x0, y0, 1.0, pano.to_scene, t, p_rows, b_rows,
                                    wrap_w=pano.W)
            persons.extend(_dedupe_points(p_rows, dist=1.5))
            balls.extend(_dedupe_points(b_rows, dist=0.8))
        # --- rim views (every analysis frame)
        rim_times.append(t)
        roi_imgs = []
        roi_idx = []
        for k, rv in enumerate(rimviews):
            roi = rv.render(pano_img, is_flat, full_h, row0)
            g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            pg = prev_gray[k]
            prev_gray[k] = g
            if pg is None:
                continue
            diff = cv2.absdiff(g, pg)
            mask = (diff > 18).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            r = rv.rim_px
            zone, net = rv.masks()
            zone_motion = float(mask[zone].mean()) if zone.any() else 0.0
            net_motion = float(mask[net].mean()) if net.any() else 0.0
            rim_net.append((t, k, net_motion, zone_motion))
            ball_area = math.pi * (0.27 * r) ** 2
            if mask[zone].sum() < 0.15 * ball_area:
                continue  # nothing moving near this rim: skip the detector
            # motion blobs as low-confidence ball candidates
            n, lab, stats, cents = cv2.connectedComponentsWithStats(mask * zone.astype(np.uint8), 8)
            for j in range(1, n):
                area = stats[j, cv2.CC_STAT_AREA]
                bw, bh = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
                if not (0.25 * ball_area <= area <= 2.5 * ball_area):
                    continue
                if max(bw, bh) > 2.2 * min(bw, bh) + 2:
                    continue
                cx, cy = cents[j]
                rim_cands.append((t, k, (cx - rv.rim_x) / r, (cy - rv.rim_y) / r, 0.12, 1, max(bw, bh) / r))
            roi_imgs.append(roi)
            roi_idx.append(k)
        if roi_imgs:
            outs = det(roi_imgs, q["roi_imgsz"], conf_ball, [COCO_BALL])
            for k, boxes in zip(roi_idx, outs):
                rv = rimviews[k]
                r = rv.rim_px
                for x1, y1, x2, y2, cf, cl in boxes:
                    size = max(x2 - x1, y2 - y1) / r
                    if size > 1.6 or size < 0.12:
                        continue
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    rim_cands.append((t, k, (cx - rv.rim_x) / r, (cy - rv.rim_y) / r, float(cf), 0, size))
                    yaw, pitch = rv.to_scene(cx, cy)
                    balls.append((t, float(yaw), float(pitch), float(cf) * 0.9))
        i += 1
        prog.update(i / total, f"Detecting plays: {format_ts(t)} of {format_ts(end)}")

    balls.sort(key=lambda r: r[0])
    d = Detections(
        times=np.array(times, np.float64),
        persons=np.array(persons, np.float64).reshape(-1, 5),
        balls=np.array(balls, np.float64).reshape(-1, 4),
        rim_cands=np.array(rim_cands, np.float64).reshape(-1, 7),
        rim_net=np.array(rim_net, np.float64).reshape(-1, 4),
        rim_times=np.array(rim_times, np.float64),
    )
    log(f"Detection done: {len(times)} full-court frames, {len(persons)} player boxes, "
        f"{len(balls)} ball boxes, {len(rim_cands)} rim candidates.")
    return d, hoops


def scout_hoops(ffmpeg, info, plan, scene, det, q, start, end, st, ctl, prog, log) -> list[Hoop]:
    """Automatic hoop guess.

    1) If models/hoop.pt exists (any YOLO model with a 'hoop'/'rim' class), use it.
    2) Otherwise, cluster high ball detections from a 2 fps scan of up to six
       minutes: balls are seen near the rims far more often than elsewhere
       above head height. This is a rough guess; picking the rims is better.
    """
    import cv2
    is_flat = info.kind == "flat"
    hoop_model = models_dir() / "hoop.pt"
    span = min(end - start, 360.0)
    reader = FrameReader(ffmpeg, info, start, span, plan, ctl, fps=2.0,
                         threads=max(1, ctl.thread_count(st.threads) // 2))
    hoop_det = None
    hoop_classes: list[int] = []
    if hoop_model.is_file():
        try:
            hoop_det = Detector(hoop_model, ctl.thread_count(st.threads), log, st.device)
            hoop_classes = [int(k) for k, v in hoop_det.names.items() if "hoop" in str(v).lower() or "rim" in str(v).lower()]
            if not hoop_classes:
                hoop_det = None
        except Exception as exc:
            log(f"Could not use models/hoop.pt: {exc}")
            hoop_det = None
    pts: list = []
    total = max(1, reader.expected_frames)
    n = 0
    for t, frame in reader:
        if is_flat:
            img = frame
            h, w = img.shape[:2]
            tiles = flat_tiles(img)
            conv = lambda x, y: flat_to_scene(x, y, w, h)
            wrap_w = 0
        else:
            img = scene.panorama(frame, t)
            pano = Panorama(img, scene.eq_h, scene.row0)
            tiles = band_tiles(pano, (-10.0, 40.0))
            conv = pano.to_scene
            wrap_w = pano.W
        if hoop_det is not None:
            outs = hoop_det([tl for tl, _, _ in tiles], q["tile_imgsz"], 0.25, hoop_classes)
            for (tl, x0, y0), boxes in zip(tiles, outs):
                for x1, y1, x2, y2, cf, cl in boxes:
                    cx = x0 + (x1 + x2) / 2
                    if wrap_w:
                        cx %= wrap_w
                    yaw, pitch = conv(cx, y0 + (y1 + y2) / 2)
                    ya, _ = conv(x0 + x1, y0 + y1)
                    yb, _ = conv(x0 + x2, y0 + y1)
                    pts.append((float(yaw), float(pitch), float(cf), float(abs(wrap180(yb - ya)))))
        else:
            outs = det([tl for tl, _, _ in tiles], q["tile_imgsz"], 0.1, [COCO_BALL])
            for (tl, x0, y0), boxes in zip(tiles, outs):
                for x1, y1, x2, y2, cf, cl in boxes:
                    cx = x0 + (x1 + x2) / 2
                    if wrap_w:
                        cx %= wrap_w
                    yaw, pitch = conv(cx, y0 + (y1 + y2) / 2)
                    if pitch > -2.0:
                        pts.append((float(yaw), float(pitch), float(cf), 0.0))
        n += 1
        prog.update(n / total, f"Finding hoops: {format_ts(t)}")
    if not pts:
        return []
    P = np.array(pts)
    # 2D histogram peaks (yaw wraps for 360)
    ybins = np.arange(-180, 181, 3.0) if not is_flat else np.arange(-FLAT_HFOV / 2, FLAT_HFOV / 2 + 1, 2.0)
    pbins = np.arange(-10, 61, 2.0)
    Hh, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=[ybins, pbins], weights=P[:, 2])
    Hh = cv2.GaussianBlur(Hh.astype(np.float32), (3, 3), 0)
    found: list[Hoop] = []
    min_sep = 60.0 if not is_flat else FLAT_HFOV * 0.3
    for _ in range(6):
        k = int(np.argmax(Hh))
        iy, ip = np.unravel_index(k, Hh.shape)
        if Hh[iy, ip] <= 0:
            break
        yaw = (ybins[iy] + ybins[iy + 1]) / 2
        pitch = (pbins[ip] + pbins[ip + 1]) / 2
        Hh[max(0, iy - 6):iy + 7, :] = 0
        if any(abs(wrap180(yaw - h.yaw)) < min_sep for h in found):
            continue
        near = P[(np.abs(wrap180(P[:, 0] - yaw)) < 4) & (np.abs(P[:, 1] - pitch) < 4)]
        if len(near):
            yaw = float(np.average(near[:, 0], weights=near[:, 2]))
            pitch = float(np.average(near[:, 1], weights=near[:, 2]))
        width = float(np.median(near[:, 3])) if hoop_det is not None and len(near) else estimate_rim_width(pitch)
        found.append(Hoop("AB"[len(found)], yaw, pitch, width, "auto-model" if hoop_det is not None else "auto-guess"))
        if len(found) == 2:
            break
    if found and hoop_det is None:
        log("Hoops were guessed from where the ball shows up most. This is rough: "
            "check them with 'Pick hoops' if the results look wrong.")
    return found


# ---------------------------------------------------------------------------
# Events (strategy A: broad candidates -> rim verify -> reject -> rank)
# ---------------------------------------------------------------------------

@dataclass
class Event:
    time: float           # moment the ball goes through (or reaches) the rim
    hoop: int
    kind: str             # made | attempt | play
    score: float
    note: str = ""
    rejected: bool = False
    reject_reason: str = ""


def _rim_weight(cf: float, src: float) -> float:
    return float(cf) if int(src) == 0 else 0.09


def verify_rim_make(cands: np.ndarray, net_rows: np.ndarray, t_hint: float,
                    window: float = 1.2, sensitivity: float = 0.5) -> Optional[tuple[float, float, str]]:
    """Confirm a made basket by requiring a downward pass through the rim into the net.

    cands columns: t, hoop, dx, dy, conf, src, size (already filtered to one hoop).
    Returns (ball_in_time, score, note) or None.
    """
    if len(cands) == 0:
        return None
    sens = float(np.clip(sensitivity, 0, 1))
    made_thr = 0.68 - 0.28 * sens
    t_arr = cands[:, 0]
    dx, dy = cands[:, 2], cands[:, 3]
    weight = np.array([_rim_weight(c[4], c[5]) for c in cands])
    sel = (t_arr >= t_hint - 0.55) & (t_arr <= t_hint + window)
    if not sel.any():
        return None
    idx = np.flatnonzero(sel)
    above = (np.abs(dx) < 1.4) & (dy > -4.5) & (dy < -0.25)
    through = (np.abs(dx) < 0.48) & (dy > -0.25) & (dy < 1.55)
    net_zone = (np.abs(dx) < 0.55) & (dy > 0.15) & (dy < 1.8)
    beside = (np.abs(dx) > 0.75) & (np.abs(dx) < 2.8) & (dy > -0.6) & (dy < 1.2)
    above_i = idx[above[idx]]
    if len(above_i) == 0:
        # Scoreboard path: allow a through/net-only confirmation near the hint.
        th_only = idx[through[idx] | net_zone[idx]]
        if len(th_only) == 0:
            return None
        j = int(th_only[np.argmax(weight[th_only])])
        t2 = float(t_arr[j])
        net_bonus, net_peak, base = _net_bonus(net_rows, t2)
        if net_bonus <= 0 and weight[j] < 0.25:
            return None
        score = 0.35 + 0.4 * min(1.0, weight[j] * 1.6) + net_bonus
        if score < made_thr:
            return None
        return t2, round(float(score), 3), f"rim confirm{' + net' if net_bonus else ''}"

    best: Optional[tuple[float, float, str]] = None
    used_until = -1e9
    for i in above_i:
        t1 = float(t_arr[i])
        if t1 < used_until:
            continue
        win = idx[(t_arr[idx] > t1) & (t_arr[idx] <= t1 + 1.0)]
        if len(win) == 0:
            continue
        # Must descend: later samples closer to / below the rim than the peak above.
        later_above = win[above[win] & (dy[win] > dy[i] - 0.05)]
        descending = len(later_above) > 0 or any(dy[j] > dy[i] for j in win)
        th = win[through[win]]
        nz = win[net_zone[win]]
        if len(th) == 0 and len(nz) == 0:
            continue
        # Prefer the earliest sample that is through the rim or in the net, with
        # dy greater than the above sample (downward in image space).
        path = np.concatenate([th, nz]) if len(th) and len(nz) else (th if len(th) else nz)
        path = path[dy[path] >= dy[i] - 0.1]
        if len(path) == 0:
            continue
        # Order by time; require overall downward progress through the cylinder.
        path = path[np.argsort(t_arr[path])]
        j = int(path[0])
        for cand_j in path:
            if dy[cand_j] > -0.05:  # at rim line or below
                j = int(cand_j)
                break
        t2 = float(t_arr[j])
        # Roll-out rejection: ball reappears clearly above the rim after "through".
        after = idx[(t_arr[idx] > t2) & (t_arr[idx] <= t2 + 0.7)]
        roll_out = after[(np.abs(dx[after]) < 1.2) & (dy[after] < -0.55) & (dy[after] < dy[j] - 0.4)]
        if len(roll_out) and weight[roll_out].max() >= 0.12:
            continue
        bes = idx[(t_arr[idx] > t1) & (t_arr[idx] <= t2 + 0.4) & beside[idx]]
        s_above = float(min(1.0, weight[i] * 1.6 + (0.12 if descending else 0)))
        s_through = float(min(1.0, weight[j] * 1.6))
        net_bonus, net_peak, base = _net_bonus(net_rows, t2)
        # Path must continue into the net zone or show net motion (swish / soft make).
        into_net = np.any(net_zone[path]) or net_bonus > 0
        if not into_net and dy[j] < 0.2:
            continue
        bes_pen = 0.22 * float(min(1.0, weight[bes].max() * 1.6)) if len(bes) else 0.0
        score = 0.4 * s_above + 0.4 * s_through + net_bonus - bes_pen + (0.12 if into_net else 0.0)
        if score < made_thr:
            continue
        note = "through rim"
        if net_bonus:
            note += " + net"
        if not descending:
            note += " (weak descent)"
        cand = (t2, round(float(score), 3), note)
        if best is None or cand[1] > best[1]:
            best = cand
        used_until = t2 + 1.4
    return best


def _net_bonus(net_rows: np.ndarray, t2: float) -> tuple[float, float, float]:
    if len(net_rows) == 0:
        return 0.0, 0.0, 0.0
    base = float(np.median(net_rows[:, 2]))
    net_sel = net_rows[(net_rows[:, 0] >= t2 - 0.1) & (net_rows[:, 0] <= t2 + 0.85)]
    net_peak = float(net_sel[:, 2].max()) if len(net_sel) else 0.0
    bonus = 0.28 if net_peak > max(0.08, 2.8 * base) else 0.0
    return bonus, net_peak, base


def _candidate_seeds(cands: np.ndarray, net_rows: np.ndarray) -> list[float]:
    """Broad candidate times: ball/motion above or through the rim, or net spikes."""
    seeds: list[float] = []
    if len(cands):
        t_arr, dx, dy = cands[:, 0], cands[:, 2], cands[:, 3]
        interesting = (
            ((np.abs(dx) < 1.6) & (dy > -4.5) & (dy < -0.2))
            | ((np.abs(dx) < 0.55) & (dy > -0.4) & (dy < 1.8))
        )
        for t in t_arr[interesting]:
            seeds.append(float(t))
    if len(net_rows):
        base = float(np.median(net_rows[:, 2])) if len(net_rows) else 0.0
        thr = max(0.1, 3.2 * base)
        for row in net_rows:
            if row[2] >= thr:
                seeds.append(float(row[0]))
    seeds.sort()
    # Collapse seeds within 0.35 s.
    out: list[float] = []
    for t in seeds:
        if not out or t - out[-1] >= 0.35:
            out.append(t)
        else:
            out[-1] = 0.5 * (out[-1] + t)
    return out


def detect_events(d: Detections, hoops: list[Hoop], sensitivity: float = 0.5) -> list[Event]:
    """Generate make/attempt events from rim candidates (no-scoreboard path)."""
    sens = float(np.clip(sensitivity, 0, 1))
    att_thr = 0.42 - 0.18 * sens
    events: list[Event] = []
    C = d.rim_cands
    N = d.rim_net
    for k in range(len(hoops)):
        c = C[C[:, 1] == k] if len(C) else C
        nn = N[N[:, 1] == k] if len(N) else N
        if len(c) == 0:
            continue
        seeds = _candidate_seeds(c, nn)
        used_until = -1e9
        for seed in seeds:
            if seed < used_until:
                continue
            verified = verify_rim_make(c, nn, seed, sensitivity=sens)
            if verified is not None:
                t2, score, note = verified
                events.append(Event(t2, k, "made", score, note))
                used_until = t2 + 1.6
                continue
            # Attempt: descending ball above the rim without a clean through.
            t_arr, dx, dy = c[:, 0], c[:, 2], c[:, 3]
            weight = np.array([_rim_weight(r[4], r[5]) for r in c])
            above = (np.abs(c[:, 0] - seed) < 0.45) & (np.abs(dx) < 1.5) & (dy > -4.5) & (dy < -0.3)
            if not above.any():
                continue
            i = int(np.flatnonzero(above)[np.argmax(weight[above])])
            win = (t_arr > t_arr[i]) & (t_arr <= t_arr[i] + 0.9)
            descending = np.any(win & (np.abs(dx) < 1.5) & (dy > dy[i]) & (dy < -0.1))
            s_above = float(min(1.0, weight[i] * 1.6 + (0.12 if descending else 0)))
            if descending and s_above >= att_thr:
                events.append(Event(float(t_arr[i]) + 0.35, k, "attempt",
                                    round(0.55 * s_above, 3), "ball at rim"))
                used_until = float(t_arr[i]) + 1.4
    events.sort(key=lambda e: e.time)
    return _dedupe_events(events)


def events_from_score_times(score_times: list[float], d: Detections, hoops: list[Hoop],
                            sensitivity: float = 0.5,
                            hoop_hints: Optional[list[Optional[int]]] = None) -> list[Event]:
    """Scoreboard / score-log path: each score change is a candidate, refined by rim check."""
    sens = float(np.clip(sensitivity, 0, 1))
    C = d.rim_cands
    N = d.rim_net
    events: list[Event] = []
    for n, t0 in enumerate(score_times):
        hint = hoop_hints[n] if hoop_hints and n < len(hoop_hints) else None
        best: Optional[Event] = None
        hoop_range = [hint] if hint is not None else list(range(len(hoops)))
        for k in hoop_range:
            if k is None or k < 0 or k >= len(hoops):
                continue
            c = C[C[:, 1] == k] if len(C) else C
            nn = N[N[:, 1] == k] if len(N) else N
            verified = verify_rim_make(c, nn, float(t0), window=1.4, sensitivity=sens)
            if verified is None:
                # Soft accept: scoreboard already confirmed a point; keep with lower score
                # only if there is *some* rim activity nearby.
                near = c[(np.abs(c[:, 0] - t0) <= 1.2)] if len(c) else c
                if len(near) == 0 and len(nn):
                    near_net = nn[(nn[:, 0] >= t0 - 0.4) & (nn[:, 0] <= t0 + 1.0)]
                    if len(near_net) and float(near_net[:, 2].max()) > 0.05:
                        ev = Event(float(t0), k, "made", 0.55, "scoreboard (weak rim)")
                        if best is None or ev.score > best.score:
                            best = ev
                    continue
                if len(near) == 0:
                    continue
                ev = Event(float(t0), k, "made", 0.58, "scoreboard (rim activity)")
            else:
                t2, score, note = verified
                # Prefer the verified rim time when it is close to the score change.
                t_use = t2 if abs(t2 - t0) <= 1.5 else float(t0)
                ev = Event(t_use, k, "made", max(score, 0.7), f"scoreboard + {note}")
            if best is None or ev.score > best.score:
                best = ev
        if best is None:
            # No rim activity at either hoop: drop (weak / obstructed / wrong time).
            events.append(Event(float(t0), 0, "made", 0.2, "scoreboard only",
                                rejected=True, reject_reason="no rim confirmation"))
        else:
            events.append(best)
    events.sort(key=lambda e: e.time)
    return _dedupe_events(events)


def _dedupe_events(events: list[Event], gap: float = 2.2) -> list[Event]:
    out: list[Event] = []
    for e in events:
        dup = next((o for o in out if o.hoop == e.hoop and abs(o.time - e.time) < gap
                    and (not o.rejected) == (not e.rejected)), None)
        if dup is None:
            out.append(e)
            continue
        better = (e.kind == "made" and dup.kind != "made") or (e.kind == dup.kind and e.score > dup.score)
        if better:
            out[out.index(dup)] = e
    return out


def person_count_near(persons: np.ndarray, hoop: Hoop, t: float, radius: float = 28.0,
                      half_window: float = 1.0) -> int:
    if len(persons) == 0:
        return 0
    sel = persons[np.abs(persons[:, 0] - t) <= half_window]
    if len(sel) == 0:
        return 0
    near = (np.abs(wrap180(sel[:, 1] - hoop.yaw)) <= radius) & (sel[:, 2] > hoop.pitch - 30) & (
        sel[:, 2] < hoop.pitch + 14)
    # Unique-ish people by yaw bin.
    ys = sel[near][:, 1]
    if len(ys) == 0:
        return 0
    bins = np.round(ys / 4.0)
    return int(len(np.unique(bins)))


def total_persons_at(persons: np.ndarray, t: float, half_window: float = 0.8) -> int:
    if len(persons) == 0:
        return 0
    sel = persons[np.abs(persons[:, 0] - t) <= half_window]
    if len(sel) == 0:
        return 0
    bins = np.round(sel[:, 1] / 5.0)
    return int(len(np.unique(bins)))


def large_person_occlusion(persons: np.ndarray, hoop: Hoop, t: float) -> bool:
    """Someone standing close in front of the rim (blocking the camera view of the basket)."""
    if len(persons) == 0:
        return False
    sel = persons[np.abs(persons[:, 0] - t) <= 0.7]
    if len(sel) == 0:
        return False
    # Very tall apparent height near the hoop yaw = close to the lens / covering the rim.
    close = (np.abs(wrap180(sel[:, 1] - hoop.yaw)) < 18.0) & (sel[:, 3] >= 28.0)
    return bool(np.any(close))


def camera_setup_end(d: Detections, start: float, max_scan: float = 90.0) -> Optional[float]:
    """Return when camera setup / people blocking the lens seem finished.

    Heuristic: early stretch with few people and/or unusually large person boxes
    (someone adjusting the tripod in front of the lens), then a quieter court.
    """
    if len(d.times) < 8:
        return None
    t0 = float(d.times[0])
    t_end = min(float(d.times[-1]), t0 + max_scan)
    setup_until = None
    t = t0
    while t < t_end:
        sel = d.persons[np.abs(d.persons[:, 0] - t) <= 1.0] if len(d.persons) else d.persons
        n = total_persons_at(d.persons, t) if len(d.persons) else 0
        big = bool(len(sel) and np.any(sel[:, 3] >= 32.0))
        # Near t=0 with almost nobody, or a huge person filling the frame.
        if t <= t0 + 25.0 and (n <= 1 or big):
            setup_until = t + 2.0
        elif setup_until is not None and n >= 3 and not big:
            break
        t += 2.0
    # Always treat the first ~2 s as suspect when the range starts at 0 (lens cap / walk-up).
    if start <= 0.5:
        setup_until = max(setup_until or 0.0, start + 2.5)
    return setup_until


def game_start_time(d: Detections, start: float) -> Optional[float]:
    """When a real game-like crowd first appears (5+ people sustained)."""
    if len(d.times) < 10 or len(d.persons) == 0:
        return None
    win = 20.0
    t = float(d.times[0])
    last = float(d.times[-1])
    while t < last:
        sel = (d.times >= t) & (d.times < t + win)
        if sel.sum() >= 4:
            counts = [total_persons_at(d.persons, float(tt)) for tt in d.times[sel]]
            if float(np.median(counts)) >= 5.0:
                return max(start, t)
        t += win / 2
    return None


def warmup_end(d: Detections, start: float) -> Optional[float]:
    """Pre-game shootaround: several balls, or lone/few shooters before the game starts."""
    multi_end = _warmup_multiball_end(d, start)
    game = game_start_time(d, start)
    lone_end = None
    if game is not None and game > start + 15:
        # Before the game, sparse players making baskets = warm-up / lone shooters.
        lone_end = game
    ends = [x for x in (multi_end, lone_end) if x is not None]
    return max(ends) if ends else multi_end


def _warmup_multiball_end(d: Detections, start: float) -> Optional[float]:
    if len(d.times) < 10 or len(d.balls) == 0:
        return None
    B = d.balls[d.balls[:, 3] >= 0.25]
    counts = []
    for t in d.times:
        sel = B[np.abs(B[:, 0] - t) < 1e-3]
        n = 0
        kept: list = []
        for row in sel:
            if all(abs(wrap180(row[1] - k[1])) > 3 or abs(row[2] - k[2]) > 3 for k in kept):
                kept.append(row)
                n += 1
        counts.append(n)
    counts = np.array(counts)
    win = 30.0
    t0 = float(d.times[0])
    last_multi = None
    quiet_run = 0
    t = t0
    while t < d.times[-1]:
        sel = (d.times >= t) & (d.times < t + win)
        if sel.sum() >= 5:
            frac = float((counts[sel] >= 2).mean())
            if frac >= 0.2:
                last_multi = t + win
                quiet_run = 0
            else:
                quiet_run += 1
                if quiet_run >= 4 and last_multi is not None:
                    break
                if quiet_run >= 4 and last_multi is None:
                    return None
        t += win
    return last_multi


def apply_rejection_rules(events: list[Event], d: Detections, hoops: list[Hoop],
                          warmup: Optional[float], setup_until: Optional[float],
                          skip_warmup: bool = True) -> list[Event]:
    """Mark events that look like camera setup, warm-ups, lone shooters, or occlusion."""
    out: list[Event] = []
    for e in events:
        if e.rejected:
            out.append(e)
            continue
        reasons: list[str] = []
        if setup_until is not None and e.time <= setup_until:
            reasons.append("camera setup")
        if skip_warmup and warmup is not None and e.time < warmup:
            reasons.append("warm-up")
        hoop = hoops[e.hoop] if 0 <= e.hoop < len(hoops) else None
        if hoop is not None:
            crowd = person_count_near(d.persons, hoop, e.time)
            total = total_persons_at(d.persons, e.time)
            # Lone shooter: one person at the hoop and almost nobody else on court.
            if crowd <= 1 and total <= 2 and e.kind == "made":
                # Allow late-game isolation (fast break) only when the other end also
                # has some activity in the recent past, or total people later grow.
                if warmup is None or e.time < (warmup + 30):
                    reasons.append("lone shooter")
            if large_person_occlusion(d.persons, hoop, e.time):
                reasons.append("occlusion")
        if reasons:
            e = Event(e.time, e.hoop, e.kind, e.score, e.note,
                      rejected=True, reject_reason="; ".join(reasons))
        out.append(e)
    return out


def rank_events(events: list[Event]) -> list[Event]:
    """Stable game-order list; rejected events keep lower effective score for review."""
    def key(e: Event):
        return (e.time, -e.score if not e.rejected else 0.0)
    return sorted(events, key=key)
