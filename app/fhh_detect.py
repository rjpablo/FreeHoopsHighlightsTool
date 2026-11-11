"""Detection pass: players, ball, hoop regions, made-basket events, warm-ups.

One decode of the chosen time range at the analysis frame rate:
  * every Nth frame: YOLO (COCO person + sports ball) on overlapping tiles of
    the court band of the 360 panorama (or the flat frame), conf 0.1, as in
    the v2 auto-zoom pipeline. These detections drive the virtual camera.
  * every frame: a close-up "rim view" of each hoop. Frame differencing
    gates YOLO ball detection there, and ball candidates above the rim
    followed by candidates inside the net zone mark made baskets. Net
    motion spikes add confidence.
Detections are cached (npz) so clips can be re-planned without re-detecting.
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
    """Guess the rim's apparent width from its elevation (camera ~1.6 m high)."""
    p = max(2.0, float(pitch))
    dist = 1.45 / math.tan(math.radians(p))
    return float(np.clip(math.degrees(0.45 / max(dist, 1.0)), 0.8, 6.0))


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
        self.half = device.startswith("cuda")
        self.model = YOLO(str(weights))
        self.names = getattr(self.model, "names", {}) or {}
        log(f"Detector: {Path(weights).name} on {'GPU (CUDA)' if self.half else 'CPU'}")
        if not self.half:
            log("No CUDA GPU found: detection runs on the CPU and will be slow. "
                "Use Detection quality 'Fast' to speed it up.")

    def __call__(self, images: list[np.ndarray], imgsz: int, conf: float, classes=None):
        if not images:
            return []
        res = self.model.predict(images, imgsz=imgsz, conf=conf, classes=classes, verbose=False,
                                 device=self.device, half=self.half)
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
        fov = float(np.clip(hoop.width * 11.0, 8.0, 32.0))
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
    det = Detector(weights, st.threads, log, st.device)
    cv2.setNumThreads(max(1, st.threads))

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
    reader = FrameReader(ffmpeg, info, start, end - start, plan, ctl, fps=fps, threads=max(1, st.threads // 2))
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
            pano_img = scene.panorama(frame)
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
    reader = FrameReader(ffmpeg, info, start, span, plan, ctl, fps=2.0, threads=max(1, st.threads // 2))
    hoop_det = None
    hoop_classes: list[int] = []
    if hoop_model.is_file():
        try:
            hoop_det = Detector(hoop_model, st.threads, log, st.device)
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
            img = scene.panorama(frame)
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
# Events
# ---------------------------------------------------------------------------

@dataclass
class Event:
    time: float           # moment the ball goes through (or reaches) the rim
    hoop: int
    kind: str             # made | attempt
    score: float
    note: str = ""


def detect_events(d: Detections, hoops: list[Hoop], sensitivity: float = 0.5) -> list[Event]:
    sens = float(np.clip(sensitivity, 0, 1))
    made_thr = 0.62 - 0.3 * sens
    att_thr = 0.45 - 0.2 * sens
    events: list[Event] = []
    C = d.rim_cands
    N = d.rim_net
    for k in range(len(hoops)):
        c = C[C[:, 1] == k] if len(C) else C
        nn = N[N[:, 1] == k] if len(N) else N
        if len(c) == 0:
            continue
        base = float(np.median(nn[:, 2])) if len(nn) else 0.0
        t_arr, dx, dy, cf, src = c[:, 0], c[:, 2], c[:, 3], c[:, 4], c[:, 5]
        above = (np.abs(dx) < 1.6) & (dy > -4.5) & (dy < -0.35)
        through = (np.abs(dx) < 0.55) & (dy > -0.3) & (dy < 1.7)
        beside = (np.abs(dx) > 0.8) & (np.abs(dx) < 2.8) & (dy > -0.6) & (dy < 1.2)
        weight = np.where(src == 0, cf, 0.09)  # YOLO confidence, or 0.09 for a motion blob
        used_until = -1e9
        for i in np.flatnonzero(above):
            t1 = t_arr[i]
            if t1 < used_until:
                continue
            win = (t_arr > t1) & (t_arr <= t1 + 0.9)
            th = np.flatnonzero(win & through)
            later_above = np.flatnonzero(win & above & (dy > dy[i]))
            descending = len(later_above) > 0
            bes = np.flatnonzero((t_arr > t1) & (t_arr <= t1 + 1.3) & beside)
            s_above = float(min(1.0, weight[i] * 1.6 + (0.15 if descending else 0)))
            if len(th):
                j = th[np.argmax(weight[th])]
                t2 = float(t_arr[j])
                s_through = float(min(1.0, weight[j] * 1.6))
                net_sel = nn[(nn[:, 0] >= t2 - 0.15) & (nn[:, 0] <= t2 + 0.8)] if len(nn) else nn
                net_peak = float(net_sel[:, 2].max()) if len(net_sel) else 0.0
                net_bonus = 0.25 if net_peak > max(0.08, 3 * base) else 0.0
                bes_pen = 0.0
                if len(bes):
                    bes_pen = 0.2 * float(min(1.0, weight[bes].max() * 1.6))
                score = 0.45 * s_above + 0.45 * s_through + net_bonus - bes_pen + 0.1
                if score >= made_thr:
                    events.append(Event(t2, k, "made", round(score, 3),
                                        f"through rim{' + net' if net_bonus else ''}"))
                    used_until = t2 + 1.5
                    continue
            if descending and s_above >= att_thr:
                events.append(Event(float(t1) + 0.3, k, "attempt", round(0.6 * s_above, 3), "ball at rim"))
                used_until = t1 + 1.5
    events.sort(key=lambda e: e.time)
    # dedupe: same hoop within 2.5 s -> keep the best (made beats attempt)
    out: list[Event] = []
    for e in events:
        dup = next((o for o in out if o.hoop == e.hoop and abs(o.time - e.time) < 2.5), None)
        if dup is None:
            out.append(e)
            continue
        better = (e.kind == "made" and dup.kind != "made") or (e.kind == dup.kind and e.score > dup.score)
        if better:
            out[out.index(dup)] = e
    return out


def warmup_end(d: Detections, start: float) -> Optional[float]:
    """Pre-game shootaround has several balls in play at once. Return when it ends."""
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
