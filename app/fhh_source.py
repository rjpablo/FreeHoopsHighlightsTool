"""Input detection, Insta360 stitching, and virtual-camera reprojection.

Supported inputs
  * .insv with two video streams (Insta360 X3/X4: one stream per lens)
  * split .insv pairs (..._00_NNN.insv + ..._10_NNN.insv, one lens each)
  * single-stream side-by-side dual fisheye (older cameras, 2:1 frame)
  * already-stitched equirectangular 360 MP4 (2:1)
  * ordinary flat video (the "virtual camera" becomes a moving crop)

Scene coordinates are degrees: yaw (left/right, -180..180) and pitch (up is
positive). For flat video they are pseudo-angles: the full frame width is
FLAT_HFOV degrees wide, so the same camera logic works for both.
"""

from __future__ import annotations

import json
import math
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from fhh_core import JobController, assets_dir, parse_ratio, popen_kwargs, probe_json

FLAT_HFOV = 100.0


def _cv2():
    import cv2  # imported lazily so the window opens even if OpenCV is missing
    return cv2


# ---------------------------------------------------------------------------
# Input detection
# ---------------------------------------------------------------------------

@dataclass
class SourceInfo:
    kind: str                 # streams | split | sbs | equirect | flat
    files: list[Path]         # 1 file, or [lens0, lens1] for split
    width: int                # decoded frame width (hstacked for dual fisheye)
    height: int
    lens_size: int = 0        # fisheye image size (0 for equirect/flat)
    fps: float = 30.0
    fps_str: str = "30/1"
    duration: float = 0.0
    has_audio: bool = False
    audio_file: int = 0       # index into files
    sample_rate: int = 48000
    creation_time: str = ""
    notes: list[str] = field(default_factory=list)
    lrv: Optional[Path] = None
    stab: object = None          # Stabilizer from the .lrv, when its motion data can be read

    @property
    def is_sphere(self) -> bool:
        return self.kind != "flat"

    @property
    def is_fisheye(self) -> bool:
        return self.kind in ("streams", "split", "sbs")

    def describe(self) -> str:
        names = {
            "streams": f"Insta360 dual-fisheye .insv (2 lenses x {self.lens_size}px)",
            "split": f"Insta360 split pair _00_/_10_ ({self.lens_size}px lenses)",
            "sbs": f"side-by-side dual fisheye ({self.lens_size}px lenses)",
            "equirect": f"stitched 360 equirectangular {self.width}x{self.height}",
            "flat": f"flat video {self.width}x{self.height}",
        }
        text = f"{names.get(self.kind, self.kind)}, {self.fps:.3f} fps, {self.duration/60:.1f} min"
        if self.stab is not None and self.lrv is not None:
            text += f". Leveling with {self.lrv.name}"
        return text


def _video_streams(data: dict) -> list[dict]:
    return [s for s in data.get("streams") or [] if s.get("codec_type") == "video"
            and (s.get("disposition") or {}).get("attached_pic", 0) == 0]


def _partner_file(path: Path) -> Optional[Path]:
    m = re.search(r"_(00|10)_(\d+)$", path.stem)
    if not m:
        return None
    other = "10" if m.group(1) == "00" else "00"
    stem = path.stem[: m.start()] + f"_{other}_{m.group(2)}"
    for cand in path.parent.glob(stem + ".*"):
        if cand.suffix.lower() == path.suffix.lower():
            return cand
    return None


def _duration(data: dict, vs: dict) -> float:
    for c in ((data.get("format") or {}).get("duration"), vs.get("duration")):
        try:
            v = float(c)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return 0.0


def _corners_dark(ffmpeg: str, path: Path, t: float, w: int, h: int) -> bool:
    """True if the four corners of each half are black (fisheye circles)."""
    try:
        frame = grab_frame_raw(ffmpeg, [path], t, "[0:v:0]scale=960:480", 960, 480)
    except Exception:
        return False
    vals = []
    p = 18
    for half in (0, 480):
        for (y, x) in ((0, half), (0, half + 480 - p), (480 - p, half), (480 - p, half + 480 - p)):
            vals.append(float(frame[y:y + p, x:x + p].mean()))
    return max(vals) < 22.0


def detect_source(path: Path, ffmpeg: str, ffprobe: str, force_kind: str = "auto") -> SourceInfo:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    data = probe_json(path, ffprobe)
    vids = _video_streams(data)
    if not vids:
        raise RuntimeError(f"{path.name} has no video stream.")
    auds = [s for s in data.get("streams") or [] if s.get("codec_type") == "audio"]
    v0 = vids[0]
    w, h = int(v0.get("width") or 0), int(v0.get("height") or 0)
    fps_str = v0.get("r_frame_rate") or v0.get("avg_frame_rate") or "30/1"
    fps = parse_ratio(fps_str, 30.0) or 30.0
    if fps > 240 or fps < 1:
        fps_str = v0.get("avg_frame_rate") or "30/1"
        fps = parse_ratio(fps_str, 30.0) or 30.0
    dur = _duration(data, v0)
    ctime = str(((data.get("format") or {}).get("tags") or {}).get("creation_time") or "")
    info = SourceInfo(kind="flat", files=[path], width=w, height=h, fps=fps, fps_str=fps_str,
                      duration=dur, has_audio=bool(auds), creation_time=ctime)
    if auds:
        try:
            info.sample_rate = int(auds[0].get("sample_rate") or 48000)
        except ValueError:
            pass
    ext = path.suffix.lower()
    kind = force_kind if force_kind != "auto" else ""

    if not kind:
        if len(vids) >= 2 and abs(w - h) <= 2 and int(vids[1].get("width") or 0) == w:
            kind = "streams"
        elif abs(w - h) <= 2 and ext == ".insv":
            partner = _partner_file(path)
            if partner is not None:
                kind = "split"
        elif abs(w - 2 * h) <= 4:
            spherical = any("spherical" in json.dumps(s.get("side_data_list") or []).lower()
                            or "equirect" in json.dumps(s.get("side_data_list") or []).lower()
                            for s in vids)
            if spherical:
                kind = "equirect"
            elif _corners_dark(ffmpeg, path, min(max(dur * 0.3, 1.0), max(dur - 1, 0)), w, h):
                kind = "sbs"
            else:
                kind = "equirect"
        else:
            kind = "flat"

    info.kind = kind
    if kind == "streams":
        info.lens_size = w
        info.width, info.height = 2 * w, w
    elif kind == "split":
        partner = _partner_file(path)
        if partner is None:
            raise RuntimeError(f"Could not find the other lens file for {path.name} (_00_/_10_ pair).")
        lens0, lens1 = (path, partner) if "_00_" in path.name else (partner, path)
        info.files = [lens0, lens1]
        info.lens_size = w
        info.width, info.height = 2 * w, w
        if lens0 != path:
            d0 = probe_json(lens0, ffprobe)
            info.has_audio = any(s.get("codec_type") == "audio" for s in d0.get("streams") or [])
        info.audio_file = 0
        info.notes.append(f"Using split pair: {lens0.name} + {lens1.name}")
    elif kind == "sbs":
        info.lens_size = h
    elif kind == "flat" and abs(w - 2 * h) <= 4:
        info.notes.append("2:1 video treated as flat because you chose 'flat'.")
    if ext == ".insv":
        from fhh_stab import find_lrv, load_stabilizer
        lrv = find_lrv(path)
        if lrv is None and info.files:
            lrv = find_lrv(info.files[0])
        if lrv is not None:
            info.lrv = lrv
            info.stab = load_stabilizer(lrv, info.duration)
            if info.stab is not None:
                info.notes.append(f"Stabilizing with {lrv.name} (level horizon, less shake).")
            else:
                info.notes.append(f"Found {lrv.name}, but it has no usable motion data. Continuing without stabilization.")
    return info


# ---------------------------------------------------------------------------
# FFmpeg frame reading
# ---------------------------------------------------------------------------

def _input_args(files: list[Path], start: float, dur: Optional[float]) -> list[str]:
    args: list[str] = []
    for f in files:
        if start > 0:
            args += ["-ss", f"{start:.3f}"]
        if dur is not None:
            args += ["-t", f"{dur:.3f}"]
        args += ["-i", str(f)]
    return args


def base_filter(info: SourceInfo) -> str:
    if info.kind == "streams":
        return "[0:v:0][0:v:1]hstack=inputs=2"
    if info.kind == "split":
        return "[0:v:0][1:v:0]hstack=inputs=2"
    return "[0:v:0]null"


def grab_frame_raw(ffmpeg: str, files: list[Path], t: float, graph: str, w: int, h: int) -> np.ndarray:
    cmd = [ffmpeg, "-v", "error", *_input_args(files, max(0.0, t), None),
           "-filter_complex", graph + ",format=bgr24[o]", "-map", "[o]",
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    r = subprocess.run(cmd, capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
    need = w * h * 3
    if r.returncode != 0 or len(r.stdout) < need:
        raise RuntimeError("Could not read a frame: " + r.stderr.decode(errors="replace")[-400:])
    return np.frombuffer(r.stdout[:need], np.uint8).reshape(h, w, 3)


@dataclass
class DecodePlan:
    width: int
    height: int
    lens_size: int


def decode_plan(info: SourceInfo, max_lens: Optional[int] = None, max_width: Optional[int] = None) -> DecodePlan:
    if info.is_fisheye:
        s = info.lens_size
        if max_lens and s > max_lens:
            s = max_lens - (max_lens % 2)
        return DecodePlan(2 * s, s, s)
    w, h = info.width, info.height
    if max_width and w > max_width:
        h = int(round(h * max_width / w)) // 2 * 2
        w = max_width
    return DecodePlan(w, h, 0)


class FrameReader:
    """Yield (time_seconds, bgr_frame) from start for dur seconds at fps (None = native)."""

    def __init__(self, ffmpeg: str, info: SourceInfo, start: float, dur: float,
                 plan: DecodePlan, ctl: JobController, fps: Optional[float] = None,
                 threads: int = 2):
        self.info, self.plan, self.ctl = info, plan, ctl
        self.start, self.dur = start, dur
        self.fps = fps or info.fps
        graph = base_filter(info)
        if (plan.width, plan.height) != ((2 * info.lens_size, info.lens_size) if info.is_fisheye else (info.width, info.height)):
            graph += f",scale={plan.width}:{plan.height}:flags=area"
        if fps:
            graph += f",fps={fps:.4f}"
        graph += ",format=bgr24[o]"
        files = info.files if info.kind == "split" else info.files[:1]
        self.cmd = [ffmpeg, "-v", "error", "-threads", str(max(1, threads)),
                    *_input_args(files, start, dur), "-filter_complex", graph, "-map", "[o]",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc: Optional[subprocess.Popen] = None
        self.expected_frames = int(math.floor(dur * self.fps + 1e-6))

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        fs = self.plan.width * self.plan.height * 3
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=0, **popen_kwargs(self.ctl.priority_mode))
        self.ctl.attach(self.proc)
        q: queue.Queue = queue.Queue(maxsize=6)
        err: list[bytes] = []

        def pump_err():
            assert self.proc and self.proc.stderr
            err.append(self.proc.stderr.read())

        def reader():
            assert self.proc and self.proc.stdout
            while True:
                buf = bytearray(fs)
                mv = memoryview(buf)
                got = 0
                while got < fs:
                    k = self.proc.stdout.readinto(mv[got:])
                    if not k:
                        break
                    got += k
                if got < fs:
                    q.put(None)
                    return
                q.put(buf)

        threading.Thread(target=pump_err, daemon=True).start()
        threading.Thread(target=reader, daemon=True).start()
        i = 0
        try:
            while True:
                self.ctl.checkpoint()
                try:
                    buf = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if buf is None:
                    break
                frame = np.frombuffer(buf, np.uint8).reshape(self.plan.height, self.plan.width, 3)
                yield self.start + i / self.fps, frame
                i += 1
        finally:
            if self.proc.poll() is None:
                try:
                    self.proc.kill()
                except OSError:
                    pass
            self.proc.wait()
        self.ctl.raise_if_cancelled()
        if i == 0:
            raise RuntimeError("No frames decoded: " + b"".join(err).decode(errors="replace")[-600:])


# ---------------------------------------------------------------------------
# Lens models and stitching (from the Quoted Cafe .insv -> 360 converter)
# ---------------------------------------------------------------------------

def rotm(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def eq_rays(W: int, H: int, lat_range: Optional[tuple[float, float]] = None) -> tuple[np.ndarray, int]:
    """Unit rays for equirect pixels. Returns (rays, first_row)."""
    r0, r1 = 0, H
    if lat_range:
        lo, hi = lat_range  # degrees, lo < hi
        r0 = max(0, int(math.floor((90 - hi) / 180 * H)))
        r1 = min(H, int(math.ceil((90 - lo) / 180 * H)))
    j, i = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(r0, r1, dtype=np.float64))
    lon = (j + 0.5) / W * 2 * np.pi - np.pi
    lat = np.pi / 2 - (i + 0.5) / H * np.pi
    x = np.cos(lat) * np.sin(lon)
    y = -np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    return np.stack([x, y, z], -1), r0


def _getR(L):
    R = rotm(L.get("yaw", 0), L.get("pitch", 0), L.get("roll", 0))
    if "rv" in L:
        cv2 = _cv2()
        R = cv2.Rodrigues(np.array(L["rv"], dtype=np.float64).reshape(3, 1))[0] @ R
    return R


def omni_project(rays, L):
    R = _getR(L)
    d = rays @ R
    n = np.linalg.norm(d, axis=-1)
    a, b, c = d[..., 0] / n, d[..., 1] / n, d[..., 2] / n
    den = c + L["xi"]
    safe = np.where(np.abs(den) < 1e-6, 1e-6, den)
    x = a / safe
    y = b / safe
    r2 = x * x + y * y
    rad = 1 + L["k1"] * r2 + L["k2"] * r2 * r2 + L["k3"] * r2 ** 3
    xd = x * rad + 2 * L["p1"] * x * y + L["p2"] * (r2 + 2 * x * x)
    yd = y * rad + L["p1"] * (r2 + 2 * y * y) + 2 * L["p2"] * x * y
    s = L["scale"]
    u = (L["fx"] * xd + L["cx"] - L["ox"]) * s
    v = (L["fy"] * yd + L["cy"] - L["oy"]) * s
    theta = np.arccos(np.clip(c, -1, 1))
    bad = den <= 0.05
    u = np.where(bad, -1e4, u)
    v = np.where(bad, -1e4, v)
    return u, v, theta


def equidistant_project(rays, L):
    R = rotm(L.get("yaw", 0), 0, 0)
    d = rays @ R
    a, b, c = d[..., 0], d[..., 1], d[..., 2]
    theta = np.arccos(np.clip(c, -1, 1))
    phi = np.arctan2(b, a)
    r = L["f"] * theta
    u = L["cx"] + r * np.cos(phi)
    v = L["cy"] + r * np.sin(phi)
    bad = theta > math.radians(L["half_fov"])
    return np.where(bad, -1e4, u), np.where(bad, -1e4, v), theta


def load_lens_models(lens_size: int, model: str = "x4", calib_path: Optional[Path] = None):
    """Two lens dicts in lens-image pixels for a lens image of lens_size px."""
    S = float(lens_size)
    if model == "generic":
        half = 100.0
        f = (S / 2) / math.radians(half)
        return [dict(kind="equi", f=f, cx=S / 2 - 0.5, cy=S / 2 - 0.5, half_fov=half, yaw=180 * k) for k in (0, 1)]
    path = calib_path or (assets_dir() / "calibration" / "insta360_x4.json")
    cal = json.loads(Path(path).read_text(encoding="utf-8"))
    ref = float(cal.get("reference_lens_size", 2880))
    x = cal["fit"]
    sc, dcx0, dcy0, dcx1, dcy1, r1, r2, r3 = x
    r = S / ref
    adj = float(cal.get("scale_adjust", 1.0))
    yaw_off = float(cal.get("yaw_offset_deg", 0.0))
    lenses = []
    for k, (dcx, dcy) in enumerate([(dcx0, dcy0), (dcx1, dcy1)]):
        base = dict(cal["lenses"][str(k)])
        L = dict(base, kind="omni", yaw=180 * k + yaw_off, pitch=0, roll=0)
        s_eff = sc * r * adj
        c0x = ((L["cx"] - 1000) * 0.48 + dcx) * r
        c0y = (L["cy"] * 0.48 + dcy) * r
        L["scale"] = s_eff
        L["ox"] = L["cx"] - c0x / s_eff
        L["oy"] = L["cy"] - c0y / s_eff
        lenses.append(L)
    lenses[1]["rv"] = [r1, r2, r3]
    return lenses


def _project(rays, L):
    return omni_project(rays, L) if L.get("kind") == "omni" else equidistant_project(rays, L)


class Stitcher:
    """Dual fisheye (hstacked, 2S x S, BGR) -> equirect (W x W/2), feathered seam.

    lat_range limits the rows (e.g. the court band for analysis) to save time.
    """

    def __init__(self, lens_size: int, width: int, model: str = "x4",
                 feather: float = 3.0, lat_range: Optional[tuple[float, float]] = None,
                 calib_path: Optional[Path] = None):
        cv2 = _cv2()
        self.S = lens_size
        self.W = int(width) // 2 * 2
        self.H = self.W // 2
        L0, L1 = load_lens_models(lens_size, model, calib_path)
        rays, self.row0 = eq_rays(self.W, self.H, lat_range)
        u0, v0, t0 = _project(rays, L0)
        u1, v1, t1 = _project(rays, L1)
        u1 = u1 + self.S  # lens 1 sits in the right half of the hstacked frame
        u1 = np.where(v1 < -1e3, -1e4, u1)
        w0 = np.clip(0.5 + np.degrees(t1 - t0) / (2 * feather), 0, 1)
        prim = w0 >= 0.5
        pu = np.where(prim, u0, u1).astype(np.float32)
        pv = np.where(prim, v0, v1).astype(np.float32)
        self.shape = pu.shape
        seam = (w0 > 0) & (w0 < 1)
        idx = np.flatnonzero(seam.ravel())
        if idx.size == 0:
            idx = np.array([0])
        n = (len(idx) + 1023) // 1024 * 1024
        idx = np.concatenate([idx, np.full(n - len(idx), idx[-1])])
        su = np.where(prim, u1, u0).ravel()[idx].astype(np.float32).reshape(-1, 1024)
        sv = np.where(prim, v1, v0).ravel()[idx].astype(np.float32).reshape(-1, 1024)
        wp = np.where(prim, w0, 1 - w0).ravel()[idx].astype(np.float32).reshape(-1, 1024, 1)
        self.m = cv2.convertMaps(pu, pv, cv2.CV_16SC2)
        self.s = cv2.convertMaps(su, sv, cv2.CV_16SC2)
        self.idx, self.wp = idx, wp

    def stitch(self, frame: np.ndarray) -> np.ndarray:
        cv2 = _cv2()
        out = cv2.remap(frame, self.m[0], self.m[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        sec = cv2.remap(frame, self.s[0], self.s[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        flat = out.reshape(-1, 3)
        prim = flat[self.idx].reshape(sec.shape).astype(np.float32)
        flat[self.idx] = (prim * self.wp + sec * (1 - self.wp) + 0.5).astype(np.uint8).reshape(-1, 3)
        return out


# ---------------------------------------------------------------------------
# Scene <-> image coordinates and the virtual camera
# ---------------------------------------------------------------------------

def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


class Panorama:
    """An equirect image (possibly a band of rows) with coordinate helpers."""

    def __init__(self, img: np.ndarray, full_height: int, row0: int = 0):
        self.img, self.H, self.row0 = img, full_height, row0
        self.W = img.shape[1]

    def to_scene(self, x, y):
        yaw = (np.asarray(x) + 0.5) / self.W * 360.0 - 180.0
        pitch = 90.0 - (np.asarray(y) + self.row0 + 0.5) / self.H * 180.0
        return yaw, pitch

    def to_px(self, yaw, pitch):
        x = (np.asarray(yaw) + 180.0) / 360.0 * self.W - 0.5
        y = (90.0 - np.asarray(pitch)) / 180.0 * self.H - 0.5 - self.row0
        return x, y

    @property
    def deg_per_px(self) -> float:
        return 360.0 / self.W


def persp_maps(yaw: float, pitch: float, hfov: float, out_w: int, out_h: int,
               eq_w: int, eq_h: int, row0: int = 0, step: int = 8):
    """Perspective view -> equirect lookup maps (float32), computed on a coarse grid."""
    cv2 = _cv2()
    gw, gh = max(2, out_w // step + 2), max(2, out_h // step + 2)
    f = (out_w / 2.0) / math.tan(math.radians(hfov) / 2)
    xs = (np.linspace(0, out_w, gw) - out_w / 2.0) / f
    ys = (np.linspace(0, out_h, gh) - out_h / 2.0) / f
    X, Y = np.meshgrid(xs, ys)
    Z = np.ones_like(X)
    p = math.radians(pitch)
    # Look up by `pitch` (y axis points down in this convention)
    y1 = Y * math.cos(p) - Z * math.sin(p)
    z1 = Y * math.sin(p) + Z * math.cos(p)
    x1 = X
    n = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    lon_rel = np.arctan2(x1, z1)
    lat = np.arcsin(np.clip(-y1 / n, -1, 1))
    u = (np.degrees(lon_rel) + yaw + 180.0) / 360.0 * eq_w - 0.5
    v = (90.0 - np.degrees(lat)) / 180.0 * eq_h - 0.5 - row0
    # Grid point k sits at output pixel k*out_w/(gw-1); resize to the exact pixel grid.
    mu = cv2.resize(u.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    mv = cv2.resize(v.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return mu, mv


def render_view(pano_img: np.ndarray, yaw: float, pitch: float, hfov: float,
                out_w: int, out_h: int, full_h: Optional[int] = None, row0: int = 0) -> np.ndarray:
    cv2 = _cv2()
    eq_h = full_h or pano_img.shape[0]
    mu, mv = persp_maps(yaw, pitch, hfov, out_w, out_h, pano_img.shape[1], eq_h, row0)
    return cv2.remap(pano_img, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def flat_to_scene(x, y, w: int, h: int):
    yaw = (np.asarray(x) / w - 0.5) * FLAT_HFOV
    pitch = -(np.asarray(y) / h - 0.5) * FLAT_HFOV * h / w
    return yaw, pitch


def flat_to_px(yaw, pitch, w: int, h: int):
    x = (np.asarray(yaw) / FLAT_HFOV + 0.5) * w
    y = (-np.asarray(pitch) / (FLAT_HFOV * h / w) + 0.5) * h
    return x, y


def render_flat_view(img: np.ndarray, yaw: float, pitch: float, hfov: float,
                     out_w: int, out_h: int) -> np.ndarray:
    """Crop-and-scale "virtual camera" for flat video (no reprojection)."""
    cv2 = _cv2()
    h, w = img.shape[:2]
    cw = w * min(1.0, hfov / FLAT_HFOV)
    ch = cw * out_h / out_w
    if ch > h:
        ch = h
        cw = ch * out_w / out_h
    cx, cy = flat_to_px(yaw, pitch, w, h)
    cx = float(np.clip(cx, cw / 2, w - cw / 2))
    cy = float(np.clip(cy, ch / 2, h - ch / 2))
    s = out_w / cw
    M = np.array([[s, 0, -(cx - cw / 2) * s], [0, s, -(cy - ch / 2) * s]], dtype=np.float32)
    interp = cv2.INTER_CUBIC if s > 1.05 else cv2.INTER_AREA
    return cv2.warpAffine(img, M, (out_w, out_h), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def flat_view_limits(hfov: float, out_w: int, out_h: int, w: int, h: int):
    """Allowed yaw/pitch ranges for the crop centre (pseudo-degrees)."""
    cw = w * min(1.0, hfov / FLAT_HFOV)
    ch = cw * out_h / out_w
    if ch > h:
        ch = h
        cw = ch * out_w / out_h
    y0, _ = flat_to_scene(cw / 2, 0, w, h)
    y1, _ = flat_to_scene(w - cw / 2, 0, w, h)
    _, p0 = flat_to_scene(0, h - ch / 2, w, h)
    _, p1 = flat_to_scene(0, ch / 2, w, h)
    return (float(y0), float(y1)), (float(p0), float(p1))


class SceneRenderer:
    """Turns decoded frames into virtual-camera views for any source kind."""

    def __init__(self, info: SourceInfo, plan: DecodePlan, lens_model: str = "x4",
                 eq_width: Optional[int] = None, lat_range: Optional[tuple[float, float]] = None):
        self.info, self.plan = info, plan
        self.stab = info.stab
        self.stitcher: Optional[Stitcher] = None
        self.out_rows = 0
        if info.is_fisheye:
            w = eq_width or min(2 * plan.lens_size * 2 // 2, 7680)
            stitch_range = lat_range
            if self.stab is not None and lat_range is not None:
                lo, hi = lat_range
                stitch_range = (max(-85.0, lo - 25.0), min(85.0, hi + 25.0))
            self.stitcher = Stitcher(plan.lens_size, w, lens_model, lat_range=stitch_range)
            self.eq_w, self.eq_h = self.stitcher.W, self.stitcher.H
            if self.stab is not None and lat_range is not None:
                lo, hi = lat_range
                self.row0 = max(0, int(math.floor((90 - hi) / 180 * self.eq_h)))
                r1 = min(self.eq_h, int(math.ceil((90 - lo) / 180 * self.eq_h)))
                self.out_rows = max(1, r1 - self.row0)
            else:
                self.row0 = self.stitcher.row0
                self.out_rows = self.stitcher.shape[0]
        elif info.kind == "equirect":
            self.eq_w, self.eq_h, self.row0 = plan.width, plan.height, 0
        else:
            self.eq_w = self.eq_h = self.row0 = 0

    def panorama(self, frame: np.ndarray, t: float = 0.0) -> np.ndarray:
        if self.stitcher is None:
            return frame
        img = self.stitcher.stitch(frame)
        if self.stab is None:
            return img
        return self.stab.level_band(img, self.stitcher.row0, self.eq_h, self.row0, self.out_rows, t)

    def view(self, pano: np.ndarray, yaw: float, pitch: float, hfov: float, out_w: int, out_h: int) -> np.ndarray:
        if self.info.kind == "flat":
            return render_flat_view(pano, yaw, pitch, hfov, out_w, out_h)
        return render_view(pano, yaw, pitch, hfov, out_w, out_h, self.eq_h, self.row0)


class DirectFisheyeView:
    """Virtual camera straight from the two fisheye images (no full stitch).

    The lens coordinates are computed on a coarse grid and interpolated, then
    both lenses are sampled and feather-blended. Much faster than stitching a
    whole 360 frame first, and sharper (one resampling step).
    """

    def __init__(self, lens_size: int, model: str = "x4", feather: float = 3.0,
                 calib_path: Optional[Path] = None):
        self.S = lens_size
        self.L0, self.L1 = load_lens_models(lens_size, model, calib_path)
        self.feather = feather

    def maps(self, yaw: float, pitch: float, hfov: float, out_w: int, out_h: int, step: int = 8,
             to_body: Optional[np.ndarray] = None):
        cv2 = _cv2()
        gw, gh = max(2, out_w // step + 2), max(2, out_h // step + 2)
        f = (out_w / 2.0) / math.tan(math.radians(hfov) / 2)
        xs = (np.linspace(0, out_w, gw) - out_w / 2.0) / f
        ys = (np.linspace(0, out_h, gh) - out_h / 2.0) / f
        X, Y = np.meshgrid(xs, ys)
        Z = np.ones_like(X)
        rays = np.stack([X, Y, Z], -1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        R = rotm(yaw, 0, 0) @ np.array(
            [[1, 0, 0], [0, math.cos(math.radians(pitch)), -math.sin(math.radians(pitch))],
             [0, math.sin(math.radians(pitch)), math.cos(math.radians(pitch))]])
        world = rays @ R.T
        if to_body is not None:
            world = world @ to_body
        u0, v0, t0 = _project(world, self.L0)
        u1, v1, t1 = _project(world, self.L1)
        u1 = np.where(v1 < -1e3, -64.0, u1 + self.S)
        u0 = np.where(v0 < -1e3, -64.0, u0)
        v0 = np.where(v0 < -1e3, -64.0, v0)
        v1 = np.where(v1 < -1e3, -64.0, v1)
        w0 = np.clip(0.5 + np.degrees(t1 - t0) / (2 * self.feather), 0, 1)

        def up(a):
            return cv2.resize(a.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return up(u0), up(v0), up(u1), up(v1), up(w0)

    def view(self, frame: np.ndarray, yaw: float, pitch: float, hfov: float, out_w: int, out_h: int,
             to_body: Optional[np.ndarray] = None) -> np.ndarray:
        cv2 = _cv2()
        u0, v0, u1, v1, w0 = self.maps(yaw, pitch, hfov, out_w, out_h, to_body=to_body)
        wmin, wmax = float(w0.min()), float(w0.max())
        if wmin >= 0.999:
            return cv2.remap(frame, u0, v0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if wmax <= 0.001:
            return cv2.remap(frame, u1, v1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a = cv2.remap(frame, u0, v0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        b = cv2.remap(frame, u1, v1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return cv2.blendLinear(a, b, w0, (1.0 - w0).astype(np.float32))


class ClipRenderer:
    """Full-resolution virtual camera for the final render (any source kind)."""

    def __init__(self, info: SourceInfo, plan: DecodePlan, lens_model: str = "x4"):
        self.info = info
        self.stab = info.stab
        self.direct = DirectFisheyeView(plan.lens_size, lens_model) if info.is_fisheye else None

    def view(self, frame: np.ndarray, yaw: float, pitch: float, hfov: float, out_w: int, out_h: int,
             t: Optional[float] = None) -> np.ndarray:
        to_body = self.stab.to_body(t) if self.stab is not None and t is not None else None
        if self.direct is not None:
            return self.direct.view(frame, yaw, pitch, hfov, out_w, out_h, to_body=to_body)
        if self.info.kind == "equirect":
            return render_view(frame, yaw, pitch, hfov, out_w, out_h)
        return render_flat_view(frame, yaw, pitch, hfov, out_w, out_h)
ANALYSIS_BAND_PICK = (-30.0, 50.0)
