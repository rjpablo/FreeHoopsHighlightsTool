"""Horizon and shake stabilization from an Insta360 .lrv next to the .insv.

The .lrv is the low-resolution proxy the camera writes beside the main file.
Its trailer holds the motion samples. When that file is present, the view is
kept level and steady: highlight angles and the rendered picture use the same
leveled frame. Without an .lrv, nothing about the picture changes.
"""

from __future__ import annotations

import math
import re
import struct
from pathlib import Path
from typing import Optional

import numpy as np

TAIL = 78
MAGIC = b"8db42d694ccc418790edff439fe026bf"
_CAPTURE = re.compile(
    r"^(?:PRO_)?(?:VID|LRV)_(\d{8}_\d{6})_(00|10|01|11)_(\d+)$", re.IGNORECASE)
_LOOSE = re.compile(r"(\d{8}_\d{6}).*_(\d+)$")

# Gyroflow's IMU axis order: X4/X5 are "yzX", X3 (and unknown 360 files) are "Xyz".
_ORIENT = {"X5": "yzX", "X4": "yzX", "X3": "Xyz"}


def find_lrv(path: Path) -> Optional[Path]:
    """The proxy file for this .insv, in the same folder, or None."""
    path = Path(path)
    if path.suffix.lower() == ".lrv" or not path.parent.is_dir():
        return None
    m = _CAPTURE.match(path.stem)
    names: list[str] = []
    if m:
        date, seq = m.group(1), m.group(3)
        for code in ("11", "01"):
            for prefix in ("LRV", "PRO_LRV"):
                for ext in (".lrv", ".insv", ".mp4"):
                    names.append(f"{prefix}_{date}_{code}_{seq}{ext}")
    else:
        loose = _LOOSE.search(path.stem)
        if loose:
            date, seq = loose.group(1), loose.group(2)
            for code in ("11", "01"):
                names.append(f"LRV_{date}_{code}_{seq}.lrv")
    if not names:
        return None
    try:
        have = {p.name.lower(): p for p in path.parent.iterdir() if p.is_file()}
    except OSError:
        return None
    for name in names:
        hit = have.get(name.lower())
        if hit is not None and hit.resolve() != path.resolve():
            return hit
    return None


def load_stabilizer(path: Path, duration: float = 0.0):
    """Motion from an .lrv, or None when the file has nothing usable."""
    try:
        meta, imu = _read_trailer(Path(path))
    except Exception:
        return None
    if imu is None:
        return None
    ts, acc, gyro = imu
    if len(ts) < 20:
        return None
    orient = _ORIENT.get(meta["family"], "Xyz")
    acc = _orient(acc, orient)
    gyro = _orient(gyro, orient)
    n = float(np.median(np.linalg.norm(acc, axis=1)))
    if 7.0 < n < 12.0:
        acc = acc / 9.80665
        n = float(np.median(np.linalg.norm(acc, axis=1)))
    if not 0.5 < n < 3.0:
        return None
    acc = acc / n
    # The stitched picture is already upright. A 90 degree axis mismatch
    # would otherwise pitch the court onto its side, so rims stand vertical.
    acc, gyro = _match_image_up(acc, gyro)
    gyro = gyro - np.median(gyro, axis=0)
    t0 = meta["t0_us"]
    if t0 is None:
        t0 = float(ts[0])
    stab = _integrate(ts, acc, gyro, t0)
    if duration > 0:
        inside = (stab.times >= -0.5) & (stab.times <= duration + 0.5)
        if int(inside.sum()) < 10:
            return None
    return stab


class Stabilizer:
    """Maps a leveled, steady direction to the camera direction at time t."""

    def __init__(self, times: np.ndarray, quats: np.ndarray):
        self.times = np.asarray(times, np.float64)
        self.quats = np.asarray(quats, np.float64)

    def to_body(self, t: float) -> np.ndarray:
        """Matrix M with body_row = stable_row @ M."""
        return _quat_matrix(_quat_at(self.times, self.quats, float(t)))

    def level_band(self, band: np.ndarray, src_row0: int, full_h: int,
                   dst_row0: int, dst_rows: int, t: float) -> np.ndarray:
        """Rebuild an equirect band so its rows are level at time t."""
        import cv2
        W = band.shape[1]
        step = 8
        ys = np.linspace(0, max(dst_rows - 1, 1), max(2, dst_rows // step))
        xs = np.linspace(0, max(W - 1, 1), max(2, W // step))
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        rays = _equirect_rays(xx, yy, W, full_h, dst_row0) @ self.to_body(t)
        lon = np.arctan2(rays[..., 0], rays[..., 2])
        lat = np.arcsin(np.clip(-rays[..., 1], -1.0, 1.0))
        u = (np.degrees(lon) + 180.0) / 360.0 * W - 0.5
        v = (90.0 - np.degrees(lat)) / 180.0 * full_h - 0.5 - src_row0
        mu = cv2.resize(u.astype(np.float32), (W, dst_rows), interpolation=cv2.INTER_LINEAR)
        mv = cv2.resize(v.astype(np.float32), (W, dst_rows), interpolation=cv2.INTER_LINEAR)
        return cv2.remap(band, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _equirect_rays(xx, yy, width: int, full_h: int, row0: int) -> np.ndarray:
    lon = (xx + 0.5) / width * 2 * np.pi - np.pi
    lat = np.pi / 2 - (yy + row0 + 0.5) / full_h * np.pi
    return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], -1)


def _orient(v: np.ndarray, code: str) -> np.ndarray:
    out = np.empty_like(v)
    for i, ch in enumerate(code):
        src = {"x": 0, "y": 1, "z": 2}[ch.lower()]
        sign = -1.0 if ch.islower() else 1.0
        out[..., i] = sign * v[..., src]
    return out


def _quarter_turns():
    """The 24 right-angle rotations (det +1)."""
    rots = []
    for perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)):
        sign = 1.0 if (perm[1] - perm[0]) % 3 == 1 else -1.0
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                sz = sign * sx * sy
                R = np.zeros((3, 3))
                R[0, perm[0]] = sx
                R[1, perm[1]] = sy
                R[2, perm[2]] = sz
                rots.append(R)
    return rots


def _match_image_up(acc: np.ndarray, gyro: np.ndarray):
    """Turn both streams so the average acceleration points near image up.

    Image up is (0, -1, 0). The labels from the camera can be a quarter turn
    off the stitched picture. Leaving that mismatch in place turns the court
    on its side, so each rim stands vertical. A tilt under 45 degrees is not
    touched here, so the leveling filter can still straighten a crooked tripod.
    """
    g = np.median(acc, axis=0).astype(np.float64)
    n = float(np.linalg.norm(g))
    if n < 1e-6:
        return acc, gyro
    g /= n
    target = np.array([0.0, -1.0, 0.0])
    if float(np.dot(g, target)) >= math.cos(math.radians(45.0)):
        return acc, gyro
    best_R, best_score = None, -1e9
    for R in _quarter_turns():
        aligned = float(np.dot(R @ g, target))
        if aligned < math.cos(math.radians(45.0)):
            continue
        score = aligned * 8.0 + float(np.trace(R))
        if score > best_score:
            best_score = score
            best_R = R
    if best_R is None:
        return acc, gyro
    return acc @ best_R.T, gyro @ best_R.T


def _integrate(ts_us: np.ndarray, acc: np.ndarray, gyro: np.ndarray, t0_us: float,
               tau: float = 2.0) -> Stabilizer:
    up0 = np.median(acc, axis=0)
    up0 = up0 / np.linalg.norm(up0)
    q = _q_from_to(up0, np.array([0.0, -1.0, 0.0]))
    times = []
    quats = []
    prev = float(ts_us[0])
    for i in range(len(ts_us)):
        dt = (float(ts_us[i]) - prev) * 1e-6
        prev = float(ts_us[i])
        if 0 < dt <= 2.0:
            steps = max(1, int(math.ceil(dt / 0.02)))
            h = dt / steps
            for _ in range(steps):
                q = _qmul(q, _dq(gyro[i], h))
            up = acc[i]
            n = float(np.linalg.norm(up))
            if 0.7 < n < 1.3:
                pred = _qrot(_qinv(q), np.array([0.0, -1.0, 0.0]))
                # corr takes the predicted up onto the measured up. Applying it
                # forward would grow the error, so the filter uses the opposite turn.
                corr = _q_from_to(pred, up / n)
                a = dt / (dt + tau)
                q = _qnorm(_qmul(q, _qinv(_qscale(corr, a))))
        t = (float(ts_us[i]) - t0_us) * 1e-6
        if t < -2.0:
            continue
        times.append(t)
        quats.append(q.copy())
    if len(times) < 2:
        times = [0.0, 1.0]
        quats = [q, q]
    return Stabilizer(np.array(times), np.array(quats))


def _quat_at(times: np.ndarray, quats: np.ndarray, t: float) -> np.ndarray:
    i = int(np.searchsorted(times, t, side="right") - 1)
    if i < 0:
        return quats[0]
    if i >= len(times) - 1:
        return quats[-1]
    span = float(times[i + 1] - times[i])
    a = 0.0 if span <= 0 else (t - float(times[i])) / span
    return _slerp(quats[i], quats[i + 1], a)


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qnorm(q):
    n = float(np.linalg.norm(q))
    return q if n < 1e-12 else q / n


def _qrot(q, v):
    u = q[:3]
    return 2 * np.dot(u, v) * u + (q[3] * q[3] - np.dot(u, u)) * v + 2 * q[3] * np.cross(u, v)


def _q_from_to(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -0.999999:
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, axis)
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    return _qnorm(np.array([c[0], c[1], c[2], 1.0 + d]))


def _dq(omega, dt):
    ang = float(np.linalg.norm(omega) * dt)
    if ang < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = omega / np.linalg.norm(omega)
    s = math.sin(ang / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(ang / 2)])


def _qscale(q, a):
    """Short rotation scaled by a in [0, 1] (a=0 is no turn)."""
    q = _qnorm(q)
    if q[3] < 0:
        q = -q
    half = math.acos(float(np.clip(q[3], -1, 1))) * a
    s = math.sin(half)
    n = float(np.linalg.norm(q[:3]))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = q[:3] / n
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)])


def _slerp(a, b, t):
    a = _qnorm(a)
    b = _qnorm(b)
    d = float(np.dot(a, b))
    if d < 0:
        b = -b
        d = -d
    if d > 0.9995:
        return _qnorm(a + t * (b - a))
    th = math.acos(float(np.clip(d, -1, 1)))
    s = math.sin(th)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / s


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = _qnorm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------------------
# Insta360 trailer
# ---------------------------------------------------------------------------

def _read_trailer(path: Path):
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if size < TAIL:
            raise ValueError("too small")
        f.seek(size - TAIL)
        tail = f.read(TAIL)
        if tail[46:] != MAGIC:
            raise ValueError("no trailer")
        length = struct.unpack_from("<I", tail, 38)[0]
        if length < TAIL or length > size:
            raise ValueError("bad trailer length")
        start = size - length
        records = _index_records(f, start, length)
        meta_at = records.get(0x101)
        if meta_at is None:
            raise ValueError("no metadata")
        meta = _parse_metadata(_read_at(f, meta_at[0], meta_at[1], 1 << 20))
        imu_at = records.get(0x300)
        imu = None
        if imu_at is not None:
            imu = _read_imu(f, imu_at[0], imu_at[1], meta)
        return meta, imu


def _read_at(f, offset: int, length: int, limit: int) -> bytes:
    if length <= 0 or length > limit:
        raise ValueError("record too large")
    f.seek(offset)
    buf = f.read(length)
    if len(buf) != length:
        raise ValueError("short read")
    return buf


def _index_records(f, start: int, length: int) -> dict:
    records = {}
    header = length - TAIL
    for _ in range(256):
        if header < 6:
            break
        f.seek(start + header)
        id_, size = struct.unpack("<HI", f.read(6))
        if size == 0 or size > header:
            break
        if id_ == 0:
            table = _read_at(f, start + header - size, size, 64 << 10)
            for i in range(0, len(table) - 9, 10):
                rec_size, rel = struct.unpack_from("<II", table, i + 2)
                if rec_size and rel + rec_size + 6 <= length:
                    f.seek(start + rel + rec_size)
                    full = struct.unpack("<H", f.read(2))[0]
                    records[full] = (start + rel, rec_size)
            break
        records[id_] = (start + header - size, size)
        header -= size + 6
    return records


def _parse_metadata(buf: bytes) -> dict:
    meta = {"t0_us": None, "acc_range": 16.0, "gyro_range": 2000.0, "delay_us": 0.0, "family": ""}
    text = buf.decode("latin1", errors="ignore")
    for name in ("X5", "X4", "X3"):
        if name in text:
            meta["family"] = name
            break
    for field, value in _proto(buf):
        if field == 24 and isinstance(value, int):
            meta["t0_us"] = float(value)
        elif field == 28 and isinstance(value, float):
            meta["delay_us"] = value * 1000.0
        elif field == 65 and isinstance(value, bytes):
            for sub, num in _proto(value):
                if sub == 1 and isinstance(num, int) and num > 0:
                    meta["acc_range"] = float(num)
                elif sub == 2 and isinstance(num, int) and num > 0:
                    meta["gyro_range"] = float(num)
    if meta["t0_us"] is not None:
        meta["t0_us"] += meta["delay_us"]
    return meta


def _proto(buf: bytes):
    i, n = 0, len(buf)
    while i < n:
        tag, i = _varint(buf, i)
        if tag is None:
            return
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, i = _varint(buf, i)
            if val is None:
                return
            yield field, val
        elif wire == 1 and i + 8 <= n:
            yield field, struct.unpack_from("<d", buf, i)[0]
            i += 8
        elif wire == 5 and i + 4 <= n:
            i += 4
        elif wire == 2:
            length, i = _varint(buf, i)
            if length is None or i + length > n:
                return
            yield field, buf[i:i + length]
            i += length
        else:
            return


def _varint(buf: bytes, i: int):
    result, shift = 0, 0
    while i < len(buf) and shift < 70:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if b < 128:
            return result, i
        shift += 7
    return None, i


def _read_imu(f, offset: int, length: int, meta: dict):
    f.seek(offset)
    peek = f.read(min(length, 56 * 8))
    sample = _sample_bytes(peek, length)
    if sample == 0:
        return None
    count = length // sample
    if count < 20 or count * sample > 400 << 20:
        return None
    f.seek(offset)
    buf = f.read(count * sample)
    if len(buf) != count * sample:
        return None
    step = _stride(buf, sample)
    if sample == 20:
        rec = np.dtype([("ts", "<u8"), ("raw", "<u2", (6,))])
        rows = np.frombuffer(buf, dtype=np.uint8).copy().view(rec)[::step]
        raw = rows["raw"].astype(np.float64)
        keep = np.all(raw != 0, axis=1)
        rows, raw = rows[keep], raw[keep]
        if len(rows) < 20:
            return None
        acc = (raw[:, 0:3] - 32768.0) / 32768.0 * meta["acc_range"]
        gyro = np.radians((raw[:, 3:6] - 32768.0) / 32768.0 * meta["gyro_range"])
        return rows["ts"].astype(np.float64), acc, gyro
    rec = np.dtype([("ts", "<u8"), ("g", "<f8", (3,)), ("a", "<f8", (3,))])
    rows = np.frombuffer(buf, dtype=np.uint8).copy().view(rec)[::step]
    if len(rows) < 20:
        return None
    return rows["ts"].astype(np.float64), rows["a"].astype(np.float64), rows["g"].astype(np.float64)


def _stride(buf: bytes, sample: int) -> int:
    n = min(64, len(buf) // sample)
    ts = np.array([struct.unpack_from("<Q", buf, i * sample)[0] for i in range(n)], np.float64)
    dt = float(np.median(np.diff(ts)))
    rate = 1e6 / dt if dt > 0 else 1000.0
    return max(1, int(round(rate / 100.0)))


def _sample_bytes(peek: bytes, length: int) -> int:
    def delta(step: int) -> float:
        if len(peek) < step * 3:
            return -1
        t = [struct.unpack_from("<Q", peek, i * step)[0] for i in range(3)]
        d = [t[i + 1] - t[i] for i in range(2)]
        if min(d) <= 0:
            return -1
        return float(np.median(d))

    if length % 20 == 0:
        d = delta(20)
        if 200 < d < 20000:
            return 20
    if length % 56 == 0:
        d = delta(56)
        if 200 < d < 20000:
            return 56
    return 20 if length % 20 == 0 else 0
