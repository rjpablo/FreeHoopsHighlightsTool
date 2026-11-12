"""Virtual-camera planning (the v2 auto-zoom logic) and keyframe interpolation.

Per clip, at 10 Hz:
  * ball track: detections gated by a constant-velocity prediction
    (Kalman-style), gaps up to 1.5 s filled by interpolation
  * players: detections minus stationary bystanders (bench, spectators)
  * target = 75% ball + 25% player cluster (players only when no ball)
  * shots: hold the shooter (with the rim if both fit) through the release,
    then pan to the rim and hold through the basket
  * smoothing: 6 deg dead zone, max 75 deg/s pan (150 deg/s for the pan to
    the rim after a long shot), acceleration limit, tilt -25..+10 deg
Keyframes every 0.5 s go to the camera CSV; the renderer interpolates them.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from fhh_detect import Detections, Hoop
from fhh_project import Clip, Keyframe
from fhh_source import wrap180

DEAD_ZONE = 6.0
MAX_PAN = 75.0
SHOT_PAN = 150.0
TILT_MIN, TILT_MAX = -25.0, 10.0
BALL_WEIGHT = 0.75
BASE_HFOV_169 = 85.0
BASE_HFOV_916 = 52.0
FLAT_ZOOM_169 = 1.5
ZOOM_MAX = 1.8


def zoom_limit(flat: bool) -> float:
    """Flat video is already a 1.5x crop, so the extra zoom stays under 1.8x total."""
    if flat:
        return ZOOM_MAX / FLAT_ZOOM_169
    return ZOOM_MAX


def _ease(u: float) -> float:
    u = 0.0 if u < 0.0 else 1.0 if u > 1.0 else u
    return u * u * (3.0 - 2.0 * u)


def animated_zoom(t: float, clip: Clip, flat: bool = False, fit: Optional[float] = None,
                  release: Optional[float] = None) -> float:
    """Ease from a wide view into the shot, hold through the basket, then ease back.

    The curve is smooth (no steps). `fit` is an optional tighter cap so a long
    shot can still show the shooter and the rim together.
    """
    cap = zoom_limit(flat)
    if fit is not None:
        cap = min(cap, max(1.0, float(fit)))
    event = clip.event_time if clip.event_time else (clip.start + clip.end) / 2.0
    rel = release if release is not None else clip.release
    if rel is None:
        rel = max(clip.start, event - 1.2)
    rel = min(max(rel, clip.start), event)
    peak = cap if clip.type in ("made", "attempt") else min(cap, 1.0 + 0.45 * (cap - 1.0))
    follow = 1.0 + 0.4 * (peak - 1.0)
    if t <= rel:
        span = max(0.5, rel - clip.start)
        return 1.0 + (follow - 1.0) * _ease((t - clip.start) / span)
    if t <= event:
        span = max(0.35, event - rel)
        return follow + (peak - follow) * _ease((t - rel) / span)
    hold = min(clip.end, event + 0.8)
    if t <= hold:
        return peak
    span = max(0.45, clip.end - hold)
    tail = min(follow, 1.0 + 0.25 * (peak - 1.0))
    return peak + (tail - peak) * _ease((t - hold) / span)


def zooms_are_flat(keys: list[Keyframe]) -> bool:
    """True when every keyframe is the old default of 1.0 (zoom was never planned)."""
    return bool(keys) and all(abs(k.zoom - 1.0) < 0.03 for k in keys)


def circ_mean(yaws, weights=None) -> float:
    a = np.radians(np.asarray(yaws, np.float64))
    w = np.ones_like(a) if weights is None else np.asarray(weights, np.float64)
    return float(np.degrees(np.arctan2((np.sin(a) * w).sum(), (np.cos(a) * w).sum())))


def blend_yaw(a: float, b: float, wa: float) -> float:
    return float(wrap180(b + wrap180(a - b) * wa))


def ball_track(B: np.ndarray, ts: np.ndarray, max_gap: float = 1.5):
    """Gated ball positions at times ts. Returns yaw, pitch, valid."""
    yaw = np.full(len(ts), np.nan)
    pitch = np.full(len(ts), np.nan)
    if len(B) == 0:
        return yaw, pitch, np.zeros(len(ts), bool)
    B = B[np.argsort(B[:, 0])]
    pts: list[tuple[float, float, float]] = []
    state = None  # (t, yaw, pitch, vy, vp)
    for t in np.unique(B[:, 0]):
        cand = B[B[:, 0] == t]
        if state is None:
            best = cand[np.argmax(cand[:, 3])]
            if best[3] < 0.2:
                continue
            state = (t, best[1], best[2], 0.0, 0.0)
            pts.append((t, best[1], best[2]))
            continue
        t0, y0, p0, vy, vp = state
        dt = t - t0
        if dt > max_gap * 2:
            best = cand[np.argmax(cand[:, 3])]
            if best[3] < 0.2:
                continue
            state = (t, best[1], best[2], 0.0, 0.0)
            pts.append((t, best[1], best[2]))
            continue
        py, pp = y0 + vy * dt, p0 + vp * dt
        gate = 6.0 + 45.0 * dt
        d = np.hypot(wrap180(cand[:, 1] - py), cand[:, 2] - pp)
        ok = d < gate
        if not ok.any():
            continue
        score = cand[:, 3] - d / (gate * 2)
        score[~ok] = -9
        best = cand[int(np.argmax(score))]
        ny, np_ = best[1], best[2]
        nvy = 0.5 * vy + 0.5 * wrap180(ny - y0) / max(dt, 1e-3)
        nvp = 0.5 * vp + 0.5 * (np_ - p0) / max(dt, 1e-3)
        nvy, nvp = float(np.clip(nvy, -120, 120)), float(np.clip(nvp, -80, 80))
        state = (t, ny, np_, nvy, nvp)
        pts.append((t, ny, np_))
    if not pts:
        return yaw, pitch, np.zeros(len(ts), bool)
    P = np.array(pts)
    # unwrap yaw for interpolation
    uy = np.degrees(np.unwrap(np.radians(P[:, 1])))
    for i, t in enumerate(ts):
        j = np.searchsorted(P[:, 0], t)
        if j < len(P) and abs(P[j, 0] - t) < 0.06:
            yaw[i], pitch[i] = P[j, 1], P[j, 2]
            continue
        if 0 < j < len(P) and P[j, 0] - P[j - 1, 0] <= max_gap:
            a = (t - P[j - 1, 0]) / (P[j, 0] - P[j - 1, 0])
            yaw[i] = wrap180(uy[j - 1] + a * (uy[j] - uy[j - 1]))
            pitch[i] = P[j - 1, 2] + a * (P[j, 2] - P[j - 1, 2])
        elif j > 0 and t - P[j - 1, 0] <= 0.3:
            yaw[i], pitch[i] = P[j - 1, 1], P[j - 1, 2]
        elif j < len(P) and P[j, 0] - t <= 0.3:
            yaw[i], pitch[i] = P[j, 1], P[j, 2]
    return yaw, pitch, ~np.isnan(yaw)


def filter_bystanders(P: np.ndarray, cells: list) -> np.ndarray:
    if len(P) == 0 or not cells:
        return P
    keep = np.ones(len(P), bool)
    for cy, cp in cells:
        keep &= ~((np.abs(wrap180(P[:, 1] - cy)) < 1.6) & (np.abs(P[:, 2] - cp) < 1.6))
    return P[keep]


def estimate_release(t_event: float, hoop: Hoop, B: np.ndarray, P: np.ndarray):
    """(release_time, shooter_yaw, shooter_pitch) from the ball's path to the rim, or None."""
    sel = B[(B[:, 0] >= t_event - 3.5) & (B[:, 0] <= t_event - 0.15)] if len(B) else B
    if len(sel) < 2:
        return None
    sel = sel[np.argsort(-sel[:, 0])]
    dist = np.hypot(wrap180(sel[:, 1] - hoop.yaw), sel[:, 2] - hoop.pitch)
    near = np.flatnonzero(dist < max(8.0, 6 * hoop.width))
    if len(near) == 0:
        return None
    cur = near[0]
    chain = [cur]
    for i in range(cur + 1, len(sel)):
        gap = sel[chain[-1], 0] - sel[i, 0]
        if gap <= 1e-6:
            continue
        if gap > 0.8:
            break
        step = math.hypot(float(wrap180(sel[i, 1] - sel[chain[-1], 1])), float(sel[i, 2] - sel[chain[-1], 2]))
        if dist[i] > dist[chain[-1]] - 0.3 and step < 4 + 45 * gap:
            chain.append(i)
    if len(chain) < 2:
        return None
    r = sel[chain[-1]]
    rt, ry, rp = float(r[0]), float(r[1]), float(r[2])
    if len(P):
        pp = P[np.abs(P[:, 0] - rt) < 0.4]
        if len(pp):
            d = np.hypot(wrap180(pp[:, 1] - ry), pp[:, 2] - rp)
            k = int(np.argmin(d))
            if d[k] < 8:
                return rt, float(pp[k, 1]), float(pp[k, 2])
    return rt, ry, rp - 4.0


def plan_clip(clip: Clip, d: Detections, hoops: list[Hoop], bystanders: list,
              base_hfov: float, step: float = 0.1, flat: bool = False) -> tuple[list[Keyframe], Optional[float]]:
    ts = np.arange(clip.start, clip.end + 1e-6, step)
    hoop = next((h for h in hoops if h.name == clip.hoop), hoops[0] if hoops else None)
    w0, w1 = clip.start - 1.5, clip.end + 1.5
    P = d.persons[(d.persons[:, 0] >= w0) & (d.persons[:, 0] <= w1)] if len(d.persons) else d.persons
    P = filter_bystanders(P, bystanders)
    B = d.balls[(d.balls[:, 0] >= w0) & (d.balls[:, 0] <= w1)] if len(d.balls) else d.balls
    by, bp, bvalid = ball_track(B, ts)
    ptimes = np.unique(P[:, 0]) if len(P) else np.zeros(0)

    ref = hoop.yaw if hoop else 0.0
    ty = np.zeros(len(ts))
    tp = np.zeros(len(ts))
    for i, t in enumerate(ts):
        pyaw = ppitch = None
        if len(ptimes):
            k = int(np.argmin(np.abs(ptimes - t)))
            if abs(ptimes[k] - t) < 0.6:
                pp = P[P[:, 0] == ptimes[k]]
                center = by[i] if bvalid[i] else ref
                near = pp[np.abs(wrap180(pp[:, 1] - center)) < 45]
                if len(near):
                    pyaw = circ_mean(near[:, 1], near[:, 4])
                    ppitch = float(np.average(near[:, 2], weights=near[:, 4]))
        if bvalid[i] and pyaw is not None:
            ty[i] = blend_yaw(by[i], pyaw, BALL_WEIGHT)
            tp[i] = BALL_WEIGHT * bp[i] + (1 - BALL_WEIGHT) * ppitch
        elif bvalid[i]:
            ty[i], tp[i] = by[i], bp[i]
        elif pyaw is not None:
            ty[i], tp[i] = pyaw, ppitch
        else:
            ty[i], tp[i] = ref, (hoop.pitch - 10 if hoop else -5)
        ref = ty[i]

    fast = np.zeros(len(ts), bool)
    release = clip.release
    shooter = None
    if hoop is not None and clip.type in ("made", "attempt"):
        est = estimate_release(clip.event_time, hoop, B, P)
        shooter = None
        if est is not None:
            release, sy, sp = est
            shooter = (sy, sp)
        elif release is None:
            release = clip.event_time - 1.2
        for i, t in enumerate(ts):
            if shooter is not None and t <= release + 0.25:
                sep = abs(float(wrap180(shooter[0] - hoop.yaw)))
                if sep <= 0.6 * base_hfov:
                    ty[i] = blend_yaw(shooter[0], hoop.yaw, 0.5)
                    tp[i] = 0.5 * (shooter[1] + hoop.pitch) - 3
                else:
                    ty[i], tp[i] = shooter[0], shooter[1] + 2
            elif release - 0.5 <= t <= clip.event_time + 1.2:
                ty[i], tp[i] = hoop.yaw, hoop.pitch - 8
                if shooter is not None and t <= release + 1.8:
                    fast[i] = True

    # smoothing
    cam_y = np.zeros(len(ts))
    cam_p = np.zeros(len(ts))
    y = float(ty[0])
    p = float(np.clip(tp[0] - 2, TILT_MIN, TILT_MAX))
    v = 0.0
    for i in range(len(ts)):
        err = float(wrap180(ty[i] - y))
        vmax = SHOT_PAN if fast[i] else MAX_PAN
        dz = 2.0 if fast[i] else DEAD_ZONE
        if abs(err) <= dz:
            want = 0.0
        else:
            want = math.copysign((abs(err) - dz) / 0.35, err)
        want = max(-vmax, min(vmax, want))
        acc = 400.0 * step
        v += max(-acc, min(acc, want - v))
        y = float(wrap180(y + v * step))
        pt = float(np.clip(tp[i] - 2, TILT_MIN, TILT_MAX))
        p += (pt - p) * min(1.0, step / 0.6)
        cam_y[i], cam_p[i] = y, p

    cam_z = np.empty(len(ts))
    z = 1.0
    rel_for_fit = release if release is not None else clip.event_time
    for i, t in enumerate(ts):
        fit = None
        if shooter is not None and hoop is not None and t <= rel_for_fit + 0.25:
            sep = abs(float(wrap180(shooter[0] - hoop.yaw)))
            if sep > 2.0:
                fit = 0.6 * base_hfov / sep
        target = animated_zoom(float(t), clip, flat=flat, fit=fit, release=release)
        z += (target - z) * min(1.0, step / 0.35)
        cam_z[i] = z

    keys: list[Keyframe] = []
    marks = {}
    if release is not None and clip.type in ("made", "attempt"):
        marks[round(release, 1)] = "release"
        marks[round(clip.event_time, 1)] = "basket" if clip.type == "made" else "rim"
    for i, t in enumerate(ts):
        on_grid = abs((t - clip.start) / 0.5 - round((t - clip.start) / 0.5)) < 1e-6
        note = marks.get(round(float(t), 1), "")
        if on_grid or note or i == len(ts) - 1:
            keys.append(Keyframe(clip.id, float(t), float(cam_y[i]), float(cam_p[i]), float(cam_z[i]), note))
    return keys, release


def camera_at(keys: list[Keyframe], t: float) -> tuple[float, float, float]:
    """Smooth (Catmull-Rom) interpolation of yaw, pitch, zoom at time t."""
    if not keys:
        return 0.0, -5.0, 1.0
    if len(keys) == 1 or t <= keys[0].time:
        k = keys[0]
        return k.yaw, k.pitch, k.zoom
    if t >= keys[-1].time:
        k = keys[-1]
        return k.yaw, k.pitch, k.zoom
    times = np.array([k.time for k in keys])
    yaws = np.degrees(np.unwrap(np.radians([k.yaw for k in keys])))
    pitches = np.array([k.pitch for k in keys])
    zooms = np.array([k.zoom for k in keys])
    j = int(np.searchsorted(times, t)) - 1
    j = max(0, min(j, len(keys) - 2))
    t0, t1 = times[j], times[j + 1]
    u = (t - t0) / max(t1 - t0, 1e-6)

    def cr(arr):
        p0 = arr[max(j - 1, 0)]
        p1, p2 = arr[j], arr[j + 1]
        p3 = arr[min(j + 2, len(arr) - 1)]
        m1 = (p2 - p0) / 2
        m2 = (p3 - p1) / 2
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        return float(h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2)

    return float(wrap180(cr(yaws))), cr(pitches), max(0.5, cr(zooms))
