"""Optional scoreboard / score-log input for made-basket times.

Two ways to seed basket times (then each is refined by the rim check):

1. Score log CSV — times of each score change (or a hand-labelled clip list
   with a ball_in / event_time / time column). This is the reliable path.
2. Scoreboard overlay video or a crop region of the main video — detect
   sustained visual changes in the score digits area. OCR is optional if
   Tesseract is installed; change detection alone is enough to propose times.

When neither is given, the app uses the no-scoreboard path (strategy A).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from fhh_core import parse_time


HOOP_NAME_TO_INDEX = {"A": 0, "B": 1, "W": 0, "E": 1, "L": 0, "R": 1}


@dataclass
class ScoreEvent:
    time: float
    hoop: Optional[int] = None   # 0/1 if known, else None
    note: str = ""
    home: Optional[int] = None
    away: Optional[int] = None


def _cell(row: dict, *names: str) -> str:
    lower = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    for name in names:
        v = lower.get(name.lower())
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _parse_hoop(text: str) -> Optional[int]:
    if not text:
        return None
    t = text.strip().upper()
    if t.isdigit():
        return int(t)
    return HOOP_NAME_TO_INDEX.get(t)


def load_score_log(path: Path) -> list[ScoreEvent]:
    """Load basket / score-change times from a CSV.

    Accepted time columns (first match wins): ball_in, event_time, score_time,
    time, src_time, t. Optional hoop column: hoop, basket, side.
    Also accepts the hand-labelled clip_list_v2 format.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Score log not found: {path}")
    events: list[ScoreEvent] = []
    with open(path, newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"Score log has no header: {path}")
        for n, row in enumerate(reader, start=2):
            raw = _cell(row, "ball_in", "event_time", "score_time", "time", "src_time", "t")
            if not raw:
                # Fall back to src_end - 1s style only when ball_in missing but src_end present.
                end = _cell(row, "src_end", "end")
                if end:
                    try:
                        raw_t = parse_time(end) - 1.0
                        raw = str(raw_t)
                    except ValueError:
                        continue
                else:
                    continue
            try:
                t = parse_time(raw)
            except ValueError:
                try:
                    t = float(raw)
                except ValueError as exc:
                    raise ValueError(f"{path.name} line {n}: bad time {raw!r}") from exc
            hoop = _parse_hoop(_cell(row, "hoop", "basket", "side"))
            note = _cell(row, "note", "type", "id")
            home = away = None
            hs, aws = _cell(row, "home", "home_score"), _cell(row, "away", "away_score")
            if hs.isdigit():
                home = int(hs)
            if aws.isdigit():
                away = int(aws)
            events.append(ScoreEvent(time=float(t), hoop=hoop, note=note, home=home, away=away))
    events.sort(key=lambda e: e.time)
    # Dedupe identical times.
    out: list[ScoreEvent] = []
    for e in events:
        if out and abs(out[-1].time - e.time) < 0.05:
            continue
        out.append(e)
    return out


def detect_scoreboard_changes(
    frames_gray: list[np.ndarray],
    times: list[float],
    min_gap: float = 2.0,
    change_thr: float = 0.045,
    persist: int = 2,
) -> list[ScoreEvent]:
    """Propose score-change times from a sequence of scoreboard-region greyscale frames.

    A change is a sustained mean absolute difference vs the previous stable frame.
    This does not read the digits; the rim check still has to confirm each make.
    """
    if len(frames_gray) < 3 or len(frames_gray) != len(times):
        return []
    events: list[ScoreEvent] = []
    ref = frames_gray[0].astype(np.float32)
    pending = 0
    pending_t = None
    last_t = -1e9
    for i in range(1, len(frames_gray)):
        g = frames_gray[i].astype(np.float32)
        diff = float(np.mean(np.abs(g - ref))) / 255.0
        if diff >= change_thr:
            pending += 1
            if pending_t is None:
                pending_t = float(times[i])
            if pending >= persist and pending_t - last_t >= min_gap:
                events.append(ScoreEvent(time=pending_t, note="scoreboard change"))
                last_t = pending_t
                ref = g
                pending = 0
                pending_t = None
        else:
            pending = 0
            pending_t = None
            # Slowly track the stable board so lighting drift does not fire forever.
            ref = 0.9 * ref + 0.1 * g
    return events


def sample_scoreboard_region(
    ffmpeg: str,
    video: Path,
    start: float,
    end: float,
    region: tuple[float, float, float, float],
    fps: float = 2.0,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[list[np.ndarray], list[float]]:
    """Decode a rectangular region (x,y,w,h as fractions of frame size) from a video.

    Used for an overlay scoreboard video or a crop of the main game video.
    """
    import subprocess
    import tempfile

    x, y, w, h = region
    # FFmpeg crop uses pixels; use relative expressions.
    crop = f"crop=iw*{w:.4f}:ih*{h:.4f}:iw*{x:.4f}:ih*{y:.4f},fps={fps:.3f},format=gray"
    dur = max(0.5, end - start)
    with tempfile.TemporaryDirectory(prefix="fhh_sb_") as td:
        pattern = str(Path(td) / "f%06d.png")
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
            "-i", str(video),
            "-vf", crop,
            "-y", pattern,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("Scoreboard sample failed:\n" + (r.stderr or r.stdout or "")[-500:])
        import cv2
        files = sorted(Path(td).glob("f*.png"))
        frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in files]
        frames = [f for f in frames if f is not None]
        times = [start + i / fps for i in range(len(frames))]
        if log:
            log(f"Scoreboard: sampled {len(frames)} frames from {video.name} "
                f"({format_region(region)}) at {fps:.1f} fps.")
        return frames, times


def format_region(region: tuple[float, float, float, float]) -> str:
    x, y, w, h = region
    return f"x={x:.2f},y={y:.2f},w={w:.2f},h={h:.2f}"


def parse_region(text: str) -> tuple[float, float, float, float]:
    """Parse 'x,y,w,h' fractions (0-1) or percentages (0-100)."""
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("Scoreboard region must be x,y,w,h (fractions of the frame)")
    vals = [float(p) for p in parts]
    if max(vals) > 1.0:
        vals = [v / 100.0 for v in vals]
    x, y, w, h = vals
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
        raise ValueError("Scoreboard region out of range")
    if x + w > 1.01 or y + h > 1.01:
        raise ValueError("Scoreboard region extends past the frame")
    return x, y, w, h
