"""Highlight clips, camera keyframes, and the editable CSV/JSON data files.

Files written next to the output video (all plain text, edit in Excel or
any text editor, then re-render):
  <name>_highlights.csv          one row per clip
  <name>_highlights_camera.csv   virtual-camera keyframes per clip
  <name>_highlights_project.json source, time range, hoops, settings
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from fhh_core import VERSION, format_ts, parse_time
from fhh_detect import Event, Hoop

CLIP_COLUMNS = ["id", "include", "order", "type", "hoop", "start", "end", "event_time",
                "score", "note", "start_s", "end_s", "event_s"]
KEY_COLUMNS = ["clip_id", "time", "yaw", "pitch", "zoom", "note", "time_s"]


@dataclass
class Clip:
    id: str
    start: float
    end: float
    event_time: float
    type: str = "made"          # made | attempt | play | manual
    hoop: str = "A"
    score: float = 0.0
    include: bool = True
    order: int = 0
    note: str = ""
    release: Optional[float] = None   # estimated release time (not in CSV)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Keyframe:
    clip_id: str
    time: float
    yaw: float
    pitch: float
    zoom: float = 1.0
    note: str = ""


@dataclass
class Project:
    input: str
    kind: str
    start: float
    end: float
    hoops: list[Hoop] = field(default_factory=list)
    bystanders: list[list[float]] = field(default_factory=list)  # [yaw, pitch] cell centres
    warmup_end: Optional[float] = None
    settings: dict = field(default_factory=dict)
    detections: str = ""
    version: str = VERSION

    def to_json(self) -> dict:
        d = asdict(self)
        d["hoops"] = [h.to_dict() for h in self.hoops]
        return d

    @staticmethod
    def from_json(d: dict) -> "Project":
        return Project(
            input=d.get("input", ""), kind=d.get("kind", ""), start=float(d.get("start", 0)),
            end=float(d.get("end", 0)), hoops=[Hoop.from_dict(h) for h in d.get("hoops", [])],
            bystanders=d.get("bystanders", []), warmup_end=d.get("warmup_end"),
            settings=d.get("settings", {}), detections=d.get("detections", ""),
            version=d.get("version", VERSION),
        )


def data_paths(output_video: Path) -> dict:
    stem = output_video.with_suffix("")
    return {
        "clips": Path(f"{stem}.csv"),
        "camera": Path(f"{stem}_camera.csv"),
        "project": Path(f"{stem}_project.json"),
    }


def save_project(p: Project, path: Path) -> None:
    path.write_text(json.dumps(p.to_json(), indent=2) + "\n", encoding="utf-8")


def load_project(path: Path) -> Project:
    return Project.from_json(json.loads(path.read_text(encoding="utf-8")))


def _t(row: dict, key: str) -> Optional[float]:
    v = (row.get(key) or "").strip()
    if v:
        return parse_time(v)
    v = (row.get(key + "_s") or "").strip()
    if v:
        return float(v)
    return None


def write_clips(clips: list[Clip], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(CLIP_COLUMNS)
        for c in clips:
            w.writerow([c.id, "yes" if c.include else "no", c.order, c.type, c.hoop,
                        format_ts(c.start), format_ts(c.end), format_ts(c.event_time),
                        f"{c.score:.2f}", c.note, f"{c.start:.2f}", f"{c.end:.2f}", f"{c.event_time:.2f}"])


def read_clips(path: Path) -> list[Clip]:
    clips: list[Clip] = []
    errors: list[str] = []
    with open(path, newline="", encoding="utf-8-sig") as fp:
        for n, row in enumerate(csv.DictReader(fp), start=2):
            cid = (row.get("id") or "").strip()
            if not cid:
                continue
            try:
                start = _t(row, "start")
                end = _t(row, "end")
                if start is None or end is None:
                    raise ValueError("start and end are required")
                if end <= start:
                    raise ValueError("end must be after start")
                ev = _t(row, "event_time")
                inc = (row.get("include") or "yes").strip().lower() not in ("no", "n", "0", "false", "off")
                if cid.startswith("#"):
                    inc = False
                order = int(float(row.get("order") or 0))
                clips.append(Clip(
                    id=cid.lstrip("#").strip() or cid, start=start, end=end,
                    event_time=ev if ev is not None else (start + end) / 2,
                    type=(row.get("type") or "manual").strip() or "manual",
                    hoop=(row.get("hoop") or "A").strip() or "A",
                    score=float(row.get("score") or 0), include=inc, order=order,
                    note=(row.get("note") or "").strip(),
                ))
            except Exception as exc:
                errors.append(f"line {n} ({cid}): {exc}")
    if errors:
        raise ValueError("Problems in the clips file:\n  " + "\n  ".join(errors))
    ids = [c.id for c in clips]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError("Duplicate clip ids in the clips file: " + ", ".join(dup))
    return clips


def write_keyframes(keys: list[Keyframe], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(KEY_COLUMNS)
        for k in keys:
            w.writerow([k.clip_id, format_ts(k.time), f"{k.yaw:.2f}", f"{k.pitch:.2f}",
                        f"{k.zoom:.2f}", k.note, f"{k.time:.2f}"])


def read_keyframes(path: Path) -> list[Keyframe]:
    keys: list[Keyframe] = []
    if not path.is_file():
        return keys
    errors: list[str] = []
    with open(path, newline="", encoding="utf-8-sig") as fp:
        for n, row in enumerate(csv.DictReader(fp), start=2):
            cid = (row.get("clip_id") or "").strip()
            if not cid or cid.startswith("#"):
                continue
            try:
                t = _t(row, "time")
                if t is None:
                    raise ValueError("time is required")
                zoom = float(row.get("zoom") or 1.0)
                if not 0.5 <= zoom <= 3.0:
                    raise ValueError("zoom must be between 0.5 and 3.0")
                keys.append(Keyframe(cid, t, float(row["yaw"]), float(row["pitch"]), zoom,
                                     (row.get("note") or "").strip()))
            except Exception as exc:
                errors.append(f"line {n} ({cid}): {exc}")
    if errors:
        raise ValueError("Problems in the camera file:\n  " + "\n  ".join(errors))
    keys.sort(key=lambda k: (k.clip_id, k.time))
    return keys


# ---------------------------------------------------------------------------
# Events -> clips
# ---------------------------------------------------------------------------

@dataclass
class ClipSettings:
    pre: float = 6.0             # seconds before the basket
    post: float = 3.0            # seconds after
    include_attempts: bool = False
    skip_warmup: bool = True
    max_total: float = 0.0       # seconds of clips (0 = no limit)


def events_to_clips(events: list[Event], hoops: list[Hoop], cs: ClipSettings,
                    range_start: float, range_end: float, warmup: Optional[float],
                    releases: Optional[list] = None) -> list[Clip]:
    raw: list[Clip] = []
    for n, e in enumerate(events):
        rel = releases[n] if releases and n < len(releases) else None
        start = e.time - cs.pre
        if rel is not None:
            start = min(start, rel - 2.0)  # long shots: start well before the release
        start = max(range_start, start)
        end = min(range_end, e.time + cs.post)
        raw.append(Clip(id="", start=start, end=end, event_time=e.time, type=e.kind,
                        hoop=hoops[e.hoop].name if e.hoop < len(hoops) else "A",
                        score=e.score, note=e.note, release=rel))
    raw.sort(key=lambda c: c.start)
    # dedupe/merge overlaps
    merged: list[Clip] = []
    for c in raw:
        if merged and c.start < merged[-1].end:
            m = merged[-1]
            overlap = m.end - c.start
            if m.hoop == c.hoop:
                # same hoop: one longer clip (e.g. a putback after a miss)
                both_made = m.type == "made" and c.type == "made"
                c_better = (c.type == "made" and m.type != "made") or (c.type == m.type and c.score > m.score)
                m.end = max(m.end, c.end)
                if c_better:
                    m.type, m.event_time, m.note, m.release = c.type, c.event_time, c.note, c.release
                m.score = max(m.score, c.score)
                if both_made:
                    m.note = (m.note + " (+ another basket)").strip()
                continue
            if overlap > 2.0:
                # different hoops overlapping a lot: one of them is wrong, keep the stronger
                if (c.type == "made", c.score) > (m.type == "made", m.score):
                    merged[-1] = c
                continue
            mid = (m.end + c.start) / 2
            m.end, c.start = mid, mid
        merged.append(c)
    for i, c in enumerate(merged, 1):
        c.id = f"c{i:03d}"
        c.order = i
        c.include = c.type == "made" or (c.type == "attempt" and cs.include_attempts)
        if cs.skip_warmup and warmup is not None and c.event_time < warmup:
            c.include = False
            c.note = (c.note + "; warm-up?").strip("; ")
    if cs.max_total and cs.max_total > 0:
        select_balanced(merged, cs.max_total)
    return merged


def select_balanced(clips: list[Clip], max_total: float) -> None:
    """Keep the best clips up to max_total seconds, alternating hoops."""
    pool = [c for c in clips if c.include]
    by_hoop: dict[str, list[Clip]] = {}
    for c in sorted(pool, key=lambda c: (c.type != "made", -c.score)):
        by_hoop.setdefault(c.hoop, []).append(c)
    chosen: set[str] = set()
    total = 0.0
    order = sorted(by_hoop)
    while any(by_hoop.values()):
        progressed = False
        for h in order:
            if not by_hoop[h]:
                continue
            c = by_hoop[h].pop(0)
            if total + c.duration <= max_total or not chosen:
                chosen.add(c.id)
                total += c.duration
                progressed = True
        if not progressed:
            break
    for c in clips:
        if c.include and c.id not in chosen:
            c.include = False
            c.note = (c.note + "; over max length").strip("; ")


def bystander_cells(persons: np.ndarray, n_frames: int, cell: float = 3.0, thr: float = 0.45) -> list[list[float]]:
    """Grid cells where someone stands in most frames (bench, spectators, filming)."""
    if n_frames < 20 or len(persons) == 0:
        return []
    ky = np.floor(persons[:, 1] / cell).astype(int)
    kp = np.floor(persons[:, 2] / cell).astype(int)
    keys = {}
    for t, a, b in zip(persons[:, 0], ky, kp):
        keys.setdefault((a, b), set()).add(round(float(t), 2))
    cells = []
    for (a, b), ts in keys.items():
        if len(ts) / n_frames >= thr:
            cells.append([(a + 0.5) * cell, (b + 0.5) * cell])
    return cells


def sorted_for_render(clips: list[Clip]) -> list[Clip]:
    return sorted([c for c in clips if c.include], key=lambda c: (c.order, c.start))
