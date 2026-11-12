"""Analyze and render jobs (shared by the window and the command line)."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from fhh_camera import BASE_HFOV_169, FLAT_ZOOM_169, estimate_release, filter_bystanders, plan_clip
from fhh_core import (
    VERSION, PERF_LABELS, JobController, Progress, ProgressCb, assets_dir, find_tool,
    format_ts, output_stem_for, priority_name, resolve_threads, work_root,
)
from fhh_detect import AnalysisSettings, Detections, Hoop, detect_events, run_analysis, warmup_end
from fhh_project import (
    ClipSettings, Project, data_paths, events_to_clips, load_project, read_clips, read_keyframes,
    save_project, sorted_for_render, write_clips, write_keyframes, bystander_cells,
)
from fhh_render import RenderSettings, render_all
from fhh_source import FLAT_HFOV, SourceInfo, detect_source

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class Job:
    input_path: Path
    output_path: Path                       # <stem>_highlights.mp4 (data files sit beside it)
    start: Optional[float] = None
    end: Optional[float] = None
    quality: str = "balanced"
    sensitivity: float = 0.5
    lens_model: str = "x4"
    device: str = "auto"
    source_kind: str = "auto"
    hoops: list[Hoop] = field(default_factory=list)
    clip: ClipSettings = field(default_factory=ClipSettings)
    render: RenderSettings = field(default_factory=RenderSettings)
    perf_mode: str = "balanced"
    perf_threads: Optional[int] = None
    overwrite: bool = False


def default_title(info: Optional[SourceInfo], path: Path) -> str:
    m = re.search(r"(20\d\d)(\d\d)(\d\d)", path.name)
    dt = None
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            dt = None
    if dt is None and info is not None and info.creation_time:
        try:
            dt = datetime.fromisoformat(info.creation_time.replace("Z", "+00:00"))
        except ValueError:
            dt = None
    if dt is None:
        dt = datetime.now()
    return f"{MONTHS[dt.month - 1]} {dt.day}, {dt.year} Highlights"


def music_tracks() -> list[Path]:
    d = assets_dir() / "music"
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in (".m4a", ".mp3", ".wav", ".aac", ".ogg", ".flac"))


def _tools():
    return find_tool("ffmpeg"), find_tool("ffprobe")


def _check_paths(job: Job) -> None:
    ins = [job.input_path.resolve()]
    if job.output_path.resolve() in ins:
        raise ValueError("The output file cannot be the input file. Your originals are never overwritten.")


def work_dir_for(output_path: Path) -> Path:
    d = work_root() / output_path.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _use_performance(job: Job, ctl: JobController, log: Callable[[str], None]) -> None:
    """Set priority and thread count. The window can change both while a run continues."""
    threads = resolve_threads(job.perf_mode, job.perf_threads)
    ctl.set_performance(job.perf_mode, threads)
    label = PERF_LABELS.get(job.perf_mode, job.perf_mode)
    log(f"Performance: {label} ({threads} threads, {priority_name(job.perf_mode)} priority). "
        "You can change this while it runs.")


def analyze(job: Job, ctl: JobController, progress: ProgressCb, log: Callable[[str], None]) -> dict:
    ffmpeg, ffprobe = _tools()
    _check_paths(job)
    threads = resolve_threads(job.perf_mode, job.perf_threads)
    _use_performance(job, ctl, log)
    info = detect_source(job.input_path, ffmpeg, ffprobe, job.source_kind)
    log(f"Input: {info.describe()}")
    for n in info.notes:
        log(n)
    start = max(0.0, job.start or 0.0)
    end = min(info.duration, job.end) if job.end else info.duration
    if end <= start + 2:
        raise ValueError(f"The time range {format_ts(start)}-{format_ts(end)} is empty or too short.")
    log(f"Range: {format_ts(start)} to {format_ts(end)} ({(end - start) / 60:.1f} min)")
    prog = Progress(progress, ctl)
    stages = [("Detecting plays", 20.0), ("Planning camera", 0.5)]
    if not job.hoops:
        stages.insert(0, ("Finding hoops", 3.0))
    prog.plan(stages)
    st = AnalysisSettings(quality=job.quality, start=start, end=end, sensitivity=job.sensitivity,
                          lens_model=job.lens_model, device=job.device, threads=threads)
    d, hoops = run_analysis(ffmpeg, info, list(job.hoops), st, ctl, prog, log, auto_hoops=not job.hoops)
    work = work_dir_for(job.output_path)
    det_path = work / "detections.npz"
    d.save(det_path)

    prog.begin("Planning camera")
    by = bystander_cells(d.persons, len(d.times))
    if by:
        log(f"Ignoring {len(by)} spots where people stand still (bench, spectators).")
    warm = warmup_end(d, start) if job.clip.skip_warmup else None
    if warm is not None:
        log(f"Warm-up detected until about {format_ts(warm)} (several balls in play). "
            "Clips before that are switched off; switch them on in the list if needed.")
    events = detect_events(d, hoops, job.sensitivity)
    P = filter_bystanders(d.persons, by)
    releases = []
    for e in events:
        est = estimate_release(e.time, hoops[e.hoop], d.balls, P) if e.hoop < len(hoops) else None
        releases.append(est[0] if est else None)
    clips = events_to_clips(events, hoops, job.clip, start, end, warm, releases)
    hf = FLAT_HFOV / FLAT_ZOOM_169 if info.kind == "flat" else BASE_HFOV_169
    keys = []
    for c in clips:
        ks, rel = plan_clip(c, d, hoops, by, hf, flat=info.kind == "flat")
        c.release = rel
        keys.extend(ks)
    paths = data_paths(job.output_path)
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_clips(clips, paths["clips"])
    write_keyframes(keys, paths["camera"])
    proj = Project(input=str(job.input_path), kind=info.kind, start=start, end=end, hoops=hoops,
                   bystanders=by, warmup_end=warm, detections=str(det_path),
                   settings=dict(quality=job.quality, sensitivity=job.sensitivity, lens_model=job.lens_model,
                                 pre=job.clip.pre, post=job.clip.post, version=VERSION,
                                 stabilized=info.stab is not None,
                                 lrv=str(info.lrv) if info.lrv else ""))
    save_project(proj, paths["project"])
    made = sum(1 for c in clips if c.type == "made")
    att = sum(1 for c in clips if c.type == "attempt")
    per = {h.name: sum(1 for c in clips if c.hoop == h.name and c.type == "made") for h in hoops}
    log(f"Found {made} made-basket clips and {att} other shots at the rim "
        f"({', '.join(f'hoop {k}: {v}' for k, v in per.items())}). "
        f"{sum(1 for c in clips if c.include)} switched on, "
        f"{sum(c.duration for c in clips if c.include):.0f} s.")
    log(f"Clip list: {paths['clips']}")
    log(f"Camera keyframes: {paths['camera']}")
    prog.update(1.0, "Analysis done", force=True)
    return dict(paths=paths, clips=clips, project=proj, info=info)


def render(job: Job, clips_csv: Path, ctl: JobController, progress: ProgressCb,
           log: Callable[[str], None]) -> list[Path]:
    ffmpeg, ffprobe = _tools()
    out = job.output_path
    paths = data_paths(Path(str(clips_csv)[:-4] + ".mp4")) if str(clips_csv).lower().endswith(".csv") else data_paths(out)
    proj = load_project(paths["project"]) if paths["project"].is_file() else None
    src = Path(proj.input) if proj and proj.input and Path(proj.input).is_file() else job.input_path
    if src is None or not Path(src).is_file():
        raise FileNotFoundError("The source video for this clip list was not found. Choose the input video again.")
    if Path(src).resolve() == out.resolve():
        raise ValueError("The output file cannot be the input file.")
    if not job.overwrite:
        targets = [out] if "16:9" in job.render.formats else []
        if "9:16" in job.render.formats:
            targets.append(out.with_name(out.stem + "_short" + out.suffix))
        taken = [t for t in targets if t.exists()]
        if taken:
            raise FileExistsError("Output already exists: " + ", ".join(t.name for t in taken)
                                  + ". Choose another name or allow overwriting.")
    threads = resolve_threads(job.perf_mode, job.perf_threads)
    _use_performance(job, ctl, log)
    job.render.threads = threads
    job.render.lens_model = job.lens_model
    info = detect_source(Path(src), ffmpeg, ffprobe, proj.kind if proj and job.source_kind == "auto" else job.source_kind)
    if not (proj and proj.settings.get("stabilized")):
        info.stab = None
    elif info.stab is None:
        log("This highlight list was leveled from an .lrv, but that file is not next to the .insv. "
            "The picture will not be stabilized.")
    log(f"Input: {info.describe()}")
    clips = read_clips(Path(clips_csv))
    keys = read_keyframes(paths["camera"])
    sel = sorted_for_render(clips)
    for c in sel:
        if c.start < 0 or c.end > info.duration + 0.05:
            raise ValueError(f"Clip {c.id} ({format_ts(c.start)}-{format_ts(c.end)}) is outside the video.")
    if not sel:
        raise ValueError("No clips are switched on (include = yes). Switch some on and try again.")
    det = None
    if proj and proj.detections and Path(proj.detections).is_file():
        try:
            det = Detections.load(Path(proj.detections))
        except Exception as exc:
            log(f"Could not read the detection cache ({exc}); clips without keyframes will hold on the hoop.")
    hoops = proj.hoops if proj else list(job.hoops)
    by = proj.bystanders if proj else []
    prog = Progress(progress, ctl)
    stages = []
    for fmt in job.render.formats:
        stages += [(f"Rendering clips ({fmt})", 8.0), (f"Assembling ({fmt})", 2.0)]
    prog.plan(stages)
    log(f"{len(sel)} clips, {sum(c.duration for c in sel):.0f} s. Music: "
        f"{Path(job.render.music).name if job.render.music else 'none'}")
    outputs = render_all(ffmpeg, ffprobe, info, sel, keys, hoops, by, det, out, job.render,
                         work_dir_for(out), ctl, prog, log)
    for o in outputs:
        log(f"Done: {o}")
    prog.update(1.0, "Done", force=True)
    return outputs
