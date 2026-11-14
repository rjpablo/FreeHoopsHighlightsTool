"""Final render: reframed clips -> intro, title, logo, promo cards, music.

1. Each included clip is decoded at full resolution, the virtual camera is
   applied per frame (keyframes from the camera CSV), and the frames are
   piped to FFmpeg with the clip's own audio (short fades at each cut).
2. One FFmpeg pass joins intro + clips (+ end card), overlays the two-line
   title, corner logo, lower-third promo cards and an optional full-screen
   end card, and mixes music under the game sound.
3. The no-music copy (for Marketing/TikTok) reuses the same video stream.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from fhh_camera import (
    BASE_HFOV_169, BASE_HFOV_916, FLAT_ZOOM_169, animated_zoom, camera_at, plan_clip, zooms_are_flat,
)
from fhh_core import (
    JobController, Progress, RenderCancelled, assets_dir, even, ffmpeg_threads_args, fonts_dir, format_ts,
    popen_kwargs, probe_duration, run_ffmpeg_progress, summarize_ffmpeg, x264_args,
)
from fhh_detect import Detections, Hoop
from fhh_project import Clip, Keyframe
from fhh_source import FLAT_HFOV, ClipRenderer, FrameReader, SourceInfo, base_filter, decode_plan, grab_frame_raw
from graphics import (
    DEFAULT_QR_URL, END_SUB, END_TITLE, HL_SUB, LT_SUB, LT_TITLE, QR_CAPTION,
    render_highlight_graphics,
)

INTRO_FALLBACK = 6.8
CARD_FADE = 0.4
PREVIEW_CHUNK_S = 60.0


@dataclass
class CardWindow:
    start: float
    duration: float


@dataclass
class RenderSettings:
    formats: list[str] = field(default_factory=lambda: ["16:9"])
    resolution: str = "auto"          # auto | 720 | 1080 | 1440 | 2160 (short side)
    crf: int = 18
    preset: str = "medium"
    intro_on: bool = True
    title_on: bool = True
    title_main: str = ""
    title_sub: str = HL_SUB
    title_duration: float = 4.0
    logo_on: bool = True
    music: str = ""                   # path to an audio file, "" = no music
    music_volume: float = 0.55        # lower default; final mix is loudnorm'd to ~-14 LUFS
    game_volume: float = 0.35         # game sound under the music
    music_lufs: float = -14.0         # target integrated loudness for the music mix
    no_music_copy: bool = True
    lt_on: bool = True
    lt_windows: list[CardWindow] = field(default_factory=lambda: [CardWindow(12.0, 6.0)])
    lt_line1: str = LT_TITLE
    lt_line2: str = LT_SUB
    end_on: bool = True
    end_at_end: bool = True
    end_start: float = 0.0
    end_duration: float = 7.0
    end_title: str = END_TITLE
    end_sub: str = END_SUB
    qr_caption: str = QR_CAPTION
    qr_url: str = DEFAULT_QR_URL
    clip_fade: float = 0.25
    lens_model: str = "x4"
    threads: int = 4
    preview_chunks: bool = False     # write ~1 min playable parts, then build the final from them


def output_size(fmt: str, info: SourceInfo, resolution: str) -> tuple[int, int]:
    if resolution == "auto":
        if info.kind == "flat":
            short = min(info.width, info.height)
            short = int(min(2160, max(720, short)))
        else:
            short = 1440 if info.lens_size >= 3800 or (info.kind == "equirect" and info.width >= 7680) else 1080
    else:
        short = int(resolution)
    long_ = int(round(short * 16 / 9))
    if fmt == "9:16":
        return even(short), even(long_)
    return even(long_), even(short)


def base_hfov(fmt: str, info: SourceInfo) -> float:
    if info.kind == "flat":
        if fmt == "9:16":
            # full source height, 9:16 slice
            return FLAT_HFOV * (info.height * 9 / 16) / info.width
        return FLAT_HFOV / FLAT_ZOOM_169
    return BASE_HFOV_916 if fmt == "9:16" else BASE_HFOV_169


def save_clip_shots(ffmpeg: str, info: SourceInfo, clips: list[Clip], keys: list[Keyframe],
                    hoops: list[Hoop], lens_model: str, dest: Path,
                    ctl: JobController, log: Callable[[str], None]) -> int:
    """One still per clip, framed the way the highlight will show the basket."""
    import cv2
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.jpg"):
        old.unlink()
    if not clips:
        return 0
    by_id: dict[str, list[Keyframe]] = {}
    for key in keys:
        by_id.setdefault(key.clip_id, []).append(key)
    plan = decode_plan(info, max_lens=960, max_width=1280)
    graph = base_filter(info)
    native = (2 * info.lens_size, info.lens_size) if info.is_fisheye else (info.width, info.height)
    if (plan.width, plan.height) != native:
        graph += f",scale={plan.width}:{plan.height}:flags=area"
    files = info.files if info.kind == "split" else info.files[:1]
    renderer = ClipRenderer(info, plan, lens_model)
    saved = 0
    for i, clip in enumerate(clips, 1):
        ctl.checkpoint()
        try:
            t = float(clip.event_time)
            ck = sorted(by_id.get(clip.id) or _fallback_keys(clip, hoops), key=lambda k: k.time)
            frame = grab_frame_raw(ffmpeg, files, t, graph, plan.width, plan.height)
            yaw, pitch, zoom = camera_at(ck, t)
            if zooms_are_flat(ck):
                zoom = animated_zoom(t, clip, info.kind == "flat")
            view = renderer.view(frame, yaw, pitch, base_hfov("16:9", info) / max(zoom, 0.5), 640, 360, t)
            label = f"{clip.id}  {format_ts(t)}  hoop {clip.hoop}"
            cv2.putText(view, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 140, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(dest / f"{clip.id}.jpg"), view, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            saved += 1
            log(f"Screenshot {i}/{len(clips)}: {clip.id} at {format_ts(t)}")
        except RenderCancelled:
            raise
        except Exception as exc:
            log(f"Screenshot {i}/{len(clips)} ({clip.id}) failed: {exc}")
    return saved


def _fallback_keys(clip: Clip, hoops: list[Hoop]) -> list[Keyframe]:
    h = next((x for x in hoops if x.name == clip.hoop), hoops[0] if hoops else None)
    yaw, pitch = (h.yaw, h.pitch - 8) if h else (0.0, -5.0)
    return [Keyframe(clip.id, clip.start, yaw, pitch, 1.0, "fixed (no detections)"),
            Keyframe(clip.id, clip.end, yaw, pitch, 1.0, "")]


def chunk_closes_before(filled: float, nxt: float, limit: float = PREVIEW_CHUNK_S) -> bool:
    """True when this clip should start a new preview part.

    A part may run a little past a minute so a short play is not left on its own.
    A play is never split. One clip longer than a minute is its own part.
    """
    if filled <= 0:
        return False
    if filled >= limit:
        return True
    return filled + nxt > limit * 1.5


def concat_copy(ffmpeg: str, parts: list[Path], dest: Path, ctl: JobController) -> None:
    """Join finished clip files into one playable preview without re-encoding."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    listfile = dest.with_suffix(".txt")
    listfile.write_text("".join("file '" + p.resolve().as_posix().replace("'", "'\\''") + "'\n" for p in parts),
                        encoding="utf-8")
    cmd = [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listfile),
           "-c", "copy", "-movflags", "+faststart", str(dest)]
    kw = {k: v for k, v in popen_kwargs(ctl.priority_mode).items() if k == "creationflags"}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    finally:
        listfile.unlink(missing_ok=True)
    ctl.raise_if_cancelled()
    if r.returncode != 0 or not dest.is_file():
        err = r.stderr[-600:] if r.stderr else ""
        raise RuntimeError(f"Could not write preview {dest.name}:\n{err}")


def render_clip_segment(ffmpeg: str, info: SourceInfo, clip: Clip, keys: list[Keyframe],
                        fmt: str, out_w: int, out_h: int, rs: RenderSettings, seg_path: Path,
                        ctl: JobController, on_frame: Callable[[int], None], log: Callable[[str], None]) -> None:
    plan = decode_plan(info)
    renderer = ClipRenderer(info, plan, rs.lens_model)
    hfov0 = base_hfov(fmt, info)
    dur = clip.duration
    threads = ctl.thread_count(rs.threads)
    reader = FrameReader(ffmpeg, info, clip.start, dur, plan, ctl, fps=None, threads=max(1, threads // 2))
    fade = min(rs.clip_fade, dur / 4)
    vf = (f"fade=t=in:st=0:d={fade:.3f},fade=t=out:st={max(0, dur - fade):.3f}:d={fade:.3f},"
          "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p")
    af = (f"aformat=sample_rates=48000:channel_layouts=stereo,afade=t=in:st=0:d={fade:.3f},"
          f"afade=t=out:st={max(0, dur - fade):.3f}:d={fade:.3f}")
    audio_in: list[str]
    if info.has_audio:
        audio_in = ["-ss", f"{clip.start:.3f}", "-t", f"{dur:.3f}", "-i", str(info.files[info.audio_file])]
    else:
        audio_in = ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
    seg_crf = max(10, rs.crf - 4)
    cmd = [ffmpeg, "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", info.fps_str, "-i", "-",
           *audio_in,
           "-map", "0:v", "-map", "1:a:0", "-vf", vf, "-af", af,
           *ffmpeg_threads_args(threads), *x264_args(rs.preset, seg_crf, threads),
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest", str(seg_path)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **popen_kwargs(ctl.priority_mode))
    ctl.attach(enc)
    n = 0
    try:
        assert enc.stdin is not None
        for t, frame in reader:
            yaw, pitch, zoom = camera_at(keys, t)
            if zooms_are_flat(keys):
                zoom = animated_zoom(t, clip, info.kind == "flat")
            hfov = max(10.0, min(150.0, hfov0 / max(0.5, zoom)))
            view = renderer.view(frame, yaw, pitch, hfov, out_w, out_h, t)
            try:
                enc.stdin.write(np.ascontiguousarray(view).tobytes())
            except (BrokenPipeError, OSError):
                break
            n += 1
            on_frame(n)
        enc.stdin.close()
    finally:
        if ctl.cancelled() and enc.poll() is None:
            enc.kill()
        rc = enc.wait()
    ctl.raise_if_cancelled()
    if rc != 0 or not seg_path.is_file():
        err = enc.stderr.read().decode(errors="replace") if enc.stderr else ""
        raise RuntimeError(f"Encoding clip {clip.id} failed:\n{err[-800:]}")


def assemble(ffmpeg: str, ffprobe: str, segs: list[Path], out_path: Path, nomusic_path: Optional[Path],
             out_w: int, out_h: int, fps_str: str, rs: RenderSettings, tmp: Path,
             ctl: JobController, prog: Progress, log: Callable[[str], None]) -> None:
    assets = assets_dir()
    gfx = render_highlight_graphics(
        tmp / "gfx", out_w, out_h, assets, fonts_dir(assets),
        title_main=rs.title_main, title_sub=rs.title_sub,
        lt_line1=rs.lt_line1 or LT_TITLE, lt_line2=rs.lt_line2 or LT_SUB, lt_logo=not rs.logo_on,
        end_title=rs.end_title or END_TITLE, end_sub=rs.end_sub or END_SUB,
        qr_caption=rs.qr_caption or QR_CAPTION, qr_url=rs.qr_url or DEFAULT_QR_URL,
    )
    seg_durs = [probe_duration(s, ffprobe, 0.0) for s in segs]
    main_dur = sum(seg_durs)
    if main_dur <= 0:
        raise RuntimeError("The rendered clips are empty.")
    listfile = tmp / "clips.txt"
    listfile.write_text("".join("file '" + s.as_posix().replace("'", "'\\''") + "'\n" for s in segs),
                        encoding="utf-8")

    intro_mov = assets / "intro.mov"
    intro_wav = assets / "intro_audio.wav"
    intro_on = rs.intro_on and intro_mov.is_file()
    I = probe_duration(intro_mov, ffprobe, INTRO_FALLBACK) if intro_on else 0.0
    end_at_end = rs.end_on and rs.end_at_end and rs.end_duration > 0
    E = rs.end_duration if end_at_end else 0.0
    total = I + main_dur + E
    music = Path(rs.music) if rs.music else None
    if music is not None and not music.is_file():
        log(f"Music file not found, continuing without music: {music}")
        music = None
    want_nm = music is not None and nomusic_path is not None

    inputs: list[list[str]] = []

    def add(*a: str) -> int:
        inputs.append(list(a))
        return len(inputs) - 1

    i_main = add("-f", "concat", "-safe", "0", "-i", str(listfile))
    F = fps_str
    f: list[str] = []
    vparts: list[str] = []
    if intro_on:
        i_plate = add("-loop", "1", "-t", f"{I:.3f}", "-i", str(gfx["intro_plate"]))
        i_anim = add("-i", str(intro_mov))
        f.append(f"[{i_plate}:v]fps={F},format=rgba,setsar=1[plate]")
        f.append(f"[{i_anim}:v]fps={F},format=rgba,scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=lanczos,setsar=1[anim]")
        f.append(f"[plate][anim]overlay=(W-w)/2:(H-h)/2:format=auto:eof_action=pass,trim=0:{I:.3f},setpts=PTS-STARTPTS,fps={F},format=yuv420p,setsar=1[vintro]")
        vparts.append("[vintro]")
    f.append(f"[{i_main}:v]fps={F},format=yuv420p,setsar=1,setpts=PTS-STARTPTS[vmain]")
    vparts.append("[vmain]")
    if end_at_end:
        i_end = add("-loop", "1", "-t", f"{E:.3f}", "-i", str(gfx["endcard"]))
        f.append(f"[{i_end}:v]fps={F},format=yuv420p,setsar=1[vend]")
        vparts.append("[vend]")
    if len(vparts) > 1:
        f.append(f"{''.join(vparts)}concat=n={len(vparts)}:v=1:a=0[vcat]")
    else:
        f.append("[vmain]null[vcat]")
    cur = "vcat"

    def overlay(src_label: str, enable_a: float, enable_b: float, name: str) -> None:
        nonlocal cur
        f.append(f"[{cur}][{src_label}]overlay=0:0:enable='between(t,{enable_a:.3f},{enable_b:.3f})'"
                 f":format=auto:eof_action=pass[{name}]")
        cur = name

    if rs.logo_on:
        i_logo = add("-loop", "1", "-t", f"{total + 1:.3f}", "-i", str(gfx["logo_bug"]))
        f.append(f"[{i_logo}:v]fps={F},format=rgba[logo]")
        overlay("logo", I, I + main_dur, "vlogo")
    if rs.title_on and (rs.title_main.strip() or rs.title_sub.strip()):
        hold = max(0.5, min(rs.title_duration, main_dur - 1.0))
        i_title = add("-loop", "1", "-t", f"{I + hold + 1.5:.3f}", "-i", str(gfx["title"]))
        f.append(f"[{i_title}:v]fps={F},format=rgba,fade=t=in:st={I:.3f}:d=0.4:alpha=1,"
                 f"fade=t=out:st={I + hold:.3f}:d=1.0:alpha=1[title]")
        overlay("title", I, I + hold + 1.0, "vtitle")
    if rs.lt_on and rs.lt_windows:
        wins = []
        for w in rs.lt_windows:
            s, d = float(w.start), float(w.duration)
            if d <= 0 or s >= total:
                log(f"Skipping lower third at {format_ts(s)}: outside the video.")
                continue
            wins.append((s, min(d, total - s)))
        if wins:
            i_lt = add("-loop", "1", "-t", f"{total + 1:.3f}", "-i", str(gfx["lower_third"]))
            outs = "".join(f"[lt{k}]" for k in range(len(wins)))
            f.append(f"[{i_lt}:v]fps={F},format=rgba,split={len(wins)}{outs}" if len(wins) > 1
                     else f"[{i_lt}:v]fps={F},format=rgba[lt0]")
            for k, (s, d) in enumerate(wins):
                fd = min(CARD_FADE, d / 3)
                f.append(f"[lt{k}]fade=t=in:st={s:.3f}:d={fd:.3f}:alpha=1,"
                         f"fade=t=out:st={s + d - fd:.3f}:d={fd:.3f}:alpha=1[ltf{k}]")
                overlay(f"ltf{k}", s, s + d, f"vlt{k}")
    if rs.end_on and not rs.end_at_end and rs.end_duration > 0:
        s, d = float(rs.end_start), float(rs.end_duration)
        if s < total:
            d = min(d, total - s)
            fd = min(CARD_FADE, d / 3)
            i_endo = add("-loop", "1", "-t", f"{s + d + 1:.3f}", "-i", str(gfx["endcard"]))
            f.append(f"[{i_endo}:v]fps={F},format=rgba,fade=t=in:st={s:.3f}:d={fd:.3f}:alpha=1,"
                     f"fade=t=out:st={s + d - fd:.3f}:d={fd:.3f}:alpha=1[endo]")
            overlay("endo", s, s + d, "vendo")
    f.append(f"[{cur}]format=yuv420p,setsar=1[vout]")

    # ---- audio
    copies = 2 if want_nm else 1
    aparts: list[list[str]] = [[] for _ in range(copies)]

    def split(src: str, base: str) -> list[str]:
        if copies == 1:
            f.append(f"{src}anull[{base}0]")
            return [f"[{base}0]"]
        f.append(f"{src}asplit=2[{base}0][{base}1]")
        return [f"[{base}0]", f"[{base}1]"]

    fmt = "aformat=sample_rates=48000:channel_layouts=stereo"
    if intro_on:
        if intro_wav.is_file():
            i_sfx = add("-i", str(intro_wav))
            f.append(f"[{i_sfx}:a]{fmt},atrim=0:{I:.3f},asetpts=PTS-STARTPTS,apad=whole_dur={I:.3f}[aintro]")
        else:
            i_sfx = add("-f", "lavfi", "-t", f"{I:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000")
            f.append(f"[{i_sfx}:a]{fmt}[aintro]")
        for k, lab in enumerate(split("[aintro]", "ai")):
            aparts[k].append(lab)
    f.append(f"[{i_main}:a]{fmt},asetpts=PTS-STARTPTS,apad=whole_dur={main_dur:.3f},atrim=0:{main_dur:.3f}[amain]")
    main_labels = split("[amain]", "am")
    if music is not None:
        i_mus = add("-stream_loop", "-1", "-i", str(music))
        fo = min(2.5, main_dur / 3)
        target_lufs = float(getattr(rs, "music_lufs", -14.0) or -14.0)
        f.append(f"[{i_mus}:a]{fmt},atrim=0:{main_dur:.3f},asetpts=PTS-STARTPTS,"
                 f"afade=t=in:st=0:d=1.5,afade=t=out:st={main_dur - fo:.3f}:d={fo:.3f},"
                 f"volume={rs.music_volume:.3f}[mus]")
        f.append(f"{main_labels[0]}volume={rs.game_volume:.3f}[gduck]")
        # Mix then loudnorm so the finished music bed lands near -14 LUFS (not ~-10).
        f.append("[gduck][mus]amix=inputs=2:duration=first:normalize=0,"
                 f"loudnorm=I={target_lufs:.1f}:TP=-1.5:LRA=11[amix]")
        aparts[0].append("[amix]")
        if want_nm:
            aparts[1].append(main_labels[1])
    else:
        aparts[0].append(main_labels[0])
    if end_at_end:
        i_sil = add("-f", "lavfi", "-t", f"{E:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000")
        f.append(f"[{i_sil}:a]{fmt}[aend]")
        for k, lab in enumerate(split("[aend]", "ae")):
            aparts[k].append(lab)
    for k in range(copies):
        if len(aparts[k]) > 1:
            f.append(f"{''.join(aparts[k])}concat=n={len(aparts[k])}:v=0:a=1[aout{k}]")
        else:
            f.append(f"{aparts[k][0]}anull[aout{k}]")

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    for g in inputs:
        cmd += g
    threads = ctl.thread_count(rs.threads)
    cmd += [*ffmpeg_threads_args(threads), "-filter_complex", ";".join(f),
            "-map", "[vout]", "-map", "[aout0]", *x264_args(rs.preset, rs.crf, threads),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", "-t", f"{total:.3f}", str(out_path)]
    nm_audio = tmp / "nomusic_audio.m4a"
    if want_nm:
        cmd += ["-map", "[aout1]", "-vn", "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.3f}", str(nm_audio)]
    cmd += ["-progress", "pipe:1", "-nostats"]
    log("Assembling: " + summarize_ffmpeg(cmd))
    run_ffmpeg_progress(cmd, total, ctl, log, lambda fr, sp: prog.update(fr, f"Assembling {out_path.name}", speed=sp))
    if want_nm:
        kw = {k: v for k, v in popen_kwargs(ctl.priority_mode).items() if k == "creationflags"}
        r = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(out_path), "-i", str(nm_audio),
                            "-map", "0:v", "-map", "1:a", "-c", "copy", "-movflags", "+faststart",
                            str(nomusic_path)], capture_output=True, text=True, **kw)
        if r.returncode != 0:
            raise RuntimeError("Could not write the no-music copy:\n" + r.stderr[-600:])
        log(f"No-music copy: {nomusic_path}")


def render_all(ffmpeg: str, ffprobe: str, info: SourceInfo, clips: list[Clip],
               keys: list[Keyframe], hoops: list[Hoop], bystanders: list,
               detections: Optional[Detections], out_path: Path, rs: RenderSettings,
               work: Path, ctl: JobController, prog: Progress, log: Callable[[str], None]) -> list[Path]:
    import cv2
    cv2.setNumThreads(ctl.thread_count(rs.threads))
    if not clips:
        raise ValueError("No clips are switched on. Include at least one clip.")
    by_clip: dict[str, list[Keyframe]] = {}
    for k in keys:
        by_clip.setdefault(k.clip_id, []).append(k)
    frames_per_fmt = max(1, sum(int(c.duration * info.fps) for c in clips))
    outputs: list[Path] = []
    for fmt in rs.formats:
        out_w, out_h = output_size(fmt, info, rs.resolution)
        if fmt == "9:16":
            dest = out_path.with_name(out_path.stem + "_short" + out_path.suffix)
        else:
            dest = out_path
        nm = dest.with_name(dest.stem + "_nomusic" + dest.suffix) if (rs.no_music_copy and rs.music) else None
        tmp = work / f"render_{fmt.replace(':', 'x')}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        log(f"Rendering {fmt} at {out_w}x{out_h}: {len(clips)} clips, "
            f"{sum(c.duration for c in clips):.0f} s of footage")
        log("Zoom eases in toward the basket, then back out (about 1.8x at most).")
        prog.begin(f"Rendering clips ({fmt})")
        segs: list[Path] = []
        chunk_dir: Optional[Path] = None
        chunk_parts: list[Path] = []
        chunk_paths: list[Path] = []
        chunk_dur = 0.0
        if rs.preview_chunks:
            chunk_dir = out_path.parent / f"{out_path.stem}_chunks" / ("16x9" if fmt == "16:9" else "9x16")
            shutil.rmtree(chunk_dir, ignore_errors=True)
            chunk_dir.mkdir(parents=True, exist_ok=True)
            log(f"Preview parts folder: {chunk_dir.parent}")

        def flush_chunk() -> None:
            nonlocal chunk_dur
            if not chunk_parts or chunk_dir is None:
                return
            part = chunk_dir / f"part_{len(chunk_paths) + 1:02d}.mp4"
            concat_copy(ffmpeg, list(chunk_parts), part, ctl)
            chunk_paths.append(part)
            log(f"Preview part {len(chunk_paths)} ready ({chunk_dur:.0f} s): {part}")
            chunk_parts.clear()
            chunk_dur = 0.0

        hf = base_hfov(fmt, info)
        done = 0
        for n, c in enumerate(clips, 1):
            ks = by_clip.get(c.id)
            if not ks:
                if detections is not None:
                    ks, _ = plan_clip(c, detections, hoops, bystanders, hf, flat=info.kind == "flat")
                    log(f"Clip {c.id}: no camera keyframes in the CSV, planned them automatically.")
                else:
                    ks = _fallback_keys(c, hoops)
                    log(f"Clip {c.id}: no keyframes and no detections; camera holds on hoop {c.hoop}.")
            seg = tmp / f"seg_{n:03d}.mp4"
            log(f"Clip {n}/{len(clips)} {c.id} ({c.type}, hoop {c.hoop}) {format_ts(c.start)}-{format_ts(c.end)}")

            def on_frame(i: int, base=done, n=n) -> None:
                prog.update((base + i) / frames_per_fmt, f"Clip {n}/{len(clips)} ({fmt})")

            render_clip_segment(ffmpeg, info, c, ks, fmt, out_w, out_h, rs, seg, ctl, on_frame, log)
            done += int(c.duration * info.fps)
            segs.append(seg)
            if rs.preview_chunks:
                if chunk_closes_before(chunk_dur, c.duration):
                    flush_chunk()
                chunk_parts.append(seg)
                chunk_dur += c.duration
                if chunk_dur >= PREVIEW_CHUNK_S:
                    flush_chunk()
        if rs.preview_chunks:
            flush_chunk()
            if not chunk_paths:
                raise RuntimeError("No preview parts were written.")
            log(f"Building the final video from {len(chunk_paths)} preview part"
                f"{'s' if len(chunk_paths) != 1 else ''}.")
        prog.begin(f"Assembling ({fmt})")
        assemble(ffmpeg, ffprobe, chunk_paths if rs.preview_chunks else segs, dest, nm,
                 out_w, out_h, info.fps_str, rs, tmp, ctl, prog, log)
        outputs.append(dest)
        if nm:
            outputs.append(nm)
        shutil.rmtree(tmp, ignore_errors=True)
    return outputs
