#!/usr/bin/env python3
"""FreeHoops Highlights Tool

Make a basketball highlights video from an Insta360 .insv (or a stitched 360
MP4, or a flat game video): stitch, follow the play with a virtual camera,
find made baskets, let you review the clip list, then render with the
FreeHoops intro, title, logo, promo cards, and music.

Run without arguments for the window. Command line:
  analyze  -i VIDEO [options]      find highlights, write the CSV files
  render   --data CLIPS.csv        render from the (edited) CSV files
  auto     -i VIDEO [options]      analyze and render in one go
Use --help on each command for the options.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fhh_core import (  # noqa: E402
    APP_NAME, DEFAULT_PERF_MODE, PERF_MODES, VERSION, JobController, ProgressInfo,
    RenderCancelled, RunLogger, X264_PRESETS, component_issues, default_output_for,
    format_eta, parse_optional_time, parse_time,
)
from fhh_detect import QUALITY, Hoop  # noqa: E402
from fhh_jobs import Job, analyze, default_title, music_tracks, render  # noqa: E402
from fhh_project import ClipSettings  # noqa: E402
from fhh_render import CardWindow, RenderSettings  # noqa: E402
from graphics import DEFAULT_QR_URL, END_SUB, END_TITLE, HL_SUB, LT_SUB, LT_TITLE, QR_CAPTION  # noqa: E402


def parse_hoops(text: str) -> list[Hoop]:
    """'A:yaw,pitch,width;B:yaw,pitch,width' (width optional)."""
    hoops: list[Hoop] = []
    for i, part in enumerate(p for p in (text or "").split(";") if p.strip()):
        name = "AB"[i] if i < 2 else f"H{i + 1}"
        if ":" in part:
            name, part = part.split(":", 1)
        nums = [float(x) for x in part.split(",") if x.strip()]
        if len(nums) < 2:
            raise ValueError(f"Hoop '{part}' needs at least yaw,pitch")
        from fhh_detect import estimate_rim_width
        hoops.append(Hoop(name.strip() or "A", nums[0], nums[1],
                          nums[2] if len(nums) > 2 else estimate_rim_width(nums[1]), "manual"))
    return hoops


def resolve_music(value: str | None) -> str:
    if value is None:
        tracks = music_tracks()
        return str(tracks[0]) if tracks else ""
    v = value.strip()
    if v.lower() in ("", "none", "off", "no"):
        return ""
    tracks = music_tracks()
    if v.lower().startswith("beat") and v[4:].isdigit():
        k = int(v[4:]) - 1
        if 0 <= k < len(tracks):
            return str(tracks[k])
    return v


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--performance", choices=PERF_MODES, default=DEFAULT_PERF_MODE,
                   help="full = all cores; balanced = half, below-normal priority; "
                        "background = quarter, idle priority; custom = --threads")
    p.add_argument("--threads", type=int, help="thread count for --performance custom")
    p.add_argument("--lens-model", choices=["x4", "generic"], default="x4",
                   help="fisheye stitching model (x4 = Insta360 X4 calibration, generic = equidistant 200 deg)")
    p.add_argument("--source-kind", choices=["auto", "streams", "split", "sbs", "equirect", "flat"], default="auto")
    p.add_argument("--device", default="auto", help="auto (CUDA if available), cpu, or cuda:0")
    p.add_argument("--overwrite", action="store_true", help="replace existing output videos")


def add_analyze(p: argparse.ArgumentParser) -> None:
    p.add_argument("-i", "--input", required=True, help=".insv (either lens file of a split pair), 360 MP4, or flat video")
    p.add_argument("-o", "--output", help="output video path; the CSV files are written next to it")
    p.add_argument("--start", help="only use the video from here (mm:ss or hh:mm:ss)")
    p.add_argument("--end", help="only use the video up to here (mm:ss or hh:mm:ss)")
    p.add_argument("--quality", choices=list(QUALITY), default="balanced")
    p.add_argument("--sensitivity", type=float, default=0.5, help="0 strict .. 1 catch more (more false clips)")
    p.add_argument("--hoops", help="'A:yaw,pitch,width;B:yaw,pitch,width' in degrees; omit to find them automatically")
    p.add_argument("--pre", type=float, default=6.0, help="seconds before each basket")
    p.add_argument("--post", type=float, default=3.0, help="seconds after each basket")
    p.add_argument("--attempts", action="store_true", help="also switch on shots at the rim that were not detected as made")
    p.add_argument("--keep-warmup", action="store_true", help="do not switch off clips during a detected warm-up")
    p.add_argument("--max-length", type=float, default=0.0, help="max seconds of clips, alternating hoops (0 = all)")


def add_render(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format", choices=["16:9", "9:16", "both"], default="16:9")
    p.add_argument("--resolution", choices=["auto", "720", "1080", "1440", "2160"], default="auto")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", choices=X264_PRESETS, default="medium")
    p.add_argument("--no-intro", action="store_true")
    p.add_argument("--title", help="title line (default '<date> Highlights')")
    p.add_argument("--subtitle", default=HL_SUB)
    p.add_argument("--title-duration", type=float, default=4.0)
    p.add_argument("--no-title", action="store_true")
    p.add_argument("--no-logo", action="store_true")
    p.add_argument("--music", help="beat1, beat2, a file path, or none (default: beat1)")
    p.add_argument("--music-volume", type=float, default=0.8)
    p.add_argument("--game-volume", type=float, default=0.35, help="game sound level under the music")
    p.add_argument("--no-nomusic-copy", action="store_true", help="skip the extra no-music copy")
    p.add_argument("--promo", action="append", metavar="START,DURATION",
                   help="lower-third promo card at a time in the OUTPUT video (repeatable), e.g. 0:12,6")
    p.add_argument("--no-promo", action="store_true")
    p.add_argument("--lt-line1", default=LT_TITLE)
    p.add_argument("--lt-line2", default=LT_SUB)
    p.add_argument("--end-card", default="end", help="'end' to append it, or a start time in the output video")
    p.add_argument("--no-end-card", action="store_true")
    p.add_argument("--end-duration", type=float, default=7.0)
    p.add_argument("--end-title", default=END_TITLE)
    p.add_argument("--end-sub", default=END_SUB)
    p.add_argument("--qr-caption", default=QR_CAPTION)
    p.add_argument("--qr-url", default=DEFAULT_QR_URL)
    p.add_argument("--preview-chunks", action="store_true",
                   help="save the reframed clips as playable parts of about 1 minute, then build the final video from them")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="FreeHoopsHighlightsTool", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{APP_NAME} {VERSION}")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("analyze", help="find highlights and write the CSV files")
    add_analyze(a)
    add_common(a)
    r = sub.add_parser("render", help="render from the CSV files")
    r.add_argument("--data", required=True, help="the <name>_highlights.csv clip list")
    r.add_argument("-i", "--input", help="source video (default: the one recorded in the project file)")
    r.add_argument("-o", "--output", help="output video (default: <data name>.mp4)")
    add_render(r)
    add_common(r)
    b = sub.add_parser("auto", help="analyze and render without stopping for review")
    add_analyze(b)
    add_render(b)
    add_common(b)
    return ap


def render_settings_from_args(args, title_default: str) -> RenderSettings:
    rs = RenderSettings()
    rs.formats = ["16:9", "9:16"] if args.format == "both" else [args.format]
    rs.resolution = args.resolution
    rs.crf, rs.preset = args.crf, args.preset
    rs.intro_on = not args.no_intro
    rs.title_on = not args.no_title
    rs.title_main = args.title if args.title is not None else title_default
    rs.title_sub = args.subtitle
    rs.title_duration = args.title_duration
    rs.logo_on = not args.no_logo
    rs.music = resolve_music(args.music)
    rs.music_volume, rs.game_volume = args.music_volume, args.game_volume
    rs.no_music_copy = not args.no_nomusic_copy
    rs.lt_on = not args.no_promo
    if args.promo:
        rs.lt_windows = []
        for v in args.promo:
            s, _, d = v.partition(",")
            rs.lt_windows.append(CardWindow(parse_time(s), parse_time(d or "6")))
    rs.lt_line1, rs.lt_line2 = args.lt_line1, args.lt_line2
    rs.end_on = not args.no_end_card
    if args.end_card.strip().lower() in ("end", "at-end", "at end"):
        rs.end_at_end = True
    else:
        rs.end_at_end = False
        rs.end_start = parse_time(args.end_card)
    rs.end_duration = args.end_duration
    rs.end_title, rs.end_sub = args.end_title, args.end_sub
    rs.qr_caption, rs.qr_url = args.qr_caption, args.qr_url
    rs.preview_chunks = bool(getattr(args, "preview_chunks", False))
    return rs


def run_cli(args) -> int:
    ctl = JobController()
    ctl.reset()
    logger = RunLogger(also=lambda line: print(line, flush=True))
    last = {"pct": -1}

    def progress(info: ProgressInfo) -> None:
        pct = int(info.fraction * 100)
        if pct != last["pct"]:
            last["pct"] = pct
            print(f"[{pct:3d}%] {info.message}  ETA {format_eta(info.eta)}", flush=True)

    def on_sigint(_s, _f):
        print("\nCancelling...", flush=True)
        ctl.request_cancel()

    signal.signal(signal.SIGINT, on_sigint)
    try:
        if args.cmd in ("analyze", "auto"):
            inp = Path(args.input)
            out = Path(args.output) if args.output else default_output_for(inp)
            job = Job(input_path=inp, output_path=out,
                      start=parse_optional_time(args.start), end=parse_optional_time(args.end),
                      quality=args.quality, sensitivity=args.sensitivity, lens_model=args.lens_model,
                      device=args.device, source_kind=args.source_kind,
                      hoops=parse_hoops(args.hoops) if args.hoops else [],
                      clip=ClipSettings(pre=args.pre, post=args.post, include_attempts=args.attempts,
                                        skip_warmup=not args.keep_warmup, max_total=args.max_length),
                      perf_mode=args.performance, perf_threads=args.threads, overwrite=args.overwrite)
            res = analyze(job, ctl, progress, logger)
            if args.cmd == "auto":
                job.render = render_settings_from_args(args, default_title(res["info"], inp))
                ctl.reset()
                render(job, res["paths"]["clips"], ctl, progress, logger)
        else:
            data = Path(args.data)
            out = Path(args.output) if args.output else data.with_suffix(".mp4")
            job = Job(input_path=Path(args.input) if args.input else None, output_path=out,
                      lens_model=args.lens_model, device=args.device, source_kind=args.source_kind,
                      perf_mode=args.performance, perf_threads=args.threads, overwrite=args.overwrite)
            src_name = Path(args.input).name if args.input else data.name
            job.render = render_settings_from_args(args, default_title(None, Path(src_name)))
            render(job, data, ctl, progress, logger)
        return 0
    except RenderCancelled:
        logger("Cancelled.")
        return 2
    except Exception as exc:
        logger(f"ERROR: {exc}")
        return 1
    finally:
        logger.close()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    issues = component_issues()
    if not argv or argv == ["--gui"]:
        from fhh_gui import run_gui
        return run_gui(issues)
    if issues:
        for i in issues:
            print("ERROR: " + i, file=sys.stderr)
        return 1
    args = build_parser().parse_args(argv)
    if not args.cmd:
        build_parser().print_help()
        return 0
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
