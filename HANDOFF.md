# FreeHoops Highlights Tool v1.2: handoff notes

This file is for whoever works on the tool next, such as a local coding agent. The user is rj pablo (FreeHoops, Philippines). He records pickup games with an Insta360 camera on a tripod by the court and posts highlight videos and Shorts to YouTube and TikTok.

**Highlight strategy** (no-scoreboard path A, scoreboard path B, rejection rules, clip timing, evaluation) is documented in `docs/HIGHLIGHT_STRATEGY.md`. Read that before changing detection.

## 1. Purpose and features

The tool turns a basketball game recording into a highlights video:

- **Inputs.** Supported inputs:
  - Insta360 `.insv` with two video streams (X3/X4 style, one stream per lens)
  - split `_00_` / `_10_` `.insv` pairs
  - single-stream side-by-side dual fisheye
  - already-stitched equirectangular 360 MP4
  - a flat (normal) video, where the "camera" becomes a moving crop
- **Time range.** Optional start and end times (GUI fields, or `--start`/`--end`) limit processing to that part of the video. A blank field means the whole video.
- **Stitching.** The two fisheye lenses are stitched with rj's fitted Insta360 X4 lens model, with a feathered seam.
- **Virtual camera.** It follows the play automatically, the same way as rj's earlier "v2 auto-zoom" edits.
- **Made-basket detection.** Broad rim candidates are verified with a downward through-rim / net check. Warm-ups (multi-ball and lone shooters), camera setup, and occlusion are rejected. Optional score-log / scoreboard seeds (path B). Clip padding defaults to ~3.5 s pre / 1 s post; nearby makes are not merged into long clips.
- **Review step.** Clips can be switched on/off, edited, added, deleted, or previewed in the GUI. The data is also kept in plain CSV files that can be edited in Excel and re-rendered.
- **Render.** It produces:
  - 16:9 and/or 9:16 (Short) outputs
  - the FreeHoops intro animation and a two-line title ("<date> Highlights" / "Next Possession Open Run")
  - a corner logo (transparent PNG)
  - customizable promo cards (from the FreeHoops Promo Tool):
    - lower third with editable text, defaults "FreeHoops.net" / "Get FreeHoops Mobile on Google Play"
    - any number of lower-third windows (start + duration in the *output* timeline)
    - end card with QR code (editable title, subtitle, QR caption and QR link), shown either at the end or at a chosen time
    - each card can be turned on or off, and all cards scale for 16:9 and 9:16
  - music under the game sound, plus an automatic **no-music copy** for Marketing/TikTok
- **Job controls.** Progress bar with ETA, log panel and log files, Pause/Resume, and Cancel. Performance modes are Full / Balanced / Background / Custom (thread limits plus process priority).
- **Output folder.** It defaults to `output/` in the program folder and is remembered once changed. The file name follows the input, and an "Open folder" button is always available.
- **Originals are never modified.** Inputs are only read, and the output can never be the input path.

## 2. Architecture and file layout

```
FreeHoopsHighlightsTool/
  run.bat, install.bat       start / set up (.venv with pip deps)
  requirements.txt           ultralytics, opencv-python, numpy, pillow (torch installed first by install.bat)
  app/                       the Python program (only this changes in normal updates)
    manifest.json            {"version": "1.0", "requires_runtime": "1.0", "requires_assets": "1.0"}
    FreeHoopsHighlightsTool.py  entry point: GUI (no args) or CLI (analyze / render / auto)
    fhh_core.py              paths, manifests, settings, time parsing, performance/priority,
                             JobController (pause/cancel), Progress, logging, ffmpeg helpers
                             (much of it copied from the Promo Tool v1.3)
    fhh_source.py            input detection, FFmpeg frame reader, lens models + Stitcher,
                             perspective reprojection, DirectFisheyeView, flat crop camera
    fhh_detect.py            YOLO wrapper, weight download, court-band tiling, rim views,
                             analysis pass, hoop auto-guess, rim verification, rejection rules,
                             warm-up / camera-setup / lone-shooter detection
    fhh_scoreboard.py        score-log CSV loader, scoreboard-region change detection
    fhh_ai_review.py         optional vision-API make confirmation (off by default)
    fhh_camera.py            virtual-camera planning (ball/player blend, shooter rule, smoothing),
                             Catmull-Rom keyframe interpolation
    fhh_project.py           Clip/Keyframe/Project data, CSV + JSON I/O, events -> clips,
                             short-clip defaults, putback-only merge, balanced selection
    fhh_render.py            per-clip render (frames piped to FFmpeg), final assembly
                             (intro, title, logo, lower thirds, end card, music @ ~-14 LUFS)
    fhh_jobs.py              analyze() and render() jobs used by both GUI and CLI
    fhh_gui.py               Tkinter window, hoop picker, clip preview, score-log field
    graphics.py              card/title/logo rendering with Pillow (from the Promo Tool, extended)
    qrcodegen.py             Nayuki QR generator (MIT)
  runtime/                   ffmpeg.exe, ffprobe.exe (gyan.dev 9.0.2 full), fonts/ (DejaVu), licenses/, manifest.json
  assets/                    intro.mov (ProRes 4444 alpha, 6.8 s), intro_audio.wav, logo.png,
                             music/*.m4a (tracked), calibration/insta360_x4.json, manifest.json
  models/                    yolov8n.pt, yolov8s.pt (bundled); yolov8m/x downloaded on first use;
                             optional hoop.pt (any YOLO model with a "hoop"/"rim" class)
  docs/HIGHLIGHT_STRATEGY.md detection / clip strategy and evaluation notes
  tests/                     unit tests + fixtures/clip_list_v2.csv (Nov 12 hand labels)
  tools/                     make_music.py, eval_highlights.py
  output/, logs/, work/      created at run time (work/<name>/detections.npz = detection cache)
  FreeHoopsHighlightsTool.settings.json   remembered GUI settings and output folder
```

The program checks the manifest versions at start-up (`component_issues()` in `fhh_core.py`). If `runtime/` or `assets/` is older than the app requires, it tells the user to unzip the full package.

## 3. Pipeline

### 3.1 Input detection and stitching (`fhh_source.py`)
- `detect_source()` uses ffprobe. The rules:
  - two square video streams: `streams`
  - a square `.insv` with a `_00_`/`_10_` partner: `split`
  - a 2:1 frame with black corners: `sbs`
  - a 2:1 frame with spherical metadata or no black corners: `equirect`
  - anything else: `flat`
  - `--source-kind` overrides the detection.
- Frames are decoded by FFmpeg (`hstack` of both lenses) as raw BGR through a pipe (`FrameReader`). `-ss` before `-i` gives fast, accurate seeking.
- **Lens model.** The model is the Insta360 X4 factory "V3" unified omnidirectional (Mei) model plus a fitted scale and relative rotation. It was fitted on rj's camera (Quoted Cafe .insv to 360 conversion, 2880 px lenses). It lives in `assets/calibration/insta360_x4.json` and is scaled for other lens sizes (rj's basketball files have 1920 px lenses). `scale_adjust` and `yaw_offset_deg` are tuning knobs. "generic" (an equidistant 200° model) is a fallback.
- `Stitcher` builds equirect lookup maps once, with a 3° feathered seam between lenses. Analysis uses only a latitude band (-40°..+50°) at 2× lens width (3840 px for 1920 lenses).
- The final render uses `DirectFisheyeView`. Lens coordinates for each output pixel are computed on a coarse grid, then both lenses are sampled and feather-blended. There is no full 360 stitch per frame, and only one resampling step.

### 3.2 Detection (`fhh_detect.py`)
One decode of the chosen range at the analysis fps (quality preset):

| preset | model | fps | full-court every Nth frame | tile imgsz | rim view px / imgsz |
|---|---|---|---|---|---|
| fast | yolov8n | 8 | 4 | 640 | 320 / 480 |
| balanced | yolov8s | 10 | 4 | 640 | 384 / 640 |
| accurate | yolov8m | 12 | 3 | 960 | 448 / 640 |
| max | yolov8x | 15 | 3 | 1280 | 512 / 800 |

- **Full-court pass.** COCO person (0) and sports ball (32) are detected at conf 0.1 on square tiles over the court band (-35°..+25°), with wrap-around. This matches the v2 pipeline (which used YOLOv8x at conf 0.1). Detections become scene coordinates (yaw/pitch degrees).
- **Rim views.** On every analysis frame, a perspective close-up of each hoop is made (FOV ≈ 11 rim widths, with the rim slightly below centre). Frame differencing gives motion, and nothing is run when nothing moves near the rim. When something does move, YOLO looks for the ball (conf 0.08–0.18 depending on sensitivity). Motion blobs of ball size count as weak candidates. Net-zone motion energy is recorded.
- **Hoops.** The best source is the user picking them (left and right rim edges give the centre and apparent width). Next is `models/hoop.pt` if present. The last resort is a rough guess from where balls are seen most above head height.
- **Made basket** (`detect_events` / `verify_rim_make`). Broad seeds (above/through/net spikes) are verified: downward path above → through rim → into net (or net-motion spike). Roll-outs after “through” are rejected. Scoreboard path uses `events_from_score_times` then the same rim check.
- **Rejection.** Camera setup (early empty / huge person / first ~2.5 s), warm-ups (multi-ball and lone shooters before a 5+ person game), occlusion, and other-side-of-court. Rejected events stay in the CSV switched off.
- **Bystanders.** Grid cells (3°) occupied in 45% or more of frames are treated as bench/spectators and ignored by the camera.
- The detection cache is `work/<output name>/detections.npz`. The render uses it to plan cameras for clips added by hand.

### 3.3 Clips (`fhh_project.py`)
- Each event becomes a clip from `event − pre` (default **3.5 s**) to `event + post` (default **1.0 s**, never less). Cap ~6.5 s per clip.
- **Merges.** Same-hoop events merge only within ~2.2 s (putbacks). Separate makes a few seconds apart stay separate clips (fixes the old 20+ s merges).
- Made baskets are switched on. Attempts / rejected / warm-up / other-side are switched off but kept for review.
- "Max total" picks the best clips round-robin across hoops, so both ends of the court get mixed in. The render order is `order`, then `start`.

### 3.4 Virtual camera (`fhh_camera.py`): rj's v2 rules
- The ball track uses constant-velocity gating (Kalman-style). Gaps up to 1.5 s are filled.
- The target is 75% ball + 25% player cluster (players within 45° of the ball). It uses players only if there is no ball, and the hoop if there is nothing at all.
- **Shooter rule.** The release is estimated by walking the ball path back from the rim; the shooter is the nearest player at release.
  - Up to release + 0.25 s, the camera frames shooter and rim together if they fit (separation ≤ 0.6·FOV). Otherwise it holds on the shooter.
  - It then pans to the rim (up to 150°/s) and holds until 1.2 s after the basket.
- **Smoothing.** Dead zone 6°, max pan 75°/s, acceleration limit 400°/s², tilt clamp -25°..+10°, and pitch smoothing with τ 0.6 s.
- **Field of view.** 85° horizontal for 16:9 and 52° for 9:16. Flat video uses a 1.5× crop for 16:9 and a full-height slice for 9:16. Zoom in the CSV divides the FOV.
- Keyframes are written every 0.5 s, plus "release"/"basket" marks. The renderer interpolates them with Catmull-Rom on unwrapped yaw.

### 3.5 CSV format
`<name>_highlights.csv`: `id, include, order, type, hoop, start, end, event_time, score, note, start_s, end_s, event_s`
- Times are `mm:ss.ss` (or `h:mm:ss.ss`) in the **source** video. The mm:ss column wins over the `_s` column.
- `include` is yes/no. An id starting with `#` also switches the row off. Duplicate ids are rejected.

`<name>_highlights_camera.csv`: `clip_id, time, yaw, pitch, zoom, note, time_s`
- Yaw is in degrees (equirect longitude; 0 = front of lens 0). Pitch is in degrees, with up positive. Zoom is 0.5–3.0.
- For flat video, these are pseudo-degrees: the full width spans 100°.
- If a clip has no rows, its camera is planned automatically at render time.

`<name>_highlights_project.json` holds: input path, kind, range, hoops `{name, yaw, pitch, width, source}`, bystander cells, warm-up end, detection cache path, and settings.

### 3.6 Rendering (`fhh_render.py`)
1. **Per clip.** The clip is decoded at full resolution and native fps. The camera view is applied to each frame, and raw BGR frames are piped to FFmpeg x264 (CRF = user CRF − 4, BT.709). The clip's own audio is added, with 0.25 s fades at each cut.
2. **Assembly.** One FFmpeg pass does all of this:
   - intro: the plate (dark plus orange bars) with the transparent `intro.mov` scaled to fit, plus `intro_audio.wav`
   - all clips through the concat demuxer, plus the end card when it is "at end"
   - overlays: logo (during the clips only), title (fade in 0.4 s, hold, fade out 1 s), each lower-third window (0.4 s alpha fades), and the end card when it is "at a time"
   - audio: game sound ducked to 0.35 with music at ~0.55, then `loudnorm` to about **-14 LUFS** (1.5 s fade in, 2.5 s fade out, looped to length)
   - when needed, a second audio-only output without music; it is remuxed with `-c copy` into `_nomusic.mp4`, so the video isn't encoded twice
3. **Output size.** "auto" gives:
   - 1080p for 360 sources (1440p for ≥ 3800 px lenses or 8K equirect)
   - the source short side for flat video (capped at 2160)
   - 9:16 is the same size turned upright (1080x1920)
4. **Card graphics.** These come from `graphics.render_highlight_graphics()`. They share layout and defaults with the Promo Tool: `make_layout()` scales from a 1080 short side and handles portrait.

## 4. rj's style preferences (keep these)
- Cut highlights from the zoomed/reframed view, never the static flat view.
- Skip pre-game warm-ups.
- Mix plays from both hoops.
- On long shots, keep the shooter visible through the release, then pan to the rim.
- Zooms should be moderate (about 1.8x max on flat video). Animate smoothly with no hard jumps.
- The intro title is "<Mon D, YYYY> Highlights" with "Next Possession Open Run" underneath. Use only the date and no game number, unless rj asks for one.
- The FreeHoops logo is the transparent PNG. The promo links are FreeHoops.net and the Google Play app (`https://play.google.com/store/apps/details?id=com.rjpablo.freehoops`).
- Use royalty-free, YouTube-safe music, and always also make a no-music version (Marketing/TikTok).
- Never modify or overwrite rj's original files. Never publish or upload anything without rj's OK.
- rj likes short, clear UI text and honest caveats.

## 5. Setup and run

Windows with Python 3.10+ (rj has 3.14.6):
```
install.bat                 (creates .venv; CUDA torch if nvidia-smi exists, else CPU torch)
run.bat                     (GUI)
run.bat analyze -i VIDEO [--start 2:00 --end 20:00] [--hoops "A:yaw,pitch,width;B:yaw,pitch,width"] [--quality fast|balanced|accurate|max]
run.bat render --data output\NAME_highlights.csv [--format 16:9|9:16|both] [--music beat1|beat2|none|PATH] [--promo 0:12,6 --promo 1:30,6] [--end-card end|START]
run.bat auto -i VIDEO ...   (analyze + render, no review)
```
For development on Linux/macOS: `python3 -m venv .venv && .venv/bin/pip install torch torchvision ultralytics opencv-python numpy pillow`, then `python app/FreeHoopsHighlightsTool.py`. FFmpeg is taken from PATH when `runtime/` holds only `.exe` files.

## 6. Shipping modular updates
- Normal fixes change only `app/`. Bump `app/manifest.json` `version` (and `VERSION` in `fhh_core.py`), then zip `app/` (plus `README.txt`/`HANDOFF.md`/`UPDATE.txt`/`run.bat`/`install.bat` if changed) as `FreeHoopsHighlightsTool_vX.Y_update.zip`. rj unzips it over the folder.
- If new FFmpeg, fonts, assets, or calibration are needed, bump that component's manifest and `requires_*` in `app/manifest.json`, and ship a full zip.
- If Python dependencies change, update `requirements.txt` and tell rj to run `install.bat` once (it is idempotent).
- **Never ship these in an update:** `.venv/`, `output/`, `logs/`, `work/`, `models/` downloads, or the settings JSON.
- Copying to rj's PC (if done by an agent): CopyFromBox in 10 MiB chunks, joined with a PowerShell .NET FileStream, then a SHA256 check.

## 7. Known limitations, untested areas, next steps

**Testing status.** v1.2 adds unit tests (`tests/test_highlight_logic.py`) and `tools/eval_highlights.py` with the Nov 12 hand-labelled fixture. There is still **no access to rj’s game footage** in this environment, so end-to-end detection on real .insv / scoreboard overlay video must be verified on his PC. Check the log file in `logs/`.

**Untested areas (need real footage)**
- Full analyze on .insv / flat 1080p games (false positives at t=0, warm-ups, rim roll-outs).
- Scoreboard overlay change detection on a real scoreboard crop.
- AI review against a live API key.
- Everything on Windows: install.bat with Python 3.14, CUDA, Pause, Tk layout.
- **Stitching at 1920 px lenses.** The X4 calibration was fitted on 2880 px lenses and scaled.
- Split `_00_`/`_10_` pairs and side-by-side single-stream files.
- 9:16 framing (portrait-specific shooter/rim plan).

**Expected detection accuracy (estimate, not measured on real video)**
- Rim verification is stricter than v1.0/v1.1; expect fewer roll-out false makes and fewer setup/warm-up clips, at the cost of some missed swishes with heavy blur.
- Score-log path should track the Nov 12 list closely once times are fed in and rim views are clear.
- The review step remains essential. Picking hoops is strongly recommended.

**Performance.** On CPU only, detection on Balanced may take about as long as the video, or longer. Fast is about 2–3× quicker. The render is about 3–8 fps of output on a laptop CPU at 1080p.

**Music.** Bundled beats are tracked in `assets/music/*.m4a` (regenerate with `tools/make_music.py`). Mix targets ~-14 LUFS.

**Suggested next steps**
1. Run analyze on a real game; score with `run.bat eval --pred ... --truth ...` against a hand-labelled CSV; tune sensitivity / rim zones.
2. Train or fine-tune a small YOLO model on rj's footage (ball + rim) as `models/hoop.pt`.
3. Read the Insta360 per-file calibration from the .insv trailer; use gyro for horizon levelling.
4. Portrait-specific camera path for 9:16.
5. Optional standout-move extras (crossovers, steals) as `type=play`.
6. GPU decode (`-hwaccel cuda`/`d3d11va`) and optional GPU remap.
