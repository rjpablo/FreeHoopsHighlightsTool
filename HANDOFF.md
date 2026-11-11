# FreeHoops Highlights Tool v1.0: handoff notes

This file is for whoever works on the tool next, such as a local coding agent. The user is rj pablo (FreeHoops, Philippines). He records pickup games with an Insta360 camera on a tripod by the court and posts highlight videos and Shorts to YouTube and TikTok.

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
- **Made-basket detection.** It works from close-up "rim views" of each hoop, plus warm-up detection, clip padding, and dedupe/merge.
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
                             analysis pass, hoop auto-guess, made-basket events, warm-up detection
    fhh_camera.py            virtual-camera planning (ball/player blend, shooter rule, smoothing),
                             Catmull-Rom keyframe interpolation
    fhh_project.py           Clip/Keyframe/Project data, CSV + JSON I/O, events -> clips,
                             balanced selection, bystander cells
    fhh_render.py            per-clip render (frames piped to FFmpeg), final assembly
                             (intro, title, logo, lower thirds, end card, music, no-music copy)
    fhh_jobs.py              analyze() and render() jobs used by both GUI and CLI
    fhh_gui.py               Tkinter window, hoop picker, clip preview
    graphics.py              card/title/logo rendering with Pillow (from the Promo Tool, extended)
    qrcodegen.py             Nayuki QR generator (MIT)
  runtime/                   ffmpeg.exe, ffprobe.exe (gyan.dev 9.0.2 full), fonts/, licenses/, manifest.json
  assets/                    intro.mov (ProRes 4444 alpha, 6.8 s), intro_audio.wav, logo.png,
                             music/*.m4a, calibration/insta360_x4.json, manifest.json
  models/                    yolov8n.pt, yolov8s.pt (bundled); yolov8m/x downloaded on first use;
                             optional hoop.pt (any YOLO model with a "hoop"/"rim" class)
  output/, logs/, work/      created at run time (work/<name>/detections.npz = detection cache)
  FreeHoopsHighlightsTool.settings.json   remembered GUI settings and output folder
  tools/                     make_music.py (how the bundled beats were generated)
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
- **Made basket** (`detect_events`). A candidate *above* the rim (|dx|<1.6, -4.5<dy<-0.35, in rim widths) must be followed within 0.9 s by a candidate *through* the rim (|dx|<0.55, -0.3<dy<1.7).
  - Score = 0.45·above + 0.45·through + net-motion bonus (0.25) − "beside the rim" penalty + 0.1.
  - Thresholds depend on sensitivity: made ≥ 0.62−0.3·s, attempt ≥ 0.45−0.2·s.
  - A descending ball above the rim with no "through" counts as an "attempt".
  - Events at the same hoop within 2.5 s are deduped.
- **Warm-up.** In 30 s windows, if 20% or more of the full-court frames show 2 or more distinct balls, it counts as shootaround. Clips before the end of that run are switched off (but kept in the list).
- **Bystanders.** Grid cells (3°) occupied in 45% or more of frames are treated as bench/spectators and ignored by the camera.
- The detection cache is `work/<output name>/detections.npz`. The render uses it to plan cameras for clips added by hand.

### 3.3 Clips (`fhh_project.py`)
- Each event becomes a clip from `event − pre` (default 6 s) to `event + post` (default 3 s). If the release was estimated, the clip starts at `release − 2 s` or earlier, so long shots show the shooter.
- **Overlaps.** Overlaps at the same hoop are merged into one longer clip. Heavy overlaps across hoops keep the stronger clip, and small ones are trimmed at the midpoint.
- Made baskets are switched on. Attempts are switched on only with the setting.
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
   - audio: game sound ducked to 0.35 with music at 0.8 (1.5 s fade in, 2.5 s fade out, looped to length)
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

**Testing status.** v1.0 was written without running it. At rj's request there was no end-to-end run, no GUI test, and no screenshots. Only `python -m py_compile` and pyflakes were run. Expect first-run bugs; check the log file in `logs/`.

**Untested areas**
- Everything on Windows: install.bat with Python 3.14 (PyTorch must have 3.14 wheels; the fallback is a 3.12/3.13 venv), the CUDA detection, Pause (NtSuspendProcess) and priority, and the Tk layout on a laptop screen.
- **Stitching at 1920 px lenses.** The X4 calibration was fitted on 2880 px lenses and scaled. If seams look doubled or bent, adjust `scale_adjust` (0.98–1.02) in the calibration JSON or switch Lens model to "generic". A proper fix is to read the per-file calibration from the .insv trailer (the `insta360-rs` crate was used earlier in /workspace/quoted_cafe/ext).
- Split `_00_`/`_10_` pairs and side-by-side single-stream files (lens order and orientation assumed).
- 9:16 framing. Keyframes are planned for the 85° view and reused with a 52° view, so the shooter-and-rim framing may need a portrait-specific plan.

**Expected detection accuracy (estimate, not measured)**
- Made-basket detection depends on the ball being visible in the rim view. With picked hoops and a clear view, expect many but not all makes. Expect false clips from dunks, net touches, rebounds above the rim, and players walking under the hoop. Swishes with heavy motion blur may be missed.
- The earlier v2 pipeline only "locked" the ball in about 6–8% of full-court frames, so the camera often follows the players more than the ball.
- The review step exists for these reasons.
- The automatic hoop guess is rough. Picking hoops is strongly recommended.

**Performance.** On CPU only, detection on Balanced may take about as long as the video, or longer. Fast is about 2–3× quicker. The render is about 3–8 fps of output on a laptop CPU at 1080p.

**Music.** The two bundled beats are simple procedurally generated tracks (safe to use, but basic). A YouTube Audio Library track can be picked with "Custom file…".

**Suggested next steps**
1. Run on one real game and tune the detection thresholds (`detect_events`) and rim-view zones against a hand-labelled list of makes.
2. Train or fine-tune a small YOLO model on rj's footage (ball + rim classes) and drop it in as `models/hoop.pt` (and use its ball class in rim views).
3. Read the Insta360 per-file calibration (gyro/offset string in the .insv trailer) instead of the fixed X4 fit, and use gyro data for horizon levelling.
4. Plan a portrait-specific camera path for 9:16.
5. Add "plays" beyond made baskets (blocks, steals, and-ones) using player-speed/audio peaks.
6. Add GPU decode (`-hwaccel cuda`/`d3d11va`) and an optional GPU remap to speed up the render.
