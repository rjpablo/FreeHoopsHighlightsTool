FreeHoops Highlights Tool 1.2
=============================

Makes a basketball highlights video straight from an Insta360 .insv file
(or a stitched 360 MP4, or a normal flat game video):

  stitch the 360 footage -> follow the play with a virtual camera ->
  find made baskets -> you review the clip list -> render with the FreeHoops
  intro, "<date> Highlights" title, corner logo, promo cards, end card with
  QR code, and music (plus a no-music copy for Marketing/TikTok).

Your original videos are only read, never changed.

How baskets are chosen (and how that matches the hand-curated workflow) is
documented in docs\HIGHLIGHT_STRATEGY.md.


First-time setup (once)
-----------------------
1. You need Python 3.10 or later (3.14 works) from python.org, with
   "Add python.exe to PATH" and the tcl/tk option ticked.
2. Unzip FreeHoopsHighlightsTool_v1.2.zip, e.g. to Videos\FreeHoopsHighlightsTool
3. Double-click install.bat. It creates a private .venv folder and
   downloads PyTorch, Ultralytics YOLO, OpenCV, NumPy and Pillow
   (about 1 GB on a normal PC; about 3.5 GB if you have an NVIDIA graphics
   card, which it uses automatically and which makes detection much faster).
4. Double-click run.bat to open the window. (run.bat also runs the
   install automatically the first time.)

FFmpeg is included in runtime\. Fonts (DejaVu Sans) and two FreeHoops beats
are included. The small (yolov8n) and standard (yolov8s) detection models are
in models\. Accurate and Max download a bigger model the first time you use them.


Using the window
----------------
1. Video: choose the .insv. For split recordings (..._00_123.insv and
   ..._10_123.insv) pick either one; the other is found automatically.
   Optional Start / End (mm:ss or hh:mm:ss) limit the part that is used,
   e.g. to skip warm-ups. The output folder defaults to the output folder
   inside the program and is remembered once you change it; "Open folder"
   always works.
2. Find highlights:
   - Pick hoops...: a frame of the court appears. Click on a rim, then in
     the zoomed view click the rim's left edge and right edge. Do the same
     for the second hoop, then Done. This takes a minute and makes the
     detection far better than the automatic guess.
   - Detection quality: Fast / Balanced / Accurate / Max. Without an NVIDIA
     GPU everything runs on the processor and is slow. Use Fast or Background
     performance if you want to keep using the PC.
   - Clip padding defaults to about 3.5 s before and 1 s after each basket
     (short clips like the hand-curated edits).
   - Optional score log CSV: times of each score / ball-in. When set, those
     times seed the baskets and each is still checked on the rim close-up.
     Leave blank for the no-scoreboard path (scan + rim verify).
   - Click "Find highlights". You get a clip list plus two CSV files.
3. Review clips: switch clips on/off, edit times, add or delete clips,
   preview a frame through the virtual camera. Every change is saved to the
   CSV straight away. You can also edit the CSV files in Excel and press
   "Reload CSV".
4. Output: 16:9 video and/or 9:16 Short, title text, intro, corner logo,
   music (two original FreeHoops beats are included, or choose your own
   royalty-free file; the mix targets about -14 LUFS), the promo lower third,
   and the end card. Then "Render highlights".

Pause / Resume and Cancel work for both steps. The log shows each step and
is also saved in logs\. Performance (Full / Balanced / Background / Custom)
limits CPU use and priority; you can change the priority while it runs.


Files it writes (next to the video)
-----------------------------------
  <name>_highlights.mp4            the 16:9 video
  <name>_highlights_short.mp4      the 9:16 Short
  ..._nomusic.mp4                  the same without music
  <name>_highlights.csv            clip list (id, include, order, type, hoop,
                                   start, end, event_time, score, note)
  <name>_highlights_camera.csv     camera keyframes per clip (clip_id, time,
                                   yaw, pitch, zoom, note)
  <name>_highlights_project.json   source file, time range, hoops, settings

CSV rules: times are mm:ss.ss in the ORIGINAL video. include = yes/no (or
start the id with # to switch a row off). order sets the clip order in the
video. Camera yaw is left/right in degrees (bigger = further right), pitch is
up/down (bigger = up), zoom 1.0 = normal, 1.3 = tighter. Clips without
keyframes are planned automatically when you render.


Command line
------------
  run.bat analyze -i "D:\Videos\VID_20261105_202800_00_021.insv" --start 2:00 --end 18:30 --hoops "A:-62.5,8.1,1.9;B:118.0,7.4,1.7"
  run.bat analyze -i "game.mp4" --score-log "score_times.csv" --quality fast
  run.bat render --data "output\VID_20261105_202800_021_highlights.csv" --format both --music beat1
  run.bat auto -i "game.insv" --quality fast --performance background
  run.bat eval --pred output\game_highlights.csv --truth tests\fixtures\clip_list_v2.csv
  run.bat analyze --help      (all options)


Updating
--------
Updates usually contain only the app folder (a small zip). Unzip it over the
program folder and choose Replace. Settings, models, logs, the .venv and your
videos are not touched. See UPDATE.txt and HANDOFF.md.
