"""Tkinter window for the FreeHoops Highlights Tool."""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional

from fhh_core import (
    APP_NAME, DEFAULT_PERF_MODE, PERF_LABELS, PERF_MODES, VERSION, X264_PRESETS,
    JobController, ProgressInfo, RenderCancelled, RunLogger, default_output_dir, find_tool,
    format_eta, format_time, format_ts, load_settings, logs_dir, open_folder, output_stem_for,
    parse_optional_time, parse_time, priority_name, resolve_threads, save_settings,
    saved_output_dir, unique_name,
)
from fhh_ai_review import AIReviewSettings
from fhh_detect import QUALITY, QUALITY_LABELS, Hoop, estimate_rim_width
from fhh_jobs import Job, analyze, default_title, music_tracks, render, shots_dir_for
from fhh_project import (
    Clip, ClipSettings, data_paths, load_project, read_clips, read_keyframes, write_clips,
)
from fhh_render import CardWindow, RenderSettings
from graphics import DEFAULT_QR_URL, END_SUB, END_TITLE, HL_SUB, LT_SUB, LT_TITLE, QR_CAPTION

CUSTOM_MUSIC = "Custom file..."
NO_MUSIC = "No music"


def run_gui(issues: list[str]) -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError:
        print("The window needs Tkinter (part of the python.org installer, tcl/tk option). "
              "You can still use the command line: run.bat --help", file=sys.stderr)
        return 1
    root = tk.Tk()
    if issues:
        messagebox.showerror(APP_NAME, "\n\n".join(issues))
        root.destroy()
        return 1
    app = App(root, tk, ttk, filedialog, messagebox)
    root.mainloop()
    return 0


class App:
    def __init__(self, root, tk, ttk, filedialog, messagebox):
        self.root, self.tk, self.ttk = root, tk, ttk
        self.fd, self.mb = filedialog, messagebox
        root.title(f"{APP_NAME} {VERSION}")
        root.minsize(900, 720)
        try:
            root.geometry("1000x980")
        except tk.TclError:
            pass
        self.saved = load_settings()
        s = self.saved
        self.q: queue.Queue = queue.Queue()
        self.ctl: Optional[JobController] = None
        self.worker: Optional[threading.Thread] = None
        self.info = None
        self.hoops: list[Hoop] = [Hoop.from_dict(h) for h in s.get("hoops", [])] if s.get("hoops_input") else []
        self.hoops_for = s.get("hoops_input", "")
        self.clips: list[Clip] = []
        self.data_path: Optional[Path] = None
        self.title_edited = False
        self.name_locked = False
        V = tk.StringVar
        B = tk.BooleanVar
        self.v_input = V()
        self.v_outdir = V(value=str(saved_output_dir(s)))
        self.v_outname = V()
        self.v_start_s = tk.DoubleVar(value=0.0)
        self.v_end_s = tk.DoubleVar(value=0.0)
        self.v_start_lab = V(value="—")
        self.v_end_lab = V(value="—")
        self._range_dur = 0.0
        self._range_sync = False
        self.v_srcinfo = V(value="Pick an .insv, 360 MP4, or game video.")
        self.v_quality = V(value=QUALITY_LABELS.get(s.get("quality", "balanced"), QUALITY_LABELS["balanced"]))
        self.v_sens = tk.DoubleVar(value=float(s.get("sensitivity", 0.5)))
        self.v_pre = V(value=str(s.get("pre", 3.5)))
        self.v_post = V(value=str(s.get("post", 1.0)))
        self.v_attempts = B(value=bool(s.get("attempts", False)))
        self.v_warmup = B(value=bool(s.get("skip_warmup", True)))
        self.v_maxlen = V(value=str(s.get("max_length", "")))
        self.v_score_log = V(value=str(s.get("score_log", "")))
        self.v_ai_review = B(value=bool(s.get("ai_review", False)))
        self.v_lens = V(value=s.get("lens_model", "x4"))
        self.v_hoops = V()
        self.v_f169 = B(value=bool(s.get("f169", True)))
        self.v_f916 = B(value=bool(s.get("f916", False)))
        self.v_res = V(value=s.get("resolution", "auto"))
        self.v_crf = V(value=str(s.get("crf", 18)))
        self.v_preset = V(value=s.get("preset", "medium") if s.get("preset") in X264_PRESETS else "medium")
        self.v_intro = B(value=bool(s.get("intro", True)))
        self.v_title_on = B(value=bool(s.get("title_on", True)))
        self.v_title = V(value="")
        self.v_sub = V(value=s.get("title_sub", HL_SUB))
        self.v_title_dur = V(value=str(s.get("title_duration", 4)))
        self.v_logo = B(value=bool(s.get("logo", True)))
        self.v_music = V(value=s.get("music_label", ""))
        self.music_custom = s.get("music_custom", "")
        self.v_nomusic = B(value=bool(s.get("nomusic_copy", True)))
        self.v_chunks = B(value=bool(s.get("preview_chunks", False)))
        self.preview_dir: Optional[Path] = None
        self.v_lt_on = B(value=bool(s.get("lt_on", True)))
        self.v_lt1 = V(value=s.get("lt_line1", LT_TITLE))
        self.v_lt2 = V(value=s.get("lt_line2", LT_SUB))
        self.v_end_on = B(value=bool(s.get("end_on", True)))
        self.v_end_mode = V(value=s.get("end_mode", "At the end"))
        self.v_end_start = V(value=s.get("end_start", ""))
        self.v_end_dur = V(value=str(s.get("end_duration", 7)))
        self.v_end1 = V(value=s.get("end_title", END_TITLE))
        self.v_end2 = V(value=s.get("end_sub", END_SUB))
        self.v_qrcap = V(value=s.get("qr_caption", QR_CAPTION))
        self.v_qrurl = V(value=s.get("qr_url", DEFAULT_QR_URL))
        pm = s.get("perf_mode", DEFAULT_PERF_MODE)
        self.v_perf = V(value=PERF_LABELS.get(pm, "Balanced"))
        self.v_threads = V(value=str(s.get("perf_threads") or resolve_threads(pm)))
        self.v_autorender = B(value=bool(s.get("auto_render", False)))
        self._perf_sync = False
        self.v_status = V(value="Ready.")
        self.v_pct = V(value="0%")
        self.v_timing = V(value="Elapsed --   ETA --")
        self.v_prog = tk.DoubleVar(value=0)
        self.lt_rows: list[dict] = []
        self._build()
        for w in s.get("lt_windows", [{"start": "0:12", "duration": "6"}]):
            self._add_lt_row(w.get("start", ""), w.get("duration", "6"))
        self._refresh_music_choices()
        self._update_hoop_label()
        self._sync_analyze_button()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self._poll)

    # ------------------------------------------------------------------ layout
    def _build(self):
        tk, ttk = self.tk, self.ttk
        footer = ttk.Frame(self.root, padding=(10, 4, 10, 8))
        footer.pack(side="bottom", fill="x")
        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)
        canvas = tk.Canvas(body, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        frm = ttk.Frame(canvas, padding=(10, 8, 16, 8))
        fid = canvas.create_window((0, 0), window=frm, anchor="nw")
        frm.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(fid, width=max(1, e.width)))

        def wheel(e):
            if getattr(e, "widget", None) is not None and str(e.widget).startswith(str(self.tree)):
                return
            canvas.yview_scroll(1 if (e.num == 5 or getattr(e, "delta", 0) < 0) else -1, "units")
        canvas.bind_all("<MouseWheel>", wheel)
        canvas.bind_all("<Button-4>", wheel)
        canvas.bind_all("<Button-5>", wheel)
        frm.columnconfigure(0, weight=1)
        P = dict(padx=6, pady=3)
        self.lockable: list = []

        def lock(w):
            self.lockable.append(w)
            return w

        # --- 1. Input and output
        f1 = ttk.LabelFrame(frm, text="1. Video", padding=8)
        f1.grid(row=0, column=0, sticky="ew", pady=4)
        f1.columnconfigure(1, weight=1)
        ttk.Label(f1, text="Input").grid(row=0, column=0, sticky="w", **P)
        lock(ttk.Entry(f1, textvariable=self.v_input)).grid(row=0, column=1, columnspan=3, sticky="ew", **P)
        lock(ttk.Button(f1, text="Browse...", command=self.browse_input)).grid(row=0, column=4, **P)
        ttk.Label(f1, textvariable=self.v_srcinfo, foreground="#555").grid(row=1, column=1, columnspan=4, sticky="w", padx=6)
        ttk.Label(f1, text="Start").grid(row=2, column=0, sticky="w", **P)
        self.start_scale = ttk.Scale(f1, from_=0, to=1, variable=self.v_start_s, command=lambda v: self.on_range_scale("start"))
        self.start_scale.grid(row=2, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
        self.start_scale.state(["disabled"])
        ttk.Label(f1, textvariable=self.v_start_lab, width=8).grid(row=2, column=4, sticky="w")
        ttk.Label(f1, text="End").grid(row=3, column=0, sticky="w", **P)
        self.end_scale = ttk.Scale(f1, from_=0, to=1, variable=self.v_end_s, command=lambda v: self.on_range_scale("end"))
        self.end_scale.grid(row=3, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
        self.end_scale.state(["disabled"])
        ttk.Label(f1, textvariable=self.v_end_lab, width=8).grid(row=3, column=4, sticky="w")
        ttk.Label(f1, text="Drag to use part of the video. Both ends at the limits means the whole file.",
                  foreground="#555").grid(row=4, column=1, columnspan=4, sticky="w", padx=6)
        ttk.Label(f1, text="Output folder").grid(row=5, column=0, sticky="w", **P)
        lock(ttk.Entry(f1, textvariable=self.v_outdir)).grid(row=5, column=1, sticky="ew", **P)
        lock(ttk.Button(f1, text="Browse...", command=self.browse_outdir)).grid(row=5, column=2, **P)
        ttk.Button(f1, text="Open folder", command=self.open_outdir).grid(row=5, column=3, **P)
        lock(ttk.Button(f1, text="Reset", command=self.reset_outdir)).grid(row=5, column=4, **P)
        ttk.Label(f1, text="File name").grid(row=6, column=0, sticky="w", **P)
        e = lock(ttk.Entry(f1, textvariable=self.v_outname))
        e.grid(row=6, column=1, sticky="ew", **P)
        e.bind("<Key>", lambda _e: setattr(self, "name_locked", True))
        ttk.Label(f1, text="CSV files are saved next to the video with the same name.", foreground="#555").grid(
            row=6, column=2, columnspan=3, sticky="w", padx=6)
        self.v_input.trace_add("write", lambda *_: self.on_input_changed())
        self.v_outdir.trace_add("write", lambda *_: self.refresh_name())

        # --- 2. Detection
        f2 = ttk.LabelFrame(frm, text="2. Find highlights", padding=8)
        f2.grid(row=1, column=0, sticky="ew", pady=4)
        f2.columnconfigure(1, weight=1)
        ttk.Label(f2, text="Hoops").grid(row=0, column=0, sticky="w", **P)
        ttk.Label(f2, textvariable=self.v_hoops).grid(row=0, column=1, sticky="w", **P)
        hb = ttk.Frame(f2)
        hb.grid(row=0, column=2, columnspan=2, sticky="e")
        lock(ttk.Button(hb, text="Pick hoops...", command=self.pick_hoops)).pack(side="left", padx=4)
        lock(ttk.Button(hb, text="Auto", command=self.clear_hoops)).pack(side="left", padx=4)
        ttk.Label(f2, text="Detection quality").grid(row=1, column=0, sticky="w", **P)
        lock(ttk.Combobox(f2, textvariable=self.v_quality, values=list(QUALITY_LABELS.values()),
                          state="readonly", width=48)).grid(row=1, column=1, sticky="w", **P)
        ttk.Label(f2, text="Lens model").grid(row=1, column=2, sticky="e", **P)
        lock(ttk.Combobox(f2, textvariable=self.v_lens, values=["x4", "generic"], state="readonly", width=9)).grid(
            row=1, column=3, sticky="w", **P)
        ttk.Label(f2, text="Sensitivity").grid(row=2, column=0, sticky="w", **P)
        sf = ttk.Frame(f2)
        sf.grid(row=2, column=1, columnspan=3, sticky="w")
        ttk.Label(sf, text="strict").pack(side="left", padx=4)
        lock(ttk.Scale(sf, from_=0.0, to=1.0, variable=self.v_sens, length=200)).pack(side="left")
        ttk.Label(sf, text="catch more").pack(side="left", padx=4)
        ttk.Label(f2, text="Clip padding").grid(row=3, column=0, sticky="w", **P)
        pf = ttk.Frame(f2)
        pf.grid(row=3, column=1, columnspan=3, sticky="w")
        lock(ttk.Entry(pf, textvariable=self.v_pre, width=5)).pack(side="left", padx=4)
        ttk.Label(pf, text="s before,").pack(side="left")
        lock(ttk.Entry(pf, textvariable=self.v_post, width=5)).pack(side="left", padx=4)
        ttk.Label(pf, text="s after each basket.   Max total").pack(side="left")
        lock(ttk.Entry(pf, textvariable=self.v_maxlen, width=6)).pack(side="left", padx=4)
        ttk.Label(pf, text="s (blank = all; picks alternate hoops)").pack(side="left")
        of = ttk.Frame(f2)
        of.grid(row=4, column=1, columnspan=3, sticky="w")
        lock(ttk.Checkbutton(of, text="Switch on missed/unclear shots at the rim too", variable=self.v_attempts)).pack(side="left", padx=4)
        lock(ttk.Checkbutton(of, text="Skip warm-ups (multi-ball / lone shooters before tip-off)",
                             variable=self.v_warmup)).pack(side="left", padx=12)
        ttk.Label(f2, text="Score log (optional)").grid(row=5, column=0, sticky="w", **P)
        sf2 = ttk.Frame(f2)
        sf2.grid(row=5, column=1, columnspan=3, sticky="ew")
        lock(ttk.Entry(sf2, textvariable=self.v_score_log)).pack(side="left", fill="x", expand=True, padx=4)
        lock(ttk.Button(sf2, text="Browse...", command=self.browse_score_log)).pack(side="left", padx=4)
        ttk.Label(f2, text="CSV of score / ball-in times. When set, those times seed baskets; each is still rim-checked. Leave blank for auto detect.",
                  foreground="#555").grid(row=6, column=1, columnspan=3, sticky="w", padx=6)
        of2 = ttk.Frame(f2)
        of2.grid(row=7, column=1, columnspan=3, sticky="w")
        lock(ttk.Checkbutton(of2, text="AI review of rim strips (needs API key; off by default, fully offline without it)",
                             variable=self.v_ai_review)).pack(side="left", padx=4)
        bf = ttk.Frame(f2)
        bf.grid(row=8, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.btn_analyze = lock(ttk.Button(bf, text="Find highlights", command=self.start_analyze))
        self.btn_analyze.pack(side="left", padx=6)
        lock(ttk.Button(bf, text="Load clip list...", command=self.load_csv)).pack(side="left", padx=6)
        ttk.Label(bf, text="No GPU? Detection runs on the CPU and can take longer than the video. 'Fast' is quickest.",
                  foreground="#555").pack(side="left", padx=8)
        af = ttk.Frame(f2)
        af.grid(row=9, column=0, columnspan=4, sticky="w", pady=(4, 0))
        lock(ttk.Checkbutton(af, text="Render the video automatically when highlights are found",
                             variable=self.v_autorender, command=self._sync_analyze_button)).pack(side="left", padx=6)
        ttk.Label(af, text="Uses the format, intro, logo, and music in section 4.",
                  foreground="#555").pack(side="left", padx=4)

        # --- 3. Clips
        f3 = ttk.LabelFrame(frm, text="3. Review clips (double-click to edit; changes are saved to the CSV)", padding=8)
        f3.grid(row=2, column=0, sticky="ew", pady=4)
        f3.columnconfigure(0, weight=1)
        cols = ("on", "id", "type", "hoop", "start", "end", "len", "score", "note")
        self.tree = ttk.Treeview(f3, columns=cols, show="headings", height=9, selectmode="extended")
        widths = dict(on=40, id=60, type=70, hoop=45, start=80, end=80, len=50, score=55, note=300)
        for c in cols:
            self.tree.heading(c, text=c.capitalize() if c != "on" else "On")
            self.tree.column(c, width=widths[c], anchor="w" if c == "note" else "center", stretch=(c == "note"))
        self.tree.grid(row=0, column=0, sticky="ew")
        tsb = ttk.Scrollbar(f3, orient="vertical", command=self.tree.yview)
        tsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tsb.set)
        self.tree.bind("<Double-1>", lambda e: self.edit_clip())
        cb = ttk.Frame(f3)
        cb.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        for text, cmd in (("On / off", self.toggle_clips), ("Edit...", self.edit_clip), ("Add clip...", self.add_clip),
                          ("Delete", self.delete_clips), ("Preview frame...", self.preview_clip),
                          ("Open CSV", self.open_csv), ("Open camera CSV", self.open_camera_csv),
                          ("Open screenshots", self.open_shots), ("Reload CSV", self.reload_csv)):
            lock(ttk.Button(cb, text=text, command=cmd)).pack(side="left", padx=3)
        self.v_clipsum = tk.StringVar(value="No clips yet.")
        ttk.Label(f3, textvariable=self.v_clipsum, foreground="#555").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.shot_canvas = tk.Canvas(f3, height=112, background="#1a1a1a", highlightthickness=0)
        self.shot_canvas.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self.shot_scroll = ttk.Scrollbar(f3, orient="horizontal", command=self.shot_canvas.xview)
        self.shot_scroll.grid(row=4, column=0, sticky="ew")
        self.shot_canvas.configure(xscrollcommand=self.shot_scroll.set)
        self.shot_canvas.bind("<Button-1>", self._shot_click)
        self._shot_photos: list = []
        self._shot_hits: list = []
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._refresh_shots())

        # --- 4. Output
        f4 = ttk.LabelFrame(frm, text="4. Output", padding=8)
        f4.grid(row=3, column=0, sticky="ew", pady=4)
        f4.columnconfigure(1, weight=1)
        r = 0
        ff = ttk.Frame(f4)
        ff.grid(row=r, column=0, columnspan=4, sticky="w")
        ttk.Label(ff, text="Formats").pack(side="left", padx=6)
        lock(ttk.Checkbutton(ff, text="16:9 video", variable=self.v_f169)).pack(side="left", padx=4)
        lock(ttk.Checkbutton(ff, text="9:16 Short", variable=self.v_f916)).pack(side="left", padx=4)
        ttk.Label(ff, text="   Resolution").pack(side="left")
        lock(ttk.Combobox(ff, textvariable=self.v_res, values=["auto", "720", "1080", "1440", "2160"],
                          state="readonly", width=6)).pack(side="left", padx=4)
        ttk.Label(ff, text="   Quality (CRF)").pack(side="left")
        lock(ttk.Spinbox(ff, from_=12, to=30, textvariable=self.v_crf, width=4)).pack(side="left", padx=4)
        ttk.Label(ff, text="Preset").pack(side="left")
        lock(ttk.Combobox(ff, textvariable=self.v_preset, values=X264_PRESETS, state="readonly", width=9)).pack(side="left", padx=4)
        r += 1
        tf = ttk.Frame(f4)
        tf.grid(row=r, column=0, columnspan=4, sticky="ew", pady=2)
        tf.columnconfigure(2, weight=1)
        lock(ttk.Checkbutton(tf, text="Intro animation", variable=self.v_intro)).grid(row=0, column=0, sticky="w", padx=6)
        lock(ttk.Checkbutton(tf, text="Corner logo", variable=self.v_logo)).grid(row=0, column=1, sticky="w", padx=6)
        lock(ttk.Checkbutton(tf, text="Title", variable=self.v_title_on)).grid(row=1, column=0, sticky="w", padx=6)
        te = lock(ttk.Entry(tf, textvariable=self.v_title))
        te.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=2)
        te.bind("<Key>", lambda _e: setattr(self, "title_edited", True))
        ttk.Label(tf, text="seconds").grid(row=1, column=3, sticky="e")
        lock(ttk.Entry(tf, textvariable=self.v_title_dur, width=5)).grid(row=1, column=4, padx=6)
        ttk.Label(tf, text="Second line").grid(row=2, column=0, sticky="w", padx=6)
        lock(ttk.Entry(tf, textvariable=self.v_sub)).grid(row=2, column=1, columnspan=2, sticky="ew", padx=6, pady=2)
        r += 1
        mf = ttk.Frame(f4)
        mf.grid(row=r, column=0, columnspan=4, sticky="w", pady=2)
        ttk.Label(mf, text="Music").pack(side="left", padx=6)
        self.cb_music = lock(ttk.Combobox(mf, textvariable=self.v_music, state="readonly", width=40))
        self.cb_music.pack(side="left", padx=4)
        self.cb_music.bind("<<ComboboxSelected>>", lambda e: self.on_music_selected())
        lock(ttk.Checkbutton(mf, text="Also make a no-music copy (for Marketing / TikTok)",
                             variable=self.v_nomusic)).pack(side="left", padx=10)
        r += 1
        lf = ttk.LabelFrame(f4, text="Promo lower third (FreeHoops.net / Google Play)", padding=6)
        lf.grid(row=r, column=0, columnspan=4, sticky="ew", pady=4)
        lf.columnconfigure(1, weight=1)
        lock(ttk.Checkbutton(lf, text="Show", variable=self.v_lt_on)).grid(row=0, column=0, sticky="w")
        ttk.Label(lf, text="Times are in the finished video (the intro counts).", foreground="#555").grid(row=0, column=1, sticky="w")
        ttk.Label(lf, text="Line 1").grid(row=1, column=0, sticky="w")
        lock(ttk.Entry(lf, textvariable=self.v_lt1)).grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        ttk.Label(lf, text="Line 2").grid(row=2, column=0, sticky="w")
        lock(ttk.Entry(lf, textvariable=self.v_lt2)).grid(row=2, column=1, sticky="ew", padx=4, pady=2)
        self.lt_frame = ttk.Frame(lf)
        self.lt_frame.grid(row=3, column=0, columnspan=2, sticky="w")
        lock(ttk.Button(lf, text="Add time", command=lambda: self._add_lt_row("", "6"))).grid(row=4, column=0, sticky="w", pady=2)
        r += 1
        ef = ttk.LabelFrame(f4, text="End card with QR code", padding=6)
        ef.grid(row=r, column=0, columnspan=4, sticky="ew", pady=4)
        ef.columnconfigure(1, weight=1)
        row0 = ttk.Frame(ef)
        row0.grid(row=0, column=0, columnspan=2, sticky="w")
        lock(ttk.Checkbutton(row0, text="Show", variable=self.v_end_on)).pack(side="left")
        lock(ttk.Combobox(row0, textvariable=self.v_end_mode, values=["At the end", "At a time"],
                          state="readonly", width=11)).pack(side="left", padx=6)
        ttk.Label(row0, text="start").pack(side="left")
        lock(ttk.Entry(row0, textvariable=self.v_end_start, width=8)).pack(side="left", padx=4)
        ttk.Label(row0, text="duration (s)").pack(side="left")
        lock(ttk.Entry(row0, textvariable=self.v_end_dur, width=5)).pack(side="left", padx=4)
        for i, (lab, var) in enumerate((("Title", self.v_end1), ("Subtitle", self.v_end2),
                                        ("QR caption", self.v_qrcap), ("QR link", self.v_qrurl)), start=1):
            ttk.Label(ef, text=lab).grid(row=i, column=0, sticky="w")
            lock(ttk.Entry(ef, textvariable=var)).grid(row=i, column=1, sticky="ew", padx=4, pady=1)
        r += 1
        lock(ttk.Checkbutton(f4, text="Save 1-minute preview parts, then build the final video from them",
                             variable=self.v_chunks)).grid(row=r, column=0, columnspan=4, sticky="w", padx=6, pady=(4, 0))
        r += 1
        ttk.Label(f4, text="Each part is about one minute of the reframed clips. You can play a part as soon as it is written. A play is kept whole.",
                  foreground="#555").grid(row=r, column=0, columnspan=4, sticky="w", padx=6)
        r += 1
        self.btn_render = lock(ttk.Button(f4, text="Render highlights", command=self.start_render))
        self.btn_render.grid(row=r, column=0, sticky="w", padx=6, pady=(6, 2))

        # --- footer: performance (stays enabled during a run), progress + log
        pfm = ttk.Frame(footer)
        pfm.pack(fill="x", pady=(0, 4))
        ttk.Label(pfm, text="Performance").pack(side="left")
        cbp = ttk.Combobox(pfm, textvariable=self.v_perf, values=[PERF_LABELS[m] for m in PERF_MODES],
                           state="readonly", width=12)
        cbp.pack(side="left", padx=(6, 4))
        cbp.bind("<<ComboboxSelected>>", lambda e: self.on_perf_changed())
        ttk.Label(pfm, text="threads").pack(side="left")
        threads = ttk.Spinbox(pfm, from_=1, to=max(1, os.cpu_count() or 1), textvariable=self.v_threads,
                              width=4, command=self.on_threads_edited)
        threads.pack(side="left", padx=4)
        threads.bind("<Return>", lambda e: self.on_threads_edited())
        threads.bind("<FocusOut>", lambda e: self.on_threads_edited())
        ttk.Label(pfm, text="You can change this while processing. Priority applies at once; fewer threads apply to detection at once and to encoding on the next clip.",
                  foreground="#555").pack(side="left", padx=6)
        pr = ttk.Frame(footer)
        pr.pack(fill="x")
        ttk.Progressbar(pr, variable=self.v_prog, maximum=1.0).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Label(pr, textvariable=self.v_pct, width=5).pack(side="left")
        self.btn_pause = ttk.Button(pr, text="Pause", command=self.pause, state="disabled")
        self.btn_pause.pack(side="left", padx=4)
        self.btn_cancel = ttk.Button(pr, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=4)
        st = ttk.Frame(footer)
        st.pack(fill="x", pady=2)
        ttk.Label(st, textvariable=self.v_status).pack(side="left")
        ttk.Label(st, textvariable=self.v_timing).pack(side="right")
        lb = ttk.Frame(footer)
        lb.pack(fill="x")
        self.log_text = tk.Text(lb, height=8, wrap="word", state="disabled", font=("Consolas", 9))
        ls = ttk.Scrollbar(lb, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=ls.set)
        self.log_text.pack(side="left", fill="x", expand=True)
        ls.pack(side="right", fill="y")
        lbb = ttk.Frame(footer)
        lbb.pack(fill="x")
        ttk.Button(lbb, text="Open logs folder", command=lambda: self._safe(open_folder, logs_dir())).pack(side="left", pady=2)
        self.btn_open_out = ttk.Button(lbb, text="Open output folder", command=self.open_outdir, state="disabled")
        self.btn_open_out.pack(side="left", padx=8, pady=2)
        self.btn_open_preview = ttk.Button(lbb, text="Open preview parts", command=self.open_preview_dir, state="disabled")
        self.btn_open_preview.pack(side="left", padx=4, pady=2)

    # ------------------------------------------------------------------ helpers
    def _safe(self, fn, *a):
        try:
            fn(*a)
        except Exception as exc:
            self.mb.showerror(APP_NAME, str(exc))

    def log(self, line: str):
        self.q.put(("log", line))

    def _append_log(self, line: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def current_outdir(self) -> Path:
        raw = self.v_outdir.get().strip().strip('"')
        return Path(raw).expanduser() if raw else default_output_dir()

    def output_path(self) -> Path:
        name = self.v_outname.get().strip() or "highlights.mp4"
        if not name.lower().endswith(".mp4"):
            name += ".mp4"
        return self.current_outdir() / name

    def refresh_name(self):
        inp = self.v_input.get().strip().strip('"')
        if not inp or self.name_locked:
            return
        folder = self.current_outdir()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self.v_outname.set(unique_name(folder, output_stem_for(Path(inp)), "_highlights"))

    def on_input_changed(self):
        self.name_locked = False
        self.refresh_name()
        p = Path(self.v_input.get().strip().strip('"'))
        if not p.is_file():
            self._set_range_duration(0)
            return
        if not self.title_edited:
            self.v_title.set(default_title(None, p))
        if self.hoops_for and self.hoops_for != str(p.parent):
            self.hoops = []
            self._update_hoop_label()
        self.v_srcinfo.set("Reading video...")
        self._set_range_duration(0)
        threading.Thread(target=self._probe_bg, args=(p,), daemon=True).start()

    def _set_range_duration(self, dur: float):
        """Slider max is this file's length. 0 disables the sliders until a video is read."""
        dur = max(0.0, float(dur or 0))
        self._range_dur = dur
        self._range_sync = True
        try:
            if dur <= 2:
                self.start_scale.state(["disabled"])
                self.end_scale.state(["disabled"])
                self.v_start_s.set(0)
                self.v_end_s.set(0)
                self.v_start_lab.set("—")
                self.v_end_lab.set("—")
                return
            self.start_scale.configure(from_=0, to=dur)
            self.end_scale.configure(from_=0, to=dur)
            self.start_scale.state(["!disabled"])
            self.end_scale.state(["!disabled"])
            self.v_start_s.set(0)
            self.v_end_s.set(dur)
            self.v_start_lab.set(format_time(0))
            self.v_end_lab.set(format_time(dur))
        finally:
            self._range_sync = False

    def on_range_scale(self, which: str):
        if self._range_sync or self._range_dur <= 2:
            return
        dur = self._range_dur
        start = min(dur, max(0.0, round(float(self.v_start_s.get()))))
        end = min(dur, max(0.0, round(float(self.v_end_s.get()))))
        if end < start + 2:
            if which == "start":
                end = min(dur, start + 2)
                if end - start < 2:
                    start = max(0.0, end - 2)
            else:
                start = max(0.0, end - 2)
                if end - start < 2:
                    end = min(dur, start + 2)
        self._range_sync = True
        try:
            if abs(float(self.v_start_s.get()) - start) > 0.01:
                self.v_start_s.set(start)
            if abs(float(self.v_end_s.get()) - end) > 0.01:
                self.v_end_s.set(end)
        finally:
            self._range_sync = False
        self.v_start_lab.set(format_time(start))
        self.v_end_lab.set(format_time(end))

    def range_start(self) -> Optional[float]:
        if self._range_dur <= 2:
            return None
        start = float(self.v_start_s.get())
        return start if start > 0.5 else None

    def range_end(self) -> Optional[float]:
        if self._range_dur <= 2:
            return None
        end = float(self.v_end_s.get())
        return end if end < self._range_dur - 0.5 else None

    def _probe_bg(self, p: Path):
        try:
            from fhh_source import detect_source
            info = detect_source(p, find_tool("ffmpeg"), find_tool("ffprobe"))
            self.q.put(("info", (p, info)))
        except Exception as exc:
            self.q.put(("info_err", str(exc)))

    def browse_input(self):
        path = self.fd.askopenfilename(title="Choose the game video", filetypes=[
            ("Videos", "*.insv *.mp4 *.mov *.mkv"), ("Insta360", "*.insv"), ("All files", "*.*")])
        if path:
            self.v_input.set(path)

    def browse_outdir(self):
        d = self.fd.askdirectory(initialdir=str(self.current_outdir()))
        if d:
            self.v_outdir.set(d)
            self._persist_outdir()

    def reset_outdir(self):
        self.v_outdir.set(str(default_output_dir()))
        self._persist_outdir()

    def open_outdir(self):
        self._safe(open_folder, self.current_outdir())

    def open_preview_dir(self):
        if self.preview_dir is not None:
            self._safe(open_folder, self.preview_dir)

    def _persist_outdir(self):
        data = load_settings()
        data["output_dir"] = str(self.current_outdir())
        save_settings(data)

    def _update_hoop_label(self):
        if self.hoops:
            self.v_hoops.set("   ".join(f"{h.name}: yaw {h.yaw:.1f}, pitch {h.pitch:.1f}, rim {h.width:.2f} deg ({h.source})"
                                       for h in self.hoops))
        else:
            self.v_hoops.set("Auto (rough guess). Picking both rims gives much better results.")

    def clear_hoops(self):
        self.hoops = []
        self._update_hoop_label()

    def _refresh_music_choices(self):
        tracks = music_tracks()
        self.music_map = {p.stem.replace("_", " "): str(p) for p in tracks}
        vals = list(self.music_map) + [NO_MUSIC, CUSTOM_MUSIC]
        if self.music_custom:
            vals.insert(len(self.music_map), Path(self.music_custom).name)
            self.music_map[Path(self.music_custom).name] = self.music_custom
        self.cb_music.configure(values=vals)
        if self.v_music.get() not in vals:
            self.v_music.set(vals[0] if self.music_map else NO_MUSIC)

    def on_music_selected(self):
        if self.v_music.get() == CUSTOM_MUSIC:
            path = self.fd.askopenfilename(title="Choose a royalty-free music file",
                                           filetypes=[("Audio", "*.mp3 *.m4a *.wav *.aac *.ogg *.flac")])
            if path:
                self.music_custom = path
                self._refresh_music_choices()
                self.v_music.set(Path(path).name)
            else:
                self._refresh_music_choices()

    def music_path(self) -> str:
        v = self.v_music.get()
        if v in (NO_MUSIC, CUSTOM_MUSIC, ""):
            return ""
        return self.music_map.get(v, "")

    def _add_lt_row(self, start: str, dur: str):
        ttk = self.ttk
        fr = ttk.Frame(self.lt_frame)
        fr.pack(anchor="w", pady=1)
        row = {"frame": fr, "start": self.tk.StringVar(value=start), "dur": self.tk.StringVar(value=dur)}
        ttk.Label(fr, text="start").pack(side="left")
        self.lockable.append(ttk.Entry(fr, textvariable=row["start"], width=8))
        self.lockable[-1].pack(side="left", padx=4)
        ttk.Label(fr, text="duration (s)").pack(side="left")
        self.lockable.append(ttk.Entry(fr, textvariable=row["dur"], width=5))
        self.lockable[-1].pack(side="left", padx=4)

        def remove():
            fr.destroy()
            self.lt_rows.remove(row)
        self.lockable.append(ttk.Button(fr, text="Remove", command=remove))
        self.lockable[-1].pack(side="left", padx=4)
        self.lt_rows.append(row)

    def perf_mode(self) -> str:
        label = self.v_perf.get()
        return next((m for m, l in PERF_LABELS.items() if l == label), DEFAULT_PERF_MODE)

    def _sync_analyze_button(self):
        if getattr(self, "btn_analyze", None) is None:
            return
        self.btn_analyze.configure(text="Find and render" if self.v_autorender.get() else "Find highlights")

    def _chosen_threads(self) -> tuple[str, int]:
        mode = self.perf_mode()
        if mode != "custom":
            return mode, resolve_threads(mode)
        try:
            n = int(float(self.v_threads.get()))
        except (TypeError, ValueError):
            n = 1
        return mode, resolve_threads("custom", n)

    def on_perf_changed(self):
        if self._perf_sync:
            return
        mode = self.perf_mode()
        if mode != "custom":
            self._perf_sync = True
            try:
                self.v_threads.set(str(resolve_threads(mode)))
            finally:
                self._perf_sync = False
        self._push_performance()

    def on_threads_edited(self):
        if self._perf_sync:
            return
        mode = self.perf_mode()
        try:
            n = int(float(self.v_threads.get()))
        except (TypeError, ValueError):
            return
        if mode != "custom" and n == resolve_threads(mode):
            return
        if mode != "custom":
            self._perf_sync = True
            try:
                self.v_perf.set(PERF_LABELS["custom"])
            finally:
                self._perf_sync = False
        self._push_performance()

    def _push_performance(self):
        mode, threads = self._chosen_threads()
        running = self.ctl is not None and self.worker is not None and self.worker.is_alive()
        if running:
            self.ctl.set_performance(mode, threads)
            self.log(f"Performance: {PERF_LABELS[mode]} ({threads} threads, {priority_name(mode)} priority).")
        try:
            self.save_ui_settings()
        except Exception:
            pass

    # ------------------------------------------------------------------ clips table
    def show_clips(self):
        self.tree.delete(*self.tree.get_children())
        for c in self.clips:
            self.tree.insert("", "end", iid=c.id, values=(
                "yes" if c.include else "no", c.id, c.type, c.hoop, format_ts(c.start), format_ts(c.end),
                f"{c.duration:.1f}", f"{c.score:.2f}", c.note))
        on = [c for c in self.clips if c.include]
        hoops = sorted({c.hoop for c in on})
        self.v_clipsum.set(f"{len(self.clips)} clips, {len(on)} on ({sum(c.duration for c in on):.0f} s"
                           + (f", hoops {', '.join(hoops)}" if hoops else "") + ")."
                           + (f"   File: {self.data_path}" if self.data_path else ""))
        self._refresh_shots()

    def save_clips(self):
        if self.data_path is None:
            return
        try:
            write_clips(self.clips, self.data_path)
        except OSError as exc:
            self.mb.showerror(APP_NAME, f"Could not save the CSV (is it open in Excel?):\n{exc}")
        self.show_clips()

    def selected(self) -> list[Clip]:
        ids = set(self.tree.selection())
        return [c for c in self.clips if c.id in ids]

    def toggle_clips(self):
        for c in self.selected():
            c.include = not c.include
        self.save_clips()

    def delete_clips(self):
        sel = self.selected()
        if sel and self.mb.askyesno(APP_NAME, f"Delete {len(sel)} clip(s) from the list?"):
            self.clips = [c for c in self.clips if c not in sel]
            self.save_clips()

    def add_clip(self):
        if self.data_path is None:
            self.mb.showinfo(APP_NAME, "Find highlights (or load a clip list) first.")
            return
        n = 1
        ids = {c.id for c in self.clips}
        while f"m{n:03d}" in ids:
            n += 1
        c = Clip(id=f"m{n:03d}", start=0.0, end=8.0, event_time=6.0, type="manual",
                 hoop=self.hoops[0].name if self.hoops else "A", include=True,
                 order=max([x.order for x in self.clips] or [0]) + 1)
        if self._clip_dialog(c, new=True):
            self.clips.append(c)
            self.save_clips()

    def edit_clip(self):
        sel = self.selected()
        if not sel:
            return
        if self._clip_dialog(sel[0]):
            self.save_clips()

    def _clip_dialog(self, c: Clip, new: bool = False) -> bool:
        tk, ttk = self.tk, self.ttk
        d = tk.Toplevel(self.root)
        d.title(f"{'Add' if new else 'Edit'} clip {c.id}")
        d.transient(self.root)
        d.grab_set()
        vs = {k: tk.StringVar(value=v) for k, v in dict(
            start=format_ts(c.start), end=format_ts(c.end), event=format_ts(c.event_time), type=c.type,
            hoop=c.hoop, note=c.note, order=str(c.order)).items()}
        inc = tk.BooleanVar(value=c.include)
        rows = [("Start", "start"), ("End", "end"), ("Basket at", "event"), ("Type", "type"),
                ("Hoop", "hoop"), ("Order", "order"), ("Note", "note")]
        for i, (lab, k) in enumerate(rows):
            ttk.Label(d, text=lab).grid(row=i, column=0, sticky="w", padx=8, pady=3)
            if k == "type":
                w = ttk.Combobox(d, textvariable=vs[k], values=["made", "attempt", "play", "manual"], width=12)
            else:
                w = ttk.Entry(d, textvariable=vs[k], width=40 if k == "note" else 14)
            w.grid(row=i, column=1, sticky="w", padx=8, pady=3)
        ttk.Checkbutton(d, text="Include in the video", variable=inc).grid(row=len(rows), column=1, sticky="w", padx=8)
        ttk.Label(d, text="Times are in the source video (mm:ss.ss).", foreground="#555").grid(
            row=len(rows) + 1, column=0, columnspan=2, padx=8)
        result = {"ok": False}

        def ok():
            try:
                s, e = parse_time(vs["start"].get()), parse_time(vs["end"].get())
                if e <= s:
                    raise ValueError("End must be after Start.")
                ev = parse_time(vs["event"].get()) if vs["event"].get().strip() else (s + e) / 2
                c.start, c.end, c.event_time = s, e, ev
                c.type = vs["type"].get().strip() or "manual"
                c.hoop = vs["hoop"].get().strip() or "A"
                c.note = vs["note"].get().strip()
                c.order = int(float(vs["order"].get() or 0))
                c.include = inc.get()
            except ValueError as exc:
                self.mb.showerror(APP_NAME, str(exc), parent=d)
                return
            result["ok"] = True
            d.destroy()
        bf = ttk.Frame(d)
        bf.grid(row=len(rows) + 2, column=0, columnspan=2, pady=8)
        ttk.Button(bf, text="OK", command=ok).pack(side="left", padx=6)
        ttk.Button(bf, text="Cancel", command=d.destroy).pack(side="left", padx=6)
        self.root.wait_window(d)
        return result["ok"]

    def _refresh_shots(self):
        canvas = self.shot_canvas
        canvas.delete("all")
        self._shot_photos = []
        self._shot_hits = []
        if not self.clips:
            canvas.create_text(8, 56, anchor="w", fill="#888",
                               text="Screenshots of each basket show up here after highlights are found.")
            canvas.configure(scrollregion=(0, 0, 1, 112))
            return
        folder = shots_dir_for(self.data_path.with_suffix(".mp4")) if self.data_path else None
        try:
            from PIL import Image, ImageTk
        except ImportError:
            canvas.create_text(8, 56, anchor="w", fill="#888", text="Install Pillow to see the screenshots.")
            return
        selected = set(self.tree.selection()) if self.tree.winfo_exists() else set()
        x = 8
        thumb_w, thumb_h = 160, 90
        for clip in self.clips:
            path = folder / f"{clip.id}.jpg" if folder else None
            if path is not None and path.is_file():
                with Image.open(path) as src:
                    img = src.convert("RGB")
                img.thumbnail((thumb_w, thumb_h))
                photo = ImageTk.PhotoImage(img)
                self._shot_photos.append(photo)
                canvas.create_image(x, 6, image=photo, anchor="nw")
            else:
                canvas.create_rectangle(x, 6, x + thumb_w, 6 + thumb_h, fill="#333", outline="")
                canvas.create_text(x + thumb_w / 2, 6 + thumb_h / 2, text="no shot", fill="#888")
            if not clip.include:
                canvas.create_rectangle(x, 6, x + thumb_w, 6 + thumb_h, fill="#000", stipple="gray50", outline="")
                canvas.create_text(x + 4, 8, anchor="nw", fill="#ff8c28", text=f"{clip.id} off")
            color = "#ff8c28" if clip.id in selected else ("#6a6a6a" if clip.include else "#444")
            canvas.create_rectangle(x - 2, 4, x + thumb_w + 2, 8 + thumb_h, outline=color,
                                    width=3 if clip.id in selected else 1)
            self._shot_hits.append((x, x + thumb_w, clip.id))
            x += thumb_w + 10
        canvas.configure(scrollregion=(0, 0, max(x, 1), 112))

    def _shot_click(self, event):
        x = self.shot_canvas.canvasx(event.x)
        for left, right, cid in self._shot_hits:
            if left <= x <= right and self.tree.exists(cid):
                self.tree.selection_set(cid)
                self.tree.see(cid)
                self.tree.focus(cid)
                return

    def open_shots(self):
        if self.data_path is None:
            self.mb.showinfo(APP_NAME, "Find highlights (or load a clip list) first.")
            return
        self._safe(open_folder, shots_dir_for(self.data_path.with_suffix(".mp4")))

    def open_csv(self):
        if self.data_path:
            self._safe(open_folder_or_file, self.data_path)

    def open_camera_csv(self):
        if self.data_path:
            p = data_paths(self.data_path.with_suffix(".mp4"))["camera"]
            self._safe(open_folder_or_file, p)

    def reload_csv(self):
        if self.data_path is None:
            return
        try:
            self.clips = read_clips(self.data_path)
            self.show_clips()
            self.log(f"Reloaded {self.data_path.name}")
        except Exception as exc:
            self.mb.showerror(APP_NAME, str(exc))

    def load_csv(self):
        path = self.fd.askopenfilename(title="Open a clip list", filetypes=[("Clip list", "*_highlights*.csv"), ("CSV", "*.csv")])
        if not path:
            return
        p = Path(path)
        if p.stem.endswith("_camera"):
            p = p.with_name(p.stem[:-7] + ".csv")
        try:
            self.clips = read_clips(p)
        except Exception as exc:
            self.mb.showerror(APP_NAME, str(exc))
            return
        self.data_path = p
        proj_p = data_paths(p.with_suffix(".mp4"))["project"]
        if proj_p.is_file():
            try:
                proj = load_project(proj_p)
                self.hoops = proj.hoops
                self._update_hoop_label()
                if proj.input and Path(proj.input).is_file():
                    self.v_input.set(proj.input)
            except Exception as exc:
                self.log(f"Could not read {proj_p.name}: {exc}")
        self.v_outdir.set(str(p.parent))
        self.name_locked = True
        self.v_outname.set(p.with_suffix(".mp4").name)
        self.show_clips()

    # ------------------------------------------------------------------ hoops + preview
    def _need_info(self) -> bool:
        if self.info is None:
            self.mb.showinfo(APP_NAME, "Choose the input video first (and wait for it to be read).")
            return False
        return True

    def browse_score_log(self):
        path = self.fd.askopenfilename(
            title="Score log CSV",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.v_score_log.set(path)

    def pick_hoops(self):
        if not self._need_info():
            return
        HoopPicker(self, self.info)

    def preview_clip(self):
        sel = self.selected()
        if not sel or not self._need_info():
            return
        ClipPreview(self, self.info, sel[0])

    # ------------------------------------------------------------------ jobs
    def build_job(self) -> Job:
        inp = Path(self.v_input.get().strip().strip('"'))
        if not inp.is_file():
            raise ValueError("Choose an input video.")
        start = self.range_start()
        end = self.range_end()
        if start is not None and end is not None and end <= start:
            raise ValueError("End time must be after the start time.")
        quality = next((k for k, v in QUALITY_LABELS.items() if v == self.v_quality.get()), "balanced")
        maxlen = float(self.v_maxlen.get()) if self.v_maxlen.get().strip() else 0.0
        job = Job(input_path=inp, output_path=self.output_path(), start=start, end=end, quality=quality,
                  sensitivity=float(self.v_sens.get()), lens_model=self.v_lens.get(),
                  hoops=list(self.hoops),
                  clip=ClipSettings(pre=float(self.v_pre.get() or 3.5), post=float(self.v_post.get() or 1.0),
                                    include_attempts=self.v_attempts.get(), skip_warmup=self.v_warmup.get(),
                                    max_total=maxlen),
                  perf_mode=self.perf_mode(),
                  perf_threads=int(self.v_threads.get() or 1) if self.perf_mode() == "custom" else None,
                  score_log=Path(self.v_score_log.get().strip().strip('"')) if self.v_score_log.get().strip() else None,
                  ai_review=AIReviewSettings(enabled=bool(self.v_ai_review.get())))
        rs = RenderSettings()
        rs.formats = [f for f, v in (("16:9", self.v_f169), ("9:16", self.v_f916)) if v.get()]
        rs.resolution = self.v_res.get()
        rs.crf = int(self.v_crf.get() or 18)
        rs.preset = self.v_preset.get()
        rs.intro_on, rs.logo_on = self.v_intro.get(), self.v_logo.get()
        rs.title_on, rs.title_main, rs.title_sub = self.v_title_on.get(), self.v_title.get(), self.v_sub.get()
        rs.title_duration = float(self.v_title_dur.get() or 4)
        rs.music = self.music_path()
        rs.no_music_copy = self.v_nomusic.get()
        rs.lt_on = self.v_lt_on.get()
        rs.lt_windows = []
        for row in self.lt_rows:
            if row["start"].get().strip():
                rs.lt_windows.append(CardWindow(parse_time(row["start"].get()), parse_time(row["dur"].get() or "6")))
        rs.lt_line1, rs.lt_line2 = self.v_lt1.get(), self.v_lt2.get()
        rs.end_on = self.v_end_on.get()
        rs.end_at_end = self.v_end_mode.get() != "At a time"
        if not rs.end_at_end:
            if not self.v_end_start.get().strip():
                raise ValueError("Enter a start time for the end card, or choose 'At the end'.")
            rs.end_start = parse_time(self.v_end_start.get())
        rs.end_duration = float(self.v_end_dur.get() or 7)
        rs.end_title, rs.end_sub = self.v_end1.get(), self.v_end2.get()
        rs.qr_caption, rs.qr_url = self.v_qrcap.get(), self.v_qrurl.get()
        rs.preview_chunks = bool(self.v_chunks.get())
        job.render = rs
        return job

    def save_ui_settings(self):
        data = load_settings()
        data.update(dict(
            quality=next((k for k, v in QUALITY_LABELS.items() if v == self.v_quality.get()), "balanced"),
            sensitivity=float(self.v_sens.get()), pre=self.v_pre.get(), post=self.v_post.get(),
            attempts=self.v_attempts.get(), skip_warmup=self.v_warmup.get(), max_length=self.v_maxlen.get(),
            score_log=self.v_score_log.get(), ai_review=self.v_ai_review.get(),
            lens_model=self.v_lens.get(), f169=self.v_f169.get(), f916=self.v_f916.get(),
            resolution=self.v_res.get(), crf=self.v_crf.get(), preset=self.v_preset.get(),
            intro=self.v_intro.get(), title_on=self.v_title_on.get(), title_sub=self.v_sub.get(),
            title_duration=self.v_title_dur.get(), logo=self.v_logo.get(), music_label=self.v_music.get(),
            music_custom=self.music_custom, nomusic_copy=self.v_nomusic.get(), lt_on=self.v_lt_on.get(),
            lt_line1=self.v_lt1.get(), lt_line2=self.v_lt2.get(),
            lt_windows=[{"start": r["start"].get(), "duration": r["dur"].get()} for r in self.lt_rows],
            end_on=self.v_end_on.get(), end_mode=self.v_end_mode.get(), end_start=self.v_end_start.get(),
            end_duration=self.v_end_dur.get(), end_title=self.v_end1.get(), end_sub=self.v_end2.get(),
            qr_caption=self.v_qrcap.get(), qr_url=self.v_qrurl.get(), perf_mode=self.perf_mode(),
            perf_threads=self.v_threads.get(), auto_render=self.v_autorender.get(),
            preview_chunks=self.v_chunks.get(),
            output_dir=str(self.current_outdir()),
            hoops=[h.to_dict() for h in self.hoops],
            hoops_input=str(Path(self.v_input.get().strip().strip('"')).parent) if self.hoops else "",
        ))
        save_settings(data)

    def set_busy(self, on: bool):
        for w in self.lockable:
            try:
                w.configure(state="disabled" if on else ("readonly" if isinstance(w, self.ttk.Combobox) else "normal"))
            except Exception:
                pass
        self.btn_pause.configure(state="normal" if on else "disabled", text="Pause")
        self.btn_cancel.configure(state="normal" if on else "disabled")

    def _start(self, kind: str, fn):
        if self.worker is not None and self.worker.is_alive():
            return
        self.ctl = JobController()
        self.ctl.reset()
        self.v_prog.set(0)
        self.set_busy(True)
        self.save_ui_settings()

        def run():
            logger = RunLogger(also=lambda line: self.q.put(("log", line)))
            try:
                res = fn(self.ctl, lambda info: self.q.put(("progress", info)), logger)
                self.q.put(("done", (kind, res)))
            except RenderCancelled:
                self.q.put(("cancelled", kind))
            except Exception as exc:
                logger("ERROR: " + str(exc))
                logger(traceback.format_exc())
                self.q.put(("error", str(exc)))
            finally:
                logger.close()
        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def start_analyze(self):
        try:
            job = self.build_job()
        except Exception as exc:
            self.mb.showerror(APP_NAME, str(exc))
            return
        if not self.hoops and not self.mb.askyesno(
                APP_NAME, "No hoops picked. The tool will guess them, which is often wrong.\n\n"
                          "Continue with the automatic guess? (Choose No to pick them first.)"):
            self.pick_hoops()
            return
        csvp = data_paths(job.output_path)["clips"]
        if csvp.exists() and not self.mb.askyesno(APP_NAME, f"{csvp.name} already exists. Replace it?"):
            return
        auto = bool(self.v_autorender.get())
        if auto:
            if not job.render.formats:
                self.mb.showerror(APP_NAME, "Tick at least one format (16:9 or 9:16) in section 4.")
                return
            taken = [p for p in self._output_files(job) if p.exists()]
            if taken and not self.mb.askyesno(APP_NAME, "Replace " + ", ".join(p.name for p in taken) + "?"):
                return
            if taken:
                job.overwrite = True
            self.v_status.set("Finding highlights, then rendering...")
            self._start("auto", lambda ctl, prog, log: self._analyze_then_render(job, ctl, prog, log))
            return
        self.v_status.set("Finding highlights...")
        self._start("analyze", lambda ctl, prog, log: analyze(job, ctl, prog, log))

    def _output_files(self, job: Job) -> list[Path]:
        outs = []
        if "16:9" in job.render.formats:
            outs.append(job.output_path)
        if "9:16" in job.render.formats:
            outs.append(job.output_path.with_name(job.output_path.stem + "_short" + job.output_path.suffix))
        return outs

    def _analyze_then_render(self, job: Job, ctl: JobController, prog, log):
        res = analyze(job, ctl, prog, log)
        self.q.put(("clips", res))
        if not any(c.include for c in res["clips"]):
            log("No clips are switched on, so the video was not rendered.")
            return {"analyze": res, "outputs": []}
        log("Highlights found. Rendering them now.")
        self.q.put(("phase", "Rendering highlights..."))
        job.perf_mode = ctl.priority_mode or job.perf_mode
        job.perf_threads = ctl.threads if ctl.priority_mode == "custom" else None
        ctl.reset()
        outputs = render(job, res["paths"]["clips"], ctl, prog, log)
        return {"analyze": res, "outputs": outputs}

    def _show_found(self, res: dict):
        self.data_path = res["paths"]["clips"]
        self.clips = res["clips"]
        self.hoops = res["project"].hoops
        self._update_hoop_label()
        self.show_clips()

    def start_render(self):
        if self.data_path is None:
            self.mb.showinfo(APP_NAME, "Find highlights (or load a clip list) first.")
            return
        try:
            job = self.build_job()
        except Exception as exc:
            self.mb.showerror(APP_NAME, str(exc))
            return
        if not job.render.formats:
            self.mb.showerror(APP_NAME, "Tick at least one format (16:9 or 9:16).")
            return
        job.output_path = self.data_path.with_suffix(".mp4")
        taken = [o for o in self._output_files(job) if o.exists()]
        if taken:
            if not self.mb.askyesno(APP_NAME, "Replace " + ", ".join(o.name for o in taken) + "?"):
                return
            job.overwrite = True
        data = self.data_path
        self.v_status.set("Rendering...")
        self._start("render", lambda ctl, prog, log: render(job, data, ctl, prog, log))

    def pause(self):
        if self.ctl is None:
            return
        paused = self.ctl.toggle_pause()
        self.btn_pause.configure(text="Resume" if paused else "Pause")
        self.v_status.set("Paused." if paused else "Working...")

    def cancel(self, ask: bool = True) -> bool:
        if self.ctl is None or self.worker is None or not self.worker.is_alive():
            return True
        if ask and not self.mb.askyesno(APP_NAME, "Stop now? Unfinished output is deleted."):
            return False
        self.ctl.request_cancel()
        self.v_status.set("Cancelling...")
        return True

    def on_close(self):
        if self.worker is not None and self.worker.is_alive():
            if not self.cancel(ask=True):
                return
        try:
            self.save_ui_settings()
        except Exception:
            pass
        self.root.destroy()

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self._append_log(val)
                    if val.startswith("Preview part ") and ": " in val:
                        self.preview_dir = Path(val.split(": ", 1)[1].strip()).parent
                        self.btn_open_preview.configure(state="normal")
                elif kind == "progress":
                    info: ProgressInfo = val
                    self.v_prog.set(info.fraction)
                    self.v_pct.set(f"{int(info.fraction * 100)}%")
                    self.v_status.set(("Paused - " if info.paused else "") + info.message)
                    self.v_timing.set(f"Elapsed {format_time(info.elapsed)}   ETA {format_eta(info.eta)}")
                elif kind == "info":
                    p, info = val
                    if Path(self.v_input.get().strip().strip('"')) == p:
                        self.info = info
                        self.v_srcinfo.set(info.describe())
                        self._set_range_duration(info.duration)
                        if not self.title_edited:
                            self.v_title.set(default_title(info, p))
                elif kind == "info_err":
                    self.info = None
                    self.v_srcinfo.set("Could not read this video: " + val)
                    self._set_range_duration(0)
                elif kind == "phase":
                    self.btn_pause.configure(text="Pause")
                    self.v_status.set(val)
                elif kind == "clips":
                    self._show_found(val)
                elif kind == "done":
                    what, res = val
                    self.set_busy(False)
                    self.v_prog.set(1.0)
                    self.v_pct.set("100%")
                    if what == "analyze":
                        self._show_found(res)
                        self.v_status.set("Done. Review the clips, then Render.")
                        self.save_ui_settings()
                    elif what == "auto":
                        self._show_found(res["analyze"])
                        self.save_ui_settings()
                        if res["outputs"]:
                            self.v_status.set("Done: " + ", ".join(Path(p).name for p in res["outputs"]))
                            self.btn_open_out.configure(state="normal")
                            self.mb.showinfo(APP_NAME, "Finished:\n" + "\n".join(str(p) for p in res["outputs"]))
                        else:
                            self.v_status.set("Highlights found, but none are switched on. Nothing was rendered.")
                            self.mb.showinfo(APP_NAME, "Highlights were found, but every clip is switched off, so no video was rendered.")
                    else:
                        self.v_status.set("Done: " + ", ".join(Path(p).name for p in res))
                        self.btn_open_out.configure(state="normal")
                        self.mb.showinfo(APP_NAME, "Finished:\n" + "\n".join(str(p) for p in res))
                elif kind == "cancelled":
                    self.set_busy(False)
                    self.v_status.set("Cancelled.")
                elif kind == "error":
                    self.set_busy(False)
                    self.v_status.set("Stopped with an error (see log).")
                    self.mb.showerror(APP_NAME, val)
                elif kind == "hoops":
                    self.hoops = val
                    self._update_hoop_label()
                    self.save_ui_settings()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)


def open_folder_or_file(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        import subprocess
        import shutil as _sh
        opener = _sh.which("xdg-open") or _sh.which("open")
        if opener:
            subprocess.Popen([opener, str(path)])


# ---------------------------------------------------------------------------
# Hoop picker: click a rim on the overview, then its left and right edge
# ---------------------------------------------------------------------------

class HoopPicker:
    def __init__(self, app: App, info):
        import numpy as np
        self.app, self.info, self.np = app, info, np
        tk, ttk = app.tk, app.ttk
        self.top = tk.Toplevel(app.root)
        self.top.title("Pick hoops")
        self.top.transient(app.root)
        self.hoops: list[Hoop] = []
        self.stage = "overview"
        self.edges: list[tuple[float, float]] = []
        self.marked = False
        self._busy = False
        self._gen = 0
        self._ctl = None
        self.aq: queue.Queue = queue.Queue()
        bar = ttk.Frame(self.top, padding=6)
        bar.pack(fill="x")
        ttk.Label(bar, text="Frame at").pack(side="left")
        start = app.range_start() or 0.0
        self.v_t = tk.StringVar(value=format_ts(min(info.duration - 1, start + 30)))
        ttk.Entry(bar, textvariable=self.v_t, width=10).pack(side="left", padx=4)
        self.btn_load = ttk.Button(bar, text="Load frame", command=self.load)
        self.btn_load.pack(side="left", padx=4)
        self.btn_auto = ttk.Button(bar, text="Auto pick", command=self.auto_pick)
        self.btn_auto.pack(side="left", padx=4)
        ttk.Button(bar, text="Start over", command=self.reset).pack(side="left", padx=4)
        self.btn_done = ttk.Button(bar, text="Done", command=self.done)
        self.btn_done.pack(side="left", padx=4)
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.v_help = tk.StringVar()
        ttk.Label(self.top, textvariable=self.v_help, padding=(8, 0)).pack(fill="x")
        self.canvas = tk.Canvas(self.top, width=1200, height=480, background="#111", highlightthickness=0)
        self.canvas.pack(padx=6, pady=6)
        self.canvas.bind("<Button-1>", self.click)
        self.photo = None
        self.pano = None
        self.load()

    def help(self):
        n = len(self.hoops)
        if self.stage == "overview" and self.marked and n:
            word = "rim" if n == 1 else "rims"
            self.v_help.set(f"Auto pick marked {n} {word}. Click Done to use them, or Start over to mark them yourself.")
        elif self.stage == "overview":
            self.v_help.set(f"Click on hoop {'AB'[min(n, 1)]}'s rim (hoop {n + 1} of 2). "
                            "If a rim is hidden, load another frame. Click Done after one or two hoops.")
        else:
            self.v_help.set("Zoomed in: click the LEFT edge of the rim, then the RIGHT edge "
                            f"({len(self.edges)}/2).")

    def load(self):
        from fhh_source import (ANALYSIS_BAND_PICK, Panorama, SceneRenderer, decode_plan, grab_frame_raw,
                                base_filter)
        self.top.configure(cursor="watch")
        self.top.update()
        try:
            t = parse_time(self.v_t.get())
            info = self.info
            ffmpeg = find_tool("ffmpeg")
            plan = decode_plan(info, max_lens=1920, max_width=1920 if info.kind == "flat" else 3840)
            graph = base_filter(info)
            native = (2 * info.lens_size, info.lens_size) if info.is_fisheye else (info.width, info.height)
            if (plan.width, plan.height) != native:
                graph += f",scale={plan.width}:{plan.height}:flags=area"
            files = info.files if info.kind == "split" else info.files[:1]
            frame = grab_frame_raw(ffmpeg, files, t, graph, plan.width, plan.height)
            if info.kind == "flat":
                self.img = frame
                self.full_h, self.row0 = frame.shape[0], 0
            else:
                sr = SceneRenderer(info, plan, self.app.v_lens.get(),
                                   eq_width=2 * plan.lens_size if info.is_fisheye else None,
                                   lat_range=ANALYSIS_BAND_PICK if info.is_fisheye else None)
                pano = sr.panorama(frame, t)
                if not info.is_fisheye:
                    # crop the band from a full equirect
                    H = pano.shape[0]
                    r0 = int((90 - ANALYSIS_BAND_PICK[1]) / 180 * H)
                    r1 = int((90 - ANALYSIS_BAND_PICK[0]) / 180 * H)
                    pano = pano[r0:r1]
                    self.full_h, self.row0 = H, r0
                else:
                    self.full_h, self.row0 = sr.eq_h, sr.row0
                self.img = pano
            self.stage = "overview"
            self.show_overview()
        except Exception as exc:
            self.app.mb.showerror(APP_NAME, f"Could not load a frame: {exc}", parent=self.top)
        finally:
            self.top.configure(cursor="")

    def _show(self, bgr, w, h):
        from PIL import Image, ImageTk
        img = Image.fromarray(bgr[:, :, ::-1].copy()).resize((w, h))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.configure(width=w, height=h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def scene_of(self, x, y):
        from fhh_source import flat_to_scene
        if self.info.kind == "flat":
            h, w = self.img.shape[:2]
            return flat_to_scene(x, y, w, h)
        W = self.img.shape[1]
        yaw = (x + 0.5) / W * 360.0 - 180.0
        pitch = 90.0 - (y + self.row0 + 0.5) / self.full_h * 180.0
        return float(yaw), float(pitch)

    def px_of(self, yaw, pitch):
        from fhh_source import flat_to_px
        if self.info.kind == "flat":
            h, w = self.img.shape[:2]
            x, y = flat_to_px(yaw, pitch, w, h)
            return float(x), float(y)
        W = self.img.shape[1]
        return (yaw + 180.0) / 360.0 * W - 0.5, (90.0 - pitch) / 180.0 * self.full_h - 0.5 - self.row0

    def show_overview(self):
        h, w = self.img.shape[:2]
        self.scale = min(1200 / w, 560 / h)
        self._show(self.img, int(w * self.scale), int(h * self.scale))
        import math
        for hp in self.hoops:
            x, y = self.px_of(hp.yaw, hp.pitch)
            if self.info.kind == "flat":
                span = hp.width / 2
            else:
                span = (hp.width / 2) / max(0.2, math.cos(math.radians(hp.pitch)))
            x1, _ = self.px_of(hp.yaw - span, hp.pitch)
            x2, _ = self.px_of(hp.yaw + span, hp.pitch)
            x, y, x1, x2 = x * self.scale, y * self.scale, x1 * self.scale, x2 * self.scale
            self.canvas.create_line(x1, y, x2, y, fill="#ff8c28", width=3)
            self.canvas.create_oval(x - 9, y - 9, x + 9, y + 9, outline="#ff8c28", width=3)
            self.canvas.create_text(x, y - 18, text=hp.name, fill="#ff8c28", font=("Arial", 14, "bold"))
        self.help()

    def show_zoom(self, yaw, pitch):
        from fhh_source import render_flat_view, render_view
        self.zoom_center = (yaw, pitch)
        self.zoom_fov = 14.0
        size = 560
        if self.info.kind == "flat":
            view = render_flat_view(self.img, yaw, pitch, self.zoom_fov, size, size)
        else:
            view = render_view(self.img, yaw, pitch, self.zoom_fov, size, size, self.full_h, self.row0)
        self.zoom_size = size
        self._show(view, size, size)
        self.stage = "zoom"
        self.edges = []
        self.help()

    def click(self, e):
        if self.stage == "overview":
            if len(self.hoops) >= 2:
                return
            yaw, pitch = self.scene_of(e.x / self.scale, e.y / self.scale)
            self.show_zoom(float(yaw), float(pitch))
            return
        import math
        size, fov = self.zoom_size, self.zoom_fov
        cy, cp = self.zoom_center
        if self.info.kind == "flat":
            dpp = fov / size
            yaw = cy + (e.x - size / 2) * dpp
            pitch = cp - (e.y - size / 2) * dpp
        else:
            f = (size / 2) / math.tan(math.radians(fov) / 2)
            yaw = cy + math.degrees(math.atan((e.x - size / 2) / f)) / max(0.2, math.cos(math.radians(cp)))
            pitch = cp - math.degrees(math.atan((e.y - size / 2) / f))
        self.edges.append((yaw, pitch))
        self.canvas.create_line(e.x, e.y - 12, e.x, e.y + 12, fill="#ff8c28", width=3)
        if len(self.edges) == 2:
            (y1, p1), (y2, p2) = self.edges
            mid_y = (y1 + y2) / 2
            mid_p = (p1 + p2) / 2
            width = abs(y2 - y1) * (math.cos(math.radians(mid_p)) if self.info.kind != "flat" else 1.0)
            if width < 0.2:
                width = estimate_rim_width(mid_p)
            self.hoops.append(Hoop("AB"[len(self.hoops)], mid_y, mid_p, width, "manual"))
            self.stage = "overview"
            self.top.after(300, self.show_overview)
        else:
            self.help()

    def _set_busy(self, on: bool):
        self._busy = on
        state = ["disabled"] if on else ["!disabled"]
        for btn in (self.btn_auto, self.btn_load, self.btn_done):
            btn.state(state)

    def auto_pick(self):
        if self._busy:
            return
        self._gen += 1
        gen = self._gen
        self._ctl = JobController()
        mode, threads = self.app._chosen_threads()
        self._ctl.set_performance(mode, threads)
        self._set_busy(True)
        self.v_help.set("Looking for the rims. This can take a few minutes. Start over cancels it.")
        threading.Thread(target=self._auto_worker, args=(gen,), daemon=True).start()
        self.top.after(200, self._poll_auto)

    def _auto_worker(self, gen: int):
        try:
            from fhh_core import Progress
            from fhh_detect import QUALITY, AnalysisSettings, Detector, ensure_weights, scout_hoops
            from fhh_source import ANALYSIS_BAND_PICK, SceneRenderer, decode_plan
            info = self.info
            quality = next((k for k, v in QUALITY_LABELS.items() if v == self.app.v_quality.get()), "balanced")
            q = QUALITY[quality]
            plan = decode_plan(info, max_lens=1920, max_width=1920 if info.kind == "flat" else 3840)
            scene = SceneRenderer(info, plan, self.app.v_lens.get(),
                                  eq_width=2 * plan.lens_size if info.is_fisheye else None,
                                  lat_range=ANALYSIS_BAND_PICK if info.is_fisheye else None)
            start = self.app.range_start() or 0.0
            end = self.app.range_end() or info.duration
            st = AnalysisSettings(quality=quality, start=start, end=end, lens_model=self.app.v_lens.get(),
                                 threads=self._ctl.thread_count(None))
            log = lambda line: self.aq.put(("status", gen, line))
            weights = ensure_weights(q["model"], log)
            det = Detector(weights, st.threads, log, "auto")
            prog = Progress(lambda p: self.aq.put(("status", gen, p.message)), self._ctl)
            prog.plan([("Finding hoops", 1.0)])
            prog.begin("Finding hoops")
            hoops = scout_hoops(find_tool("ffmpeg"), info, plan, scene, det, q, start, end, st,
                               self._ctl, prog, log)
            self.aq.put(("done", gen, hoops))
        except RenderCancelled:
            self.aq.put(("cancelled", gen))
        except Exception as exc:
            self.aq.put(("error", gen, str(exc)))

    def _poll_auto(self):
        if not self.top.winfo_exists():
            return
        try:
            while True:
                kind, gen, val = self.aq.get_nowait()
                if gen != self._gen:
                    continue
                if kind == "status":
                    self.v_help.set(val)
                elif kind == "done":
                    self._set_busy(False)
                    self.hoops = list(val)
                    self.marked = bool(self.hoops)
                    self.stage = "overview"
                    if self.hoops and getattr(self, "img", None) is not None:
                        self.show_overview()
                    elif self.hoops:
                        n = len(self.hoops)
                        word = "rim" if n == 1 else "rims"
                        self.v_help.set(f"Auto pick marked {n} {word}. Click Done to use them, or Start over to mark them yourself.")
                    else:
                        self.v_help.set("Auto pick could not see the rims. Click each rim, or load another frame and try again.")
                elif kind == "error":
                    self._set_busy(False)
                    self.app.mb.showerror(APP_NAME, val, parent=self.top)
                    self.help()
                elif kind == "cancelled":
                    self._set_busy(False)
        except queue.Empty:
            pass
        if self._busy and self.top.winfo_exists():
            self.top.after(200, self._poll_auto)

    def reset(self):
        self._gen += 1
        if self._ctl is not None:
            self._ctl.request_cancel()
        self.marked = False
        self.hoops = []
        self.stage = "overview"
        self._set_busy(False)
        if getattr(self, "img", None) is not None:
            self.show_overview()

    def close(self):
        self._gen += 1
        if self._ctl is not None:
            self._ctl.request_cancel()
        self.top.destroy()

    def done(self):
        if self._busy:
            return
        if self.hoops:
            self.app.q.put(("hoops", list(self.hoops)))
        self.top.destroy()


# ---------------------------------------------------------------------------
# Preview: one frame of a clip through the virtual camera
# ---------------------------------------------------------------------------

class ClipPreview:
    def __init__(self, app: App, info, clip: Clip):
        tk, ttk = app.tk, app.ttk
        self.app, self.info, self.clip = app, info, clip
        self.top = tk.Toplevel(app.root)
        self.top.title(f"Preview {clip.id}")
        bar = ttk.Frame(self.top, padding=6)
        bar.pack(fill="x")
        self.v_t = tk.DoubleVar(value=clip.event_time)
        ttk.Scale(bar, from_=clip.start, to=clip.end, variable=self.v_t, length=500).pack(side="left", padx=4)
        self.v_lab = tk.StringVar(value=format_ts(clip.event_time))
        self.v_t.trace_add("write", lambda *_: self.v_lab.set(format_ts(self.v_t.get())))
        ttk.Label(bar, textvariable=self.v_lab, width=10).pack(side="left")
        self.v_fmt = tk.StringVar(value="16:9")
        ttk.Combobox(bar, textvariable=self.v_fmt, values=["16:9", "9:16"], state="readonly", width=6).pack(side="left", padx=4)
        ttk.Button(bar, text="Show", command=self.show).pack(side="left", padx=4)
        self.canvas = tk.Canvas(self.top, width=960, height=540, background="#111", highlightthickness=0)
        self.canvas.pack(padx=6, pady=6)
        self.photo = None
        self.show()

    def show(self):
        from PIL import Image, ImageTk
        from fhh_camera import animated_zoom, camera_at, zooms_are_flat
        from fhh_render import _fallback_keys, base_hfov
        from copy import copy
        from fhh_source import ClipRenderer, base_filter, decode_plan, grab_frame_raw
        self.top.configure(cursor="watch")
        self.top.update()
        try:
            info = self.info
            if self.app.data_path:
                proj_p = data_paths(self.app.data_path.with_suffix(".mp4"))["project"]
                if proj_p.is_file():
                    try:
                        if not load_project(proj_p).settings.get("stabilized"):
                            info = copy(self.info)
                            info.stab = None
                    except Exception:
                        pass
            t = float(self.v_t.get())
            keys = []
            if self.app.data_path:
                cam = data_paths(self.app.data_path.with_suffix(".mp4"))["camera"]
                keys = [k for k in read_keyframes(cam) if k.clip_id == self.clip.id]
            if not keys:
                keys = _fallback_keys(self.clip, self.app.hoops)
            plan = decode_plan(info)
            files = info.files if info.kind == "split" else info.files[:1]
            frame = grab_frame_raw(find_tool("ffmpeg"), files, t, base_filter(info), plan.width, plan.height)
            yaw, pitch, zoom = camera_at(keys, t)
            if zooms_are_flat(keys):
                zoom = animated_zoom(t, self.clip, info.kind == "flat")
            fmt = self.v_fmt.get()
            w, h = (960, 540) if fmt == "16:9" else (304, 540)
            view = ClipRenderer(info, plan, self.app.v_lens.get()).view(
                frame, yaw, pitch, base_hfov(fmt, info) / zoom, w, h, t)
            self.photo = ImageTk.PhotoImage(Image.fromarray(view[:, :, ::-1].copy()))
            self.canvas.configure(width=w, height=h)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
            self.canvas.create_text(8, 8, anchor="nw", fill="#ff8c28",
                                    text=f"{format_ts(t)}  yaw {yaw:.1f}  pitch {pitch:.1f}  zoom {zoom:.2f}")
        except Exception as exc:
            self.app.mb.showerror(APP_NAME, f"Preview failed: {exc}", parent=self.top)
        finally:
            self.top.configure(cursor="")
