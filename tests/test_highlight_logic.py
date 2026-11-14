#!/usr/bin/env python3
"""Unit tests for rim verification, rejection rules, score log, clips, eval.

Run: python -m pytest tests/test_highlight_logic.py -q
  or: python tests/test_highlight_logic.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tools"))

from fhh_detect import (  # noqa: E402
    Detections, Event, Hoop, apply_rejection_rules, camera_setup_end,
    detect_events, estimate_rim_width, events_from_score_times, verify_rim_make,
)
from fhh_project import ClipSettings, events_to_clips  # noqa: E402
from fhh_scoreboard import load_score_log, detect_scoreboard_changes, parse_region  # noqa: E402
import eval_highlights as ev  # noqa: E402


def _cands(rows):
    """rows: (t, hoop, dx, dy, conf, src, size)"""
    return np.array(rows, dtype=np.float64).reshape(-1, 7)


def _net(rows):
    """rows: (t, hoop, net_motion, zone_motion)"""
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


class TestRimWidth(unittest.TestCase):
    def test_estimate_not_too_wide(self):
        w = estimate_rim_width(8.0)
        self.assertLessEqual(w, 3.2)
        self.assertGreaterEqual(w, 0.7)
        # Farther (lower pitch) -> narrower or equal apparent width.
        self.assertLessEqual(estimate_rim_width(6.0), estimate_rim_width(12.0) + 0.01)


class TestVerifyRimMake(unittest.TestCase):
    def test_accepts_downward_through_with_net(self):
        # Synthetic make: above -> through -> net, with net motion spike.
        c = _cands([
            (10.00, 0, 0.1, -2.0, 0.6, 0, 0.4),
            (10.15, 0, 0.05, -1.0, 0.55, 0, 0.4),
            (10.30, 0, 0.05, 0.2, 0.7, 0, 0.35),
            (10.40, 0, 0.05, 0.8, 0.5, 0, 0.3),
        ])
        n = _net([
            (10.25, 0, 0.02, 0.1),
            (10.35, 0, 0.25, 0.2),
            (10.45, 0, 0.18, 0.15),
        ])
        got = verify_rim_make(c, n, 10.0, sensitivity=0.5)
        self.assertIsNotNone(got)
        t, score, note = got
        self.assertGreaterEqual(score, 0.5)
        self.assertAlmostEqual(t, 10.30, delta=0.2)
        self.assertIn("rim", note)

    def test_rejects_roll_out(self):
        # Ball goes through then reappears above the rim (miss that rolled out).
        c = _cands([
            (5.00, 0, 0.1, -1.8, 0.6, 0, 0.4),
            (5.20, 0, 0.05, 0.3, 0.65, 0, 0.35),
            (5.45, 0, 0.1, -1.2, 0.55, 0, 0.4),  # roll-out
        ])
        n = _net([(5.1, 0, 0.02, 0.1), (5.2, 0, 0.03, 0.1)])
        got = verify_rim_make(c, n, 5.0, sensitivity=0.5)
        self.assertIsNone(got)

    def test_rejects_beside_only(self):
        c = _cands([
            (7.0, 0, 0.2, -1.5, 0.5, 0, 0.4),
            (7.2, 0, 1.2, 0.2, 0.5, 0, 0.4),  # beside, not through
        ])
        n = _net([(7.1, 0, 0.02, 0.05)])
        self.assertIsNone(verify_rim_make(c, n, 7.0, sensitivity=0.3))


class TestDetectEventsSynthetic(unittest.TestCase):
    def _det(self, cands, net):
        return Detections(
            times=np.array([9.0, 10.0, 11.0]),
            persons=np.zeros((0, 5)),
            balls=np.zeros((0, 4)),
            rim_cands=cands,
            rim_net=net,
            rim_times=np.array([9.0, 10.0, 11.0]),
        )

    def test_detects_one_make(self):
        c = _cands([
            (10.00, 0, 0.1, -2.0, 0.6, 0, 0.4),
            (10.20, 0, 0.05, 0.3, 0.7, 0, 0.35),
            (10.35, 0, 0.05, 0.9, 0.5, 0, 0.3),
        ])
        n = _net([(10.25, 0, 0.22, 0.2), (10.0, 0, 0.02, 0.05)])
        events = detect_events(self._det(c, n), [Hoop("A", 0, 8, 1.5)], 0.5)
        makes = [e for e in events if e.kind == "made"]
        self.assertGreaterEqual(len(makes), 1)
        self.assertAlmostEqual(makes[0].time, 10.2, delta=0.3)


class TestRejection(unittest.TestCase):
    def test_camera_setup_and_lone_shooter(self):
        # Early: one huge person (setup). Later: one person near hoop (lone).
        persons = np.array([
            [0.5, 0.0, 0.0, 40.0, 0.9],   # huge, near t=0
            [12.0, -5.0, 0.0, 12.0, 0.8],  # lone shooter
            [40.0, -10, 0, 10, 0.8],
            [40.0, 0, 0, 10, 0.8],
            [40.0, 10, 0, 10, 0.8],
            [40.0, 20, 0, 10, 0.8],
            [40.0, 30, 0, 10, 0.8],
            [40.0, -20, 0, 10, 0.8],
        ], dtype=np.float64)
        d = Detections(
            times=np.linspace(0, 60, 31),
            persons=persons,
            balls=np.zeros((0, 4)),
            rim_cands=np.zeros((0, 7)),
            rim_net=np.zeros((0, 4)),
            rim_times=np.linspace(0, 60, 31),
        )
        setup = camera_setup_end(d, 0.0)
        self.assertIsNotNone(setup)
        self.assertGreaterEqual(setup, 2.0)
        hoops = [Hoop("A", 0, 8, 1.5), Hoop("B", 120, 8, 1.5)]
        events = [
            Event(0.8, 0, "made", 0.9, "setup basket"),
            Event(12.0, 0, "made", 0.9, "lone"),
            Event(40.0, 0, "made", 0.9, "game"),
        ]
        out = apply_rejection_rules(events, d, hoops, warmup=15.0, setup_until=setup, skip_warmup=True)
        self.assertTrue(out[0].rejected)
        self.assertIn("setup", out[0].reject_reason)
        self.assertTrue(out[1].rejected)
        self.assertFalse(out[2].rejected)


class TestScoreLog(unittest.TestCase):
    def test_load_clip_list_v2(self):
        path = ROOT / "tests" / "fixtures" / "clip_list_v2.csv"
        events = load_score_log(path)
        self.assertEqual(len(events), 31)
        self.assertAlmostEqual(events[0].time, 40.0, delta=0.05)
        self.assertEqual(events[0].hoop, 0)  # W -> 0
        self.assertAlmostEqual(events[2].time, 2 * 60 + 36.4, delta=0.05)

    def test_events_from_score_times_refines(self):
        c = _cands([
            (40.0 - 0.3, 0, 0.1, -1.8, 0.55, 0, 0.4),
            (40.0, 0, 0.05, 0.4, 0.7, 0, 0.35),
            (40.15, 0, 0.05, 0.9, 0.45, 0, 0.3),
        ])
        n = _net([(40.0, 0, 0.2, 0.15), (39.5, 0, 0.02, 0.05)])
        d = Detections(
            times=np.array([39.0, 40.0, 41.0]),
            persons=np.zeros((0, 5)),
            balls=np.zeros((0, 4)),
            rim_cands=c,
            rim_net=n,
            rim_times=np.array([39.0, 40.0, 41.0]),
        )
        events = events_from_score_times([40.0], d, [Hoop("A", 0, 8, 1.5)], 0.5, [0])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "made")
        self.assertFalse(events[0].rejected)
        self.assertIn("scoreboard", events[0].note)

    def test_scoreboard_change_detection(self):
        a = np.zeros((40, 80), np.uint8)
        b = np.zeros((40, 80), np.uint8)
        b[10:30, 20:60] = 200
        frames = [a, a, b, b, b, a]
        times = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        evs = detect_scoreboard_changes(frames, times, min_gap=0.5, change_thr=0.02, persist=2)
        self.assertGreaterEqual(len(evs), 1)
        self.assertAlmostEqual(evs[0].time, 1.0, delta=0.6)

    def test_parse_region(self):
        self.assertEqual(parse_region("0.1,0.2,0.3,0.15"), (0.1, 0.2, 0.3, 0.15))
        self.assertEqual(parse_region("10,20,30,15"), (0.1, 0.2, 0.3, 0.15))


class TestClipTiming(unittest.TestCase):
    def test_short_clips_end_one_second_after(self):
        hoops = [Hoop("A", 0, 8, 1.5)]
        events = [Event(100.0, 0, "made", 0.9, "x"), Event(108.0, 0, "made", 0.9, "y")]
        clips = events_to_clips(events, hoops, ClipSettings(pre=3.5, post=1.0, max_clip=6.5),
                                0.0, 200.0, None)
        self.assertEqual(len(clips), 2)
        for c in clips:
            self.assertAlmostEqual(c.end, c.event_time + 1.0, delta=0.05)
            self.assertLessEqual(c.duration, 6.6)
            self.assertGreaterEqual(c.duration, 3.0)

    def test_does_not_merge_distant_makes(self):
        hoops = [Hoop("A", 0, 8, 1.5)]
        # Old bug: pre=6/post=3 would merge these into one long clip.
        events = [Event(20.0, 0, "made", 0.9, "a"), Event(28.0, 0, "made", 0.9, "b"),
                  Event(36.0, 0, "made", 0.9, "c")]
        clips = events_to_clips(events, hoops, ClipSettings(pre=3.5, post=1.0, max_clip=6.5, merge_gap=2.2),
                                0.0, 100.0, None)
        self.assertEqual(len(clips), 3)
        self.assertTrue(all(c.duration <= 6.6 for c in clips))

    def test_putback_still_merges(self):
        hoops = [Hoop("A", 0, 8, 1.5)]
        events = [Event(50.0, 0, "attempt", 0.5, "miss"), Event(51.5, 0, "made", 0.9, "putback")]
        clips = events_to_clips(events, hoops, ClipSettings(pre=3.5, post=1.0, merge_gap=2.2),
                                0.0, 100.0, None)
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].type, "made")

    def test_rejected_switched_off(self):
        hoops = [Hoop("A", 0, 8, 1.5)]
        events = [Event(10.0, 0, "made", 0.9, "x", rejected=True, reject_reason="warm-up")]
        clips = events_to_clips(events, hoops, ClipSettings(), 0.0, 100.0, None)
        self.assertEqual(len(clips), 1)
        self.assertFalse(clips[0].include)
        self.assertIn("warm-up", clips[0].note)


class TestEvalScript(unittest.TestCase):
    def test_metrics_on_fixture_self(self):
        path = ROOT / "tests" / "fixtures" / "clip_list_v2.csv"
        truth = ev.load_truth_times(path)
        self.assertEqual(len(truth), 31)
        tp, miss, fp = ev.match_times(truth, truth, 0.75)
        m = ev.metrics(tp, len(truth), len(truth))
        self.assertEqual(m["tp"], 31)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(miss, [])
        self.assertEqual(fp, [])

    def test_tolerance_mismatch(self):
        truth = [10.0, 20.0, 30.0]
        pred = [10.2, 21.5, 30.1]
        tp, miss, fp = ev.match_times(pred, truth, 0.5)
        self.assertEqual(tp, 2)
        self.assertEqual(miss, [20.0])
        self.assertEqual(fp, [21.5])

    def test_eval_main_runs(self):
        path = ROOT / "tests" / "fixtures" / "clip_list_v2.csv"
        # Build a tiny pred CSV from the fixture times with noise.
        truth = ev.load_truth_times(path)
        with tempfile.TemporaryDirectory() as td:
            pred_path = Path(td) / "pred.csv"
            with open(pred_path, "w", encoding="utf-8") as fp:
                fp.write("id,include,type,event_time\n")
                for i, t in enumerate(truth[:5]):
                    fp.write(f"c{i:03d},yes,made,{t + 0.2:.2f}\n")
                fp.write("c999,yes,made,999.0\n")  # false positive
            rc = ev.main(["--pred", str(pred_path), "--truth", str(path), "--tolerance", "0.5"])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
