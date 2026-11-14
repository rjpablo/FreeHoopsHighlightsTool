# Highlight selection strategy (v1.2)

This document describes how the FreeHoops Highlights Tool chooses made-basket
clips. It mirrors the hand-curated workflow that worked well on real games
(Oct 12 flat reframed export without a scoreboard; Nov 12 with a scoreboard
overlay). The tool is offline-first: everything runs without a network except
an opt-in AI review step.

## Goals

- Prefer **confirmed makes** (ball through the rim into the net), not every
  motion near a hoop.
- Keep clips short (~3–5 s), ending about **1 s after** the ball goes in.
- Mix both hoops in game order.
- Leave a **user review** step: every candidate stays in the CSV; rejected
  ones are switched off with a note.

## Path A — no scoreboard (default)

Used when no score log / scoreboard overlay is given (Oct 12 style).

1. **Scan.** Decode the chosen time range at the analysis frame rate. Full-
   court YOLO (person + sports ball) runs every Nth frame for the virtual
   camera. Every analysis frame builds a close-up **rim view** of each hoop.
2. **Candidates (broad).** Seed times from:
   - ball / motion blobs above or through the rim
   - net-zone motion spikes
3. **Rim verification (strict).** For each seed, require a **downward** path:
   candidate above the rim → through the rim cylinder → into the net zone
   (and/or a net-motion spike). Roll-outs (ball reappears above the rim after
   “through”) are rejected. Many first-pass “makes” are actually misses.
4. **Rejection rules** (candidates stay in the list, `include=no`):
   - **Camera setup** — early stretch with almost nobody on court, or a very
     large person filling the frame (tripod / walk-up / someone blocking the
     lens). The first ~2.5 s of a range that starts at 0 is always treated as
     setup.
   - **Warm-ups** — multi-ball shootaround windows, and/or lone / sparse
     shooters before a game-like crowd (5+ people) appears.
   - **Occlusion** — a large person box covering the rim view.
   - **Other side of the court** — quiet basket while the game is at the
     other end (existing rule).
5. **Score & rank.** Makes are scored from above/through confidence + net
   bonus − beside-rim penalty. Events are kept in game time order for review.
6. **Clips.** Default padding: **3.5 s before**, **1.0 s after** (never less
   than 1 s after). Max clip length ~6.5 s. Same-hoop events merge only when
   within ~2.2 s (putbacks), so separate baskets never become a 20 s clip.
7. **Framing.** Hold the shooter through release (longer on long shots), then
   pan to the rim and hold through the finish.

Optional extras (attempts / standout dribbles) can be switched on in review;
attempts are off by default.

## Path B — scoreboard / score log

Used when a score log CSV and/or scoreboard overlay is provided (Nov 12 style).

1. **Seed times** from:
   - `--score-log` / GUI “Score log”: CSV with `ball_in`, `event_time`,
     `score_time`, or `time` (also accepts the hand-labelled `clip_list_v2`
     columns). Optional `hoop` column (`A`/`B`/`W`/…).
   - `--scoreboard-video` + `--scoreboard-region x,y,w,h`: change detection
     on the score digits crop (OCR optional if Tesseract is installed later;
     change detection alone is enough to propose times).
2. **Refine** each seed with the same rim verification as path A. Prefer the
   verified rim time when it is within ~1.5 s of the score change.
3. **Drop** weak / obstructed seeds with no rim activity (marked rejected).
4. Fall back to path A if no score input is given or sampling fails.

## Optional AI review (off by default)

`--ai-review` with `OPENAI_API_KEY` / `--ai-api-key` can send rim close-up
strips to a vision model for a YES/NO make confirmation. The tool works fully
offline without it. Failures leave the offline decision unchanged.

## Tunable parameters

| Knob | Where | Default | Notes |
|------|--------|---------|-------|
| Sensitivity | GUI / `--sensitivity` | 0.5 | 0 = strict, 1 = catch more |
| Detection quality | GUI / `--quality` | balanced | fps, YOLO size |
| Clip pre / post | GUI / `--pre` `--post` | 3.5 / 1.0 | post floored at 1.0 |
| Max clip | `--max-clip` | 6.5 | stops multi-basket merges |
| Max total | GUI / `--max-length` | off | balanced across hoops |
| Skip warm-ups | GUI checkbox | on | multi-ball + lone shooter |
| Score log | GUI / `--score-log` | none | path B |
| Music LUFS | `--music-lufs` | -14 | final music mix loudness |
| AI review | `--ai-review` | off | needs API key |

Rim width is estimated conservatively from pitch (or measured when picking
hoops). An overestimated width widens the through-rim zone and lets roll-outs
look like makes — prefer **Pick hoops** and mark left/right rim edges.

## Evaluating against a hand-labelled CSV

Use the evaluation script:

```text
run.bat eval --pred output\GAME_highlights.csv --truth tests\fixtures\clip_list_v2.csv --tolerance 0.75
```

Or:

```text
python tools/eval_highlights.py --pred PRED.csv --truth TRUTH.csv --tolerance 0.75
```

- **Truth** times come from `ball_in` / `event_time` / `time`.
- **Pred** times come from included rows’ `event_time` (or all rows with
  `--all-rows`).
- A predicted make matches a truth make if `|t_pred - t_truth| ≤ tolerance`
  (default 0.75 s).
- Reports precision, recall, F1, unmatched truth/pred times.

The Nov 12 Game 1 hand-verified list (`tests/fixtures/clip_list_v2.csv`, 31
makes) is the reference format. Without rj’s footage this repo only runs the
evaluator on synthetic / unit-test data; real accuracy must be measured on a
game recording.

## Known limits

- Detection needs a clear rim view; heavy blur, bodies over the rim, and
  extreme fisheye foreshortening still miss or false-fire.
- Scoreboard change detection proposes times only; OCR of the actual score is
  not required and not bundled.
- Camera framing still depends on YOLO person/ball tracks; review the camera
  CSV when a shooter is clipped.
