"""Optional AI review of rim close-up strips (off by default, fully offline without it).

When enabled, each candidate make can be confirmed by a vision model API using a
user-provided key (OPENAI_API_KEY or --ai-api-key). The tool never requires a
network call for normal operation.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from fhh_detect import Event


@dataclass
class AIReviewSettings:
    enabled: bool = False
    api_key: str = ""
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1/chat/completions"
    timeout: float = 45.0


def resolve_api_key(explicit: str = "") -> str:
    return (explicit or os.environ.get("OPENAI_API_KEY") or os.environ.get("FHH_AI_API_KEY") or "").strip()


def encode_strip_jpeg(frames: list[np.ndarray], quality: int = 80) -> bytes:
    """Horizontal contact strip of rim close-ups -> JPEG bytes."""
    import cv2
    if not frames:
        raise ValueError("No frames for AI review strip")
    h = min(f.shape[0] for f in frames)
    resized = [cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in frames]
    strip = np.concatenate(resized, axis=1)
    ok, buf = cv2.imencode(".jpg", strip, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Could not encode rim strip for AI review")
    return buf.tobytes()


def ask_vision_make(jpeg: bytes, st: AIReviewSettings, log: Callable[[str], None]) -> Optional[bool]:
    """Return True/False if the model answers, or None on failure (leave event unchanged)."""
    key = resolve_api_key(st.api_key)
    if not key:
        log("AI review skipped: no API key (set OPENAI_API_KEY or --ai-api-key).")
        return None
    b64 = base64.b64encode(jpeg).decode("ascii")
    body = {
        "model": st.model,
        "temperature": 0,
        "max_tokens": 40,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "These are consecutive close-up frames of a basketball rim/net "
                    "(left to right in time, ~0.1 s apart). "
                    "Reply with exactly YES if the ball clearly passes downward through "
                    "the rim into the net (a make). Reply with exactly NO if it is a miss, "
                    "roll-out, rebound, warm-up, occlusion, or unclear."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    }
    req = urllib.request.Request(
        st.base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=st.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        log(f"AI review request failed: {exc}")
        return None
    try:
        text = data["choices"][0]["message"]["content"].strip().upper()
    except (KeyError, IndexError, AttributeError, TypeError):
        log("AI review: unexpected API response")
        return None
    if text.startswith("YES"):
        return True
    if text.startswith("NO"):
        return False
    log(f"AI review: unclear reply {text!r}")
    return None


def review_events_with_ai(
    events: list[Event],
    strips: dict[int, bytes],
    st: AIReviewSettings,
    log: Callable[[str], None],
) -> list[Event]:
    """Apply AI YES/NO to events by index. Missing strips or failures leave the event alone."""
    if not st.enabled:
        return events
    out: list[Event] = []
    for i, e in enumerate(events):
        if e.rejected or e.kind != "made":
            out.append(e)
            continue
        jpeg = strips.get(i)
        if not jpeg:
            out.append(e)
            continue
        ans = ask_vision_make(jpeg, st, log)
        if ans is True:
            note = (e.note + "; AI: make").strip("; ")
            out.append(Event(e.time, e.hoop, e.kind, min(1.0, e.score + 0.05), note,
                             rejected=False, reject_reason=""))
        elif ans is False:
            out.append(Event(e.time, e.hoop, e.kind, e.score, e.note,
                             rejected=True, reject_reason="AI: not a make"))
        else:
            out.append(e)
    return out
