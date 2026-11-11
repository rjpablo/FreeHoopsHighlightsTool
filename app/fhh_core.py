"""Shared helpers for the FreeHoops Highlights Tool.

Paths, manifests, settings, time parsing, performance modes (threads and
process priority), the job controller (progress, pause, cancel), logging,
FFmpeg discovery, and media probing. Large parts are reused from the
FreeHoops Promo Tool v1.3 so the two programs behave the same way.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

VERSION = "1.0"
APP_NAME = "FreeHoops Highlights Tool"
SETTINGS_NAME = "FreeHoopsHighlightsTool.settings.json"
APP_REQUIRES_RUNTIME = "1.0"
APP_REQUIRES_ASSETS = "1.0"
FULL_ZIP = "FreeHoopsHighlightsTool_v1.0.zip"
PERF_MODES = ("full", "balanced", "background", "custom")
PERF_LABELS = {
    "full": "Full",
    "balanced": "Balanced",
    "background": "Background",
    "custom": "Custom",
}
X264_PRESETS = (
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
)
DEFAULT_PERF_MODE = "balanced"
DEFAULT_X264_PRESET = "medium"

# Windows priority classes (creationflags / SetPriorityClass)
_NORMAL_PRIORITY_CLASS = 0x00000020
_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
_IDLE_PRIORITY_CLASS = 0x00000040
_PROCESS_SET_INFORMATION = 0x0200


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def install_root() -> Path:
    """Folder that holds Run.bat, app/, runtime/, assets/, settings, logs, and output/."""
    here = app_dir()
    parent = here.parent
    if here.name in {"app", "src"}:
        if any((parent / name).exists() for name in ("Run.bat", "runtime", "assets", "app")):
            return parent
    for candidate in (here, parent):
        if (candidate / "runtime").is_dir() or (candidate / "app").is_dir():
            return candidate
    return here


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for bit in str(value or "0").split("."):
        num = ""
        for ch in bit:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts) or (0,)


def assets_dir(override: Optional[Path] = None) -> Path:
    if override:
        return Path(override)
    root = install_root()
    here = app_dir()
    for candidate in (root / "assets", here / "assets", here.parent / "assets"):
        if candidate.is_dir():
            return candidate
    return root / "assets"


def fonts_dir(assets: Optional[Path] = None) -> Path:
    root = install_root()
    candidates = [
        root / "runtime" / "fonts",
        (assets or assets_dir()) / "fonts",
    ]
    for c in candidates:
        if (c / "DejaVuSans-Bold.ttf").is_file() or (c / "DejaVuSans.ttf").is_file():
            return c
    return candidates[0]


def asset_path(name: str, override: Optional[Path] = None) -> Path:
    path = assets_dir(override) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required asset: {path}")
    return path


def find_tool(name: str) -> str:
    """Find ffmpeg or ffprobe. Prefer runtime/, then a copy next to the program, then PATH."""
    root = install_root()
    here = app_dir()
    names = [name]
    if not name.endswith(".exe"):
        names.append(name + ".exe")
    search = []
    for n in names:
        search.extend(
            [
                root / "runtime" / n,
                root / n,
                here / n,
                here / "bin" / n,
                here / "runtime" / n,
            ]
        )
    for candidate in search:
        if not candidate.is_file():
            continue
        # A Windows package stores ffmpeg.exe in runtime/; on Linux skip
        # those binaries and fall through to PATH.
        if sys.platform != "win32" and candidate.suffix.lower() == ".exe":
            continue
        if sys.platform != "win32" and not os.access(candidate, os.X_OK):
            continue
        return str(candidate)
    found = shutil.which(name) or shutil.which(name + ".exe")
    if found:
        return found
    raise FileNotFoundError(
        f"Could not find {name}. It should be in the runtime folder next to "
        "this program (from the full package), or installed and on PATH."
    )


def parse_time(value: str) -> float:
    """Parse ss, mm:ss, or hh:mm:ss, with optional fractional seconds."""
    if value is None:
        raise ValueError("Missing time value")
    text = str(value).strip()
    if not text:
        raise ValueError("Missing time value")
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"Invalid time: {value!r} (use mm:ss or hh:mm:ss)")
    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid time: {value!r}") from exc
    if any(n < 0 for n in nums):
        raise ValueError(f"Invalid time: {value!r}")
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


def format_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def parse_ratio(value: Optional[str], default: float = 1.0) -> float:
    if not value or value in {"N/A", "0:1", "0/0"}:
        return default
    if ":" in value:
        a, b = value.split(":", 1)
    elif "/" in value:
        a, b = value.split("/", 1)
    else:
        try:
            return float(value)
        except ValueError:
            return default
    try:
        aa, bb = float(a), float(b)
        return aa / bb if bb else default
    except ValueError:
        return default


def even(n: int) -> int:
    n = int(n)
    return n if n % 2 == 0 else n + 1


def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def settings_path() -> Path:
    return install_root() / SETTINGS_NAME


def load_settings() -> dict:
    path = settings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    try:
        settings_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def resolve_threads(mode: str, custom: int | None = None) -> int:
    """Thread count for a performance mode."""
    n = cpu_count()
    mode = (mode or DEFAULT_PERF_MODE).strip().lower()
    if mode == "full":
        return max(1, n)
    if mode == "balanced":
        return max(1, n // 2)
    if mode == "background":
        # About a quarter of the cores, at least 1, and 2 when there are 4+ cores.
        quarter = max(1, (n + 3) // 4)
        if n >= 4:
            return max(2, quarter)
        return max(1, quarter)
    if mode == "custom":
        try:
            t = int(custom) if custom is not None else 1
        except (TypeError, ValueError):
            t = 1
        return max(1, min(t, n))
    return max(1, n // 2)


def priority_name(mode: str) -> str:
    mode = (mode or DEFAULT_PERF_MODE).strip().lower()
    return {
        "full": "normal",
        "balanced": "below-normal",
        "background": "idle/low",
        "custom": "below-normal",
    }.get(mode, "below-normal")


def windows_priority_class(mode: str) -> int:
    mode = (mode or DEFAULT_PERF_MODE).strip().lower()
    if mode == "full":
        return _NORMAL_PRIORITY_CLASS
    if mode == "background":
        return _IDLE_PRIORITY_CLASS
    return _BELOW_NORMAL_PRIORITY_CLASS


def unix_nice(mode: str) -> int:
    mode = (mode or DEFAULT_PERF_MODE).strip().lower()
    if mode == "full":
        return 0
    if mode == "background":
        return 15
    return 10


def lookahead_threads(threads: int) -> int:
    return max(1, threads // 2)


def performance_summary(mode: str, threads: int, preset: str) -> str:
    label = PERF_LABELS.get((mode or "").lower(), mode or DEFAULT_PERF_MODE)
    return (
        f"{label} ({threads} thread{'s' if threads != 1 else ''}, "
        f"{priority_name(mode)} priority, x264 preset {preset})"
    )


def apply_process_priority(pid: int, mode: str) -> None:
    """Set the ffmpeg process (and its group) to the requested priority."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(_PROCESS_SET_INFORMATION, False, pid)
        if not handle:
            raise OSError(f"Could not open process {pid} to set priority.")
        try:
            if not kernel32.SetPriorityClass(handle, windows_priority_class(mode)):
                raise OSError("SetPriorityClass failed.")
        finally:
            kernel32.CloseHandle(handle)
        # Also try children (ffmpeg may spawn helpers)
        try:
            for child in _windows_child_pids(pid):
                h = kernel32.OpenProcess(_PROCESS_SET_INFORMATION, False, child)
                if not h:
                    continue
                try:
                    kernel32.SetPriorityClass(h, windows_priority_class(mode))
                finally:
                    kernel32.CloseHandle(h)
        except Exception:
            pass
        return
    nice = unix_nice(mode)
    errors: list[str] = []
    try:
        os.setpriority(os.PRIO_PROCESS, pid, nice)
    except (OSError, ProcessLookupError, AttributeError) as exc:
        errors.append(str(exc))
    try:
        os.setpriority(os.PRIO_PGRP, os.getpgid(pid), nice)
    except (OSError, ProcessLookupError, AttributeError) as exc:
        errors.append(str(exc))
    if errors and os.getpriority(os.PRIO_PROCESS, pid) != nice:
        raise OSError("Could not set niceness: " + "; ".join(errors))


def default_output_dir() -> Path:
    path = install_root() / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def saved_output_dir(saved: dict | None = None) -> Path:
    data = saved if saved is not None else load_settings()
    raw = str(data.get("output_dir") or "").strip().strip('"')
    if raw:
        path = Path(raw).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            pass
    return default_output_dir()



def open_folder(path: Path) -> None:
    """Create the folder if needed and open it in the system file manager."""
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return
    opener = shutil.which("xdg-open")
    if opener:
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    raise OSError(f"Could not open folder: {path}")


# ---------------------------------------------------------------------------
# Render control: progress, pause/resume, cancel, logs
# ---------------------------------------------------------------------------

PREP_WEIGHT = 0.05
ENCODE_WEIGHT = 0.93
FINAL_WEIGHT = 0.02


class RenderCancelled(Exception):
    """The user cancelled the render."""


@dataclass
class ProgressInfo:
    fraction: float
    stage: str
    message: str
    out_time: float = 0.0
    expected: float = 0.0
    speed: float = 0.0
    fps: float = 0.0
    elapsed: float = 0.0
    eta: float | None = None
    paused: bool = False


ProgressCb = Optional[Callable[[ProgressInfo], None]]


def logs_dir() -> Path:
    path = install_root() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    """Timestamped logger that writes to a file and optionally to the UI or CLI."""

    def __init__(self, also: Callable[[str], None] | None = None, log_path: Path | None = None):
        self.path = log_path or (logs_dir() / f"FreeHoopsHighlightsTool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a", encoding="utf-8")
        self.also = also
        self._closed = False
        self("Log started: " + str(self.path))

    def __call__(self, msg: str) -> None:
        line = f"{now_stamp()}  {msg}"
        try:
            self._fp.write(line + "\n")
            self._fp.flush()
        except OSError:
            pass
        if self.also:
            self.also(line)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._fp.close()
            except OSError:
                pass


def summarize_ffmpeg(cmd: list[str]) -> str:
    parts: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "-filter_complex" and i + 1 < len(cmd):
            parts.append("-filter_complex")
            parts.append(f"<{len(cmd[i + 1])} chars>")
            i += 2
            continue
        parts.append(cmd[i])
        i += 1
    return " ".join(shlex.quote(p) for p in parts)


def terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    pid = proc.pid
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=8,
                creationflags=flags,
            )
        except Exception:
            try:
                proc.kill()
            except OSError:
                pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                proc.terminate()
            except OSError:
                pass
        try:
            proc.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                try:
                    proc.kill()
                except OSError:
                    pass
    try:
        proc.wait(timeout=3)
    except Exception:
        pass


def _windows_child_pids(pid: int) -> list[int]:
    """Best-effort child list so pause can freeze helper processes too."""
    import ctypes
    from ctypes import wintypes

    TH32CS_SNAPPROCESS = 0x00000002
    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.CHAR * 260),
        ]

    kernel32 = ctypes.windll.kernel32
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == wintypes.HANDLE(-1).value or snap == 0xFFFFFFFFFFFFFFFF:
        return []
    pids: list[int] = []
    try:
        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        if not kernel32.Process32First(snap, ctypes.byref(entry)):
            return []
        while True:
            if entry.th32ParentProcessID == pid:
                pids.append(int(entry.th32ProcessID))
            if not kernel32.Process32Next(snap, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snap)
    return pids


def _nt_suspend_resume(pid: int, resume: bool) -> None:
    import ctypes
    from ctypes import wintypes

    PROCESS_SUSPEND_RESUME = 0x0800
    kernel32 = ctypes.windll.kernel32
    ntdll = ctypes.windll.ntdll
    handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if not handle:
        raise OSError(f"Could not open process {pid} to {'resume' if resume else 'pause'} it.")
    try:
        status = ntdll.NtResumeProcess(handle) if resume else ntdll.NtSuspendProcess(handle)
        if status != 0:
            raise OSError(f"Nt{'Resume' if resume else 'Suspend'}Process failed: {status}")
    finally:
        kernel32.CloseHandle(handle)


def suspend_process(pid: int) -> None:
    if sys.platform == "win32":
        errors = []
        for p in [pid] + _windows_child_pids(pid):
            try:
                _nt_suspend_resume(p, resume=False)
            except OSError as exc:
                errors.append(str(exc))
        if errors and errors[0].startswith("Could not open process " + str(pid)):
            raise OSError(errors[0])
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGSTOP)
    except (OSError, ProcessLookupError):
        os.kill(pid, signal.SIGSTOP)


def resume_process(pid: int) -> None:
    if sys.platform == "win32":
        for p in [pid] + _windows_child_pids(pid):
            try:
                _nt_suspend_resume(p, resume=True)
            except OSError:
                pass
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGCONT)
    except (OSError, ProcessLookupError):
        os.kill(pid, signal.SIGCONT)


class RenderController:
    """Shared pause / cancel handle for the worker thread and the UI or CLI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._paused = False
        self._cancel = threading.Event()
        self.pause_started: float | None = None
        self.paused_total = 0.0
        self.run_started: float | None = None

    def reset(self) -> None:
        with self._lock:
            self._proc = None
            self._paused = False
            self._cancel.clear()
            self.pause_started = None
            self.paused_total = 0.0
            self.run_started = time.monotonic()

    def attach(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._proc = proc
        if self._cancel.is_set():
            self.terminate()

    def set_priority(self, mode: str) -> None:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        apply_process_priority(proc.pid, mode)

    def request_cancel(self) -> None:
        self._cancel.set()
        if self.is_paused():
            try:
                self._set_paused(False)
            except Exception:
                pass
        self.terminate()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def raise_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise RenderCancelled("Cancelled.")

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def toggle_pause(self) -> bool:
        """Return the new paused state."""
        with self._lock:
            want = not self._paused
        self._set_paused(want)
        return want

    def elapsed(self) -> float:
        if self.run_started is None:
            return 0.0
        now = time.monotonic()
        with self._lock:
            extra = self.paused_total
            if self._paused and self.pause_started is not None:
                extra += now - self.pause_started
        return max(0.0, now - self.run_started - extra)

    def terminate(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is None:
            return
        terminate_process_tree(proc)

    def _set_paused(self, paused: bool) -> None:
        with self._lock:
            proc = self._proc
            already = self._paused
        if already == paused:
            return
        if proc is None or proc.poll() is not None:
            return
        if paused:
            suspend_process(proc.pid)
            with self._lock:
                self._paused = True
                self.pause_started = time.monotonic()
        else:
            resume_process(proc.pid)
            with self._lock:
                if self.pause_started is not None:
                    self.paused_total += time.monotonic() - self.pause_started
                self.pause_started = None
                self._paused = False


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    return format_time(seconds)


def format_speed(speed: float) -> str:
    if speed <= 0.05:
        return "--"
    return f"{speed:.1f}x"

@dataclass
class MediaInfo:
    path: Path
    width: int
    height: int
    fps: str
    fps_value: float
    duration: float
    has_audio: bool
    sample_rate: int
    pix_fmt: str = ""
    color_space: str = ""
    color_primaries: str = ""
    color_trc: str = ""
    color_range: str = ""
    rotation: float = 0.0


def _stream_rotation(stream: dict) -> float:
    tags = stream.get("tags") or {}
    for key in ("rotate", "ROTATE"):
        if key in tags:
            try:
                return float(tags[key])
            except ValueError:
                pass
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return float(sd["rotation"])
            except ValueError:
                pass
    return 0.0


def probe(path: Path, ffprobe: str) -> MediaInfo:
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "ffprobe failed").strip()
        raise RuntimeError(f"Could not read {path.name}: {err}")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video:
        raise RuntimeError(f"{path.name} has no video track.")

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width < 2 or height < 2:
        raise RuntimeError(f"{path.name} has an invalid size ({width}x{height}).")

    rotation = _stream_rotation(video)
    if abs(rotation) % 180 == 90:
        width, height = height, width

    sar = parse_ratio(video.get("sample_aspect_ratio"), 1.0)
    if abs(sar - 1.0) > 0.01:
        width = max(2, int(round(width * sar)))

    fps_str = video.get("r_frame_rate") or video.get("avg_frame_rate") or "30/1"
    fps_val = parse_ratio(fps_str, 0.0)
    if fps_val <= 0.1:
        fps_str = video.get("avg_frame_rate") or "30/1"
        fps_val = parse_ratio(fps_str, 30.0)
    if fps_val <= 0.1:
        fps_str, fps_val = "30/1", 30.0

    duration = 0.0
    for candidate in (
        (data.get("format") or {}).get("duration"),
        video.get("duration"),
        audio.get("duration") if audio else None,
    ):
        if candidate not in (None, "N/A", ""):
            try:
                duration = float(candidate)
                if duration > 0:
                    break
            except ValueError:
                pass
    if duration <= 0 and video.get("nb_frames") not in (None, "N/A"):
        try:
            duration = float(video["nb_frames"]) / fps_val
        except (ValueError, ZeroDivisionError):
            pass
    if duration <= 0:
        raise RuntimeError(
            f"Could not determine the duration of {path.name}. "
            "Please pick a video file."
        )

    sample_rate = 48000
    if audio and audio.get("sample_rate"):
        try:
            sample_rate = int(audio["sample_rate"])
        except ValueError:
            sample_rate = 48000

    return MediaInfo(
        path=Path(path),
        width=width,
        height=height,
        fps=fps_str,
        fps_value=fps_val,
        duration=duration,
        has_audio=audio is not None,
        sample_rate=sample_rate,
        pix_fmt=str(video.get("pix_fmt") or ""),
        color_space=str(video.get("color_space") or ""),
        color_primaries=str(video.get("color_primaries") or ""),
        color_trc=str(video.get("color_transfer") or ""),
        color_range=str(video.get("color_range") or ""),
        rotation=rotation,
    )


def probe_duration(path: Path, ffprobe: str, fallback: float) -> float:
    try:
        cmd = [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            value = float(result.stdout.strip())
            if value > 0:
                return value
    except (ValueError, OSError):
        pass
    return fallback


# ---------------------------------------------------------------------------
# Highlights-tool additions
# ---------------------------------------------------------------------------

def component_issues() -> list[str]:
    """Human-readable problems with runtime/ or assets/, or [] when fine."""
    root = install_root()
    app_manifest = read_manifest(app_dir() / "manifest.json")
    packaged = bool(app_manifest)
    issues: list[str] = []
    if not packaged:
        return issues
    need_rt = str(app_manifest.get("requires_runtime") or APP_REQUIRES_RUNTIME)
    need_as = str(app_manifest.get("requires_assets") or APP_REQUIRES_ASSETS)
    rt = root / "runtime"
    have = str(read_manifest(rt / "manifest.json").get("version") or "")
    if not rt.is_dir() or not have:
        issues.append(
            f"The runtime folder is missing or has no version file (needs {need_rt}). "
            f"Unzip the full package {FULL_ZIP} once."
        )
    elif _version_tuple(have) < _version_tuple(need_rt):
        issues.append(
            f"This app needs runtime {need_rt}, but runtime is {have}. "
            f"Unzip the full package {FULL_ZIP}."
        )
    elif sys.platform == "win32" and not (rt / "ffmpeg.exe").is_file():
        issues.append(f"runtime\\ffmpeg.exe is missing. Unzip the full package {FULL_ZIP}.")
    a = root / "assets"
    have = str(read_manifest(a / "manifest.json").get("version") or "")
    if not a.is_dir():
        issues.append(f"The assets folder is missing. Unzip the full package {FULL_ZIP}.")
    elif have and _version_tuple(have) < _version_tuple(need_as):
        issues.append(
            f"This app needs assets {need_as}, but assets is {have}. "
            f"Unzip the full package {FULL_ZIP}."
        )
    return issues


def models_dir() -> Path:
    path = install_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def work_root() -> Path:
    path = install_root() / "work"
    path.mkdir(parents=True, exist_ok=True)
    return path


def unique_name(folder: Path, stem: str, suffix: str, ext: str = ".mp4") -> str:
    base = f"{stem}{suffix}{ext}"
    if not (folder / base).exists():
        return base
    n = 2
    while (folder / f"{stem}{suffix}_{n}{ext}").exists():
        n += 1
    return f"{stem}{suffix}_{n}{ext}"


def output_stem_for(input_path: Path) -> str:
    stem = Path(input_path).stem
    # Split Insta360 pairs share one name: VID_x_00_021 / VID_x_10_021 -> VID_x_021
    stem = re.sub(r"_(00|10)_(\d+)$", r"_\2", stem)
    return stem


def default_output_for(input_path: Path, folder: Path | None = None) -> Path:
    dest = folder if folder is not None else saved_output_dir()
    dest.mkdir(parents=True, exist_ok=True)
    return dest / unique_name(dest, output_stem_for(input_path), "_highlights")


def format_ts(seconds: float) -> str:
    """mm:ss.ss (or h:mm:ss.ss) for CSV files."""
    if seconds is None:
        return ""
    neg = seconds < 0
    seconds = abs(float(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h >= 1:
        txt = f"{int(h)}:{int(m):02d}:{s:05.2f}"
    else:
        txt = f"{int(m):02d}:{s:05.2f}"
    return ("-" if neg else "") + txt


def parse_optional_time(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return parse_time(text)


def set_own_priority(mode: str) -> None:
    """Lower this Python process (detection and reprojection run here)."""
    try:
        if sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), windows_priority_class(mode))
        else:
            want = unix_nice(mode)
            cur = os.getpriority(os.PRIO_PROCESS, 0)
            if want > cur:
                os.setpriority(os.PRIO_PROCESS, 0, want)
    except Exception:
        pass


def popen_kwargs(priority_mode: str, text: bool = False) -> dict:
    kw: dict = {}
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        flags |= windows_priority_class(priority_mode)
        kw["creationflags"] = flags
    else:
        kw["start_new_session"] = True
    if text:
        kw.update(text=True, bufsize=1, universal_newlines=True)
    return kw


class JobController:
    """Pause / cancel handle shared by the worker thread and the UI or CLI.

    Python-side loops call checkpoint(), which blocks while paused and raises
    RenderCancelled after Cancel. Child processes (FFmpeg decoders/encoders)
    are suspended too, so nothing keeps using the CPU while paused.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: list[subprocess.Popen] = []
        self._cancel = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self.pause_started: float | None = None
        self.paused_total = 0.0
        self.run_started: float | None = None
        self.priority_mode = "balanced"

    def reset(self) -> None:
        with self._lock:
            self._procs = []
        self._cancel.clear()
        self._running.set()
        self.pause_started = None
        self.paused_total = 0.0
        self.run_started = time.monotonic()

    def attach(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs = [p for p in self._procs if p.poll() is None]
            self._procs.append(proc)
        try:
            apply_process_priority(proc.pid, self.priority_mode)
        except Exception:
            pass
        if self._cancel.is_set():
            terminate_process_tree(proc)
        elif self.is_paused():
            try:
                suspend_process(proc.pid)
            except Exception:
                pass

    def live_procs(self) -> list[subprocess.Popen]:
        with self._lock:
            return [p for p in self._procs if p.poll() is None]

    def set_priority(self, mode: str) -> None:
        self.priority_mode = mode
        set_own_priority(mode)
        for p in self.live_procs():
            try:
                apply_process_priority(p.pid, mode)
            except Exception:
                pass

    def request_cancel(self) -> None:
        self._cancel.set()
        if self.is_paused():
            self._set_paused(False)
        for p in self.live_procs():
            terminate_process_tree(p)

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def raise_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise RenderCancelled("Cancelled.")

    def checkpoint(self) -> None:
        while not self._running.wait(0.2):
            if self._cancel.is_set():
                break
        self.raise_if_cancelled()

    def is_paused(self) -> bool:
        return not self._running.is_set()

    def toggle_pause(self) -> bool:
        want = not self.is_paused()
        self._set_paused(want)
        return want

    def elapsed(self) -> float:
        if self.run_started is None:
            return 0.0
        now = time.monotonic()
        extra = self.paused_total
        if self.is_paused() and self.pause_started is not None:
            extra += now - self.pause_started
        return max(0.0, now - self.run_started - extra)

    def _set_paused(self, paused: bool) -> None:
        if paused == self.is_paused():
            return
        if paused:
            self._running.clear()
            self.pause_started = time.monotonic()
            for p in self.live_procs():
                try:
                    suspend_process(p.pid)
                except Exception:
                    pass
        else:
            for p in self.live_procs():
                try:
                    resume_process(p.pid)
                except Exception:
                    pass
            if self.pause_started is not None:
                self.paused_total += time.monotonic() - self.pause_started
            self.pause_started = None
            self._running.set()


class Progress:
    """Maps stage-local progress onto one 0..1 bar with ETA."""

    def __init__(self, cb: Optional[Callable[[ProgressInfo], None]], ctl: JobController):
        self.cb = cb
        self.ctl = ctl
        self.stages: list[tuple[str, float]] = []
        self.base = 0.0
        self.weight = 1.0
        self.stage = ""
        self._last = 0.0
        self._stage_t0 = time.monotonic()

    def plan(self, stages: list[tuple[str, float]]) -> None:
        total = sum(w for _, w in stages) or 1.0
        self.stages = [(n, w / total) for n, w in stages]

    def begin(self, name: str) -> None:
        base = 0.0
        weight = 0.0
        for n, w in self.stages:
            if n == name:
                weight = w
                break
            base += w
        self.base, self.weight, self.stage = base, weight, name
        self._stage_t0 = time.monotonic()
        self.update(0.0, name, force=True)

    def update(self, local: float, message: str = "", force: bool = False, speed: float = 0.0) -> None:
        if self.cb is None:
            return
        now = time.monotonic()
        if not force and now - self._last < 0.25:
            return
        self._last = now
        local = clamp(local, 0.0, 1.0)
        frac = clamp(self.base + self.weight * local, 0.0, 1.0)
        el = self.ctl.elapsed()
        eta = None
        if not self.ctl.is_paused() and frac > 0.02 and el > 2:
            eta = el * (1 - frac) / frac
        self.cb(ProgressInfo(
            fraction=frac, stage=self.stage, message=message or self.stage,
            speed=speed, elapsed=el, eta=eta, paused=self.ctl.is_paused(),
        ))


def run_ffmpeg_progress(
    cmd: list[str],
    expected: float,
    ctl: JobController,
    log: Callable[[str], None],
    on_progress: Optional[Callable[[float, float], None]] = None,
) -> None:
    """Run an FFmpeg command that has '-progress pipe:1'; report (fraction, speed)."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        **popen_kwargs(ctl.priority_mode, text=True),
    )
    ctl.attach(proc)
    errs: list[str] = []

    def pump():
        assert proc.stderr is not None
        for line in proc.stderr:
            errs.append(line)
            if "error" in line.lower():
                log(line.rstrip())

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    out_time = 0.0
    speed = 0.0
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            if ctl.cancelled():
                break
            line = raw.strip()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                v = line.split("=", 1)[1]
                if v.isdigit():
                    out_time = int(v) / 1e6
            elif line.startswith("speed="):
                v = line.split("=", 1)[1].rstrip("x").strip()
                try:
                    speed = float(v)
                except ValueError:
                    pass
            elif line.startswith("progress=") and on_progress:
                on_progress(clamp(out_time / expected, 0, 1) if expected > 0 else 0.0, speed)
    finally:
        if ctl.cancelled():
            terminate_process_tree(proc)
        rc = proc.wait()
        t.join(timeout=5)
    ctl.raise_if_cancelled()
    if rc != 0:
        raise RuntimeError("FFmpeg failed:\n" + "".join(errs[-30:]).strip())


def ffmpeg_threads_args(threads: int) -> list[str]:
    return ["-threads", str(threads), "-filter_threads", str(threads),
            "-filter_complex_threads", str(threads)]


def x264_args(preset: str, crf: int, threads: int) -> list[str]:
    return [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-x264-params", f"threads={threads}:lookahead-threads={lookahead_threads(threads)}",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-color_range", "tv",
    ]


def probe_json(path: Path, ffprobe: str) -> dict:
    r = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Could not read {Path(path).name}: {(r.stderr or '').strip()}")
    return json.loads(r.stdout or "{}")
