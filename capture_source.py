from __future__ import annotations

import threading
import ctypes

from PIL import Image
from windows_capture import WindowsCapture


class CaptureUnavailable(RuntimeError):
    """The target window is not currently producing capturable frames."""


def window_exists(title_match: str) -> bool:
    """Return whether a visible top-level window contains the requested title."""
    if not title_match:
        return False

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    found = False

    @callback_type
    def visit(hwnd, _lparam):
        nonlocal found
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, title, len(title))
                if title_match.casefold() in title.value.casefold():
                    found = True
                    return 0
        return 1

    user32.EnumWindows.argtypes = [callback_type, ctypes.c_void_p]
    user32.EnumWindows.restype = ctypes.c_int
    user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
    user32.IsWindowVisible.restype = ctypes.c_int
    user32.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.EnumWindows(visit, None)
    return found


class WindowCaptureSource:
    """Continuously keeps the latest frame from one named Windows window."""

    def __init__(self, title_match: str, minimum_update_interval_ms: int = 500):
        self.title_match = title_match
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._closed = threading.Event()
        self._frame: Image.Image | None = None
        self._error: str | None = None
        self._capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            minimum_update_interval=minimum_update_interval_ms,
            window_name=title_match,
        )

        @self._capture.event
        def on_frame_arrived(frame, _capture_control):
            try:
                # Windows Graphics Capture supplies BGRA pixels. Copy the color
                # channels before the native mapped frame buffer is released.
                image = Image.fromarray(frame.frame_buffer[:, :, [2, 1, 0]].copy(), mode="RGB")
                with self._lock:
                    self._frame = image
                    self._error = None
                self._frame_ready.set()
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                self._frame_ready.set()

        @self._capture.event
        def on_closed():
            self._closed.set()
            self._frame_ready.set()

        self._control = None

    def start(self) -> None:
        self._control = self._capture.start_free_threaded()

    def snapshot(self, timeout: float = 8.0) -> Image.Image:
        if not self._frame_ready.wait(timeout):
            raise CaptureUnavailable(f"No captured frame arrived from '{self.title_match}'.")
        if self._closed.is_set():
            raise CaptureUnavailable(f"Window capture ended for '{self.title_match}'.")
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            error = self._error
        raise CaptureUnavailable(error or f"Could not capture '{self.title_match}'.")

    def stop(self) -> None:
        if self._control is not None:
            self._control.stop()
            self._control = None
