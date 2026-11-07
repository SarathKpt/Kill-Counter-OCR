from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import webbrowser
from pathlib import Path

import pystray
from PIL import Image, ImageDraw, ImageFont
from werkzeug.serving import make_server


APP_NAME = "FENIKZ BGMI Kill OCR"
LOCAL_APP_DATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
USER_DATA_DIR = LOCAL_APP_DATA / "FENIKZ" / "BGMI Kill OCR"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = USER_DATA_DIR / "app.log"
logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
log = logging.getLogger(APP_NAME)
_GPU_DLL_HANDLES = []


def _prepare_gpu_dlls(device: str) -> None:
    if device != "cuda":
        return
    bundled_nvidia = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "nvidia"
    if not bundled_nvidia.is_dir():
        return
    # PyInstaller places bundled NVIDIA wheel DLLs outside PATH. Keep the
    # directory handles alive so ONNX Runtime can load CUDA/cuDNN on Windows.
    for dll_dir in {dll.parent for dll in bundled_nvidia.rglob("*.dll")}:
        try:
            _GPU_DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
        except OSError:
            log.exception("Could not register NVIDIA DLL directory: %s", dll_dir)


def _single_instance_mutex():
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel32.CreateMutexW(None, True, "Local\\FENIKZ_BGMI_Kill_OCR")
    if not handle or ctypes.get_last_error() == 183:
        if handle:
            kernel32.CloseHandle(handle)
        return None
    return kernel32, handle


def _icon_image() -> Image.Image:
    image = Image.new("RGBA", (64, 64), (20, 27, 39, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, 62, 62), radius=14, fill=(38, 93, 160, 255))
    try:
        font = ImageFont.truetype("segoeui.ttf", 41)
    except OSError:
        font = ImageFont.load_default()
    draw.text((17, 5), "8", font=font, fill="white", stroke_width=1)
    return image


def main() -> int:
    # CPU/GPU entry points set this before importing this module.
    device = os.environ.get("FENIKZ_OCR_DEVICE", "cpu").lower()
    if device not in {"cpu", "cuda"}:
        device = "cpu"
        os.environ["FENIKZ_OCR_DEVICE"] = device

    mutex = _single_instance_mutex()
    if os.name == "nt" and mutex is None:
        ctypes.windll.user32.MessageBoxW(
            None,
            "FENIKZ BGMI Kill OCR is already running. Exit the current tray app before switching between CPU and GPU builds.",
            APP_NAME,
            0x40,
        )
        return 0

    try:
        _prepare_gpu_dlls(device)
        import app as service

        config = service.load_config()
        host = "0.0.0.0" if config.get("lan_access_enabled", False) else "127.0.0.1"
        port = int(config.get("port", 8765))
        server = make_server(host, port, service.app, threaded=True)
        threading.Thread(target=server.serve_forever, name="web-server", daemon=True).start()
        service.start_worker()
        control_url = f"http://127.0.0.1:{port}/"
        widget_url = f"http://{host}:{port}/widget/kill"
        config_path = service.CONFIG_PATH
        log.info("Tray app started (%s); config: %s", device, config_path)

        menu = pystray.Menu(
            pystray.MenuItem("Open control page", lambda icon, item: webbrowser.open(control_url), default=True),
            pystray.MenuItem("Open OBS widget", lambda icon, item: webbrowser.open(widget_url)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open configuration folder", lambda icon, item: os.startfile(str(config_path.parent))),
            pystray.MenuItem("Open configuration file", lambda icon, item: os.startfile(str(config_path))),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", lambda icon, item: icon.stop()),
        )
        icon = pystray.Icon(APP_NAME, _icon_image(), f"{APP_NAME} ({device.upper()})", menu)
        icon.run()
        service.state.stop_requested.set()
        server.shutdown()
        server.server_close()
        if service.worker_thread is not None:
            service.worker_thread.join(timeout=5)
        log.info("Tray app stopped")
        return 0
    except Exception as exc:
        log.exception("Could not start or run the tray app")
        if os.name == "nt":
            ctypes.windll.user32.MessageBoxW(None, f"{exc}\n\nDetails: {LOG_PATH}", APP_NAME, 0x10)
        return 1
    finally:
        if mutex:
            mutex[0].ReleaseMutex(mutex[1])
            mutex[0].CloseHandle(mutex[1])


if __name__ == "__main__":
    raise SystemExit(main())
