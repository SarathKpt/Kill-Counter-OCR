from __future__ import annotations

import json
import io
import base64
import logging
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request
from PIL import Image, ImageChops, ImageFilter, ImageOps
from capture_source import CaptureUnavailable, WindowCaptureSource, window_exists

ROOT = Path(__file__).resolve().parent
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", ROOT))


def _user_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "FENIKZ" / "BGMI Kill OCR"
    return Path.home() / "AppData" / "Local" / "FENIKZ" / "BGMI Kill OCR"


if getattr(sys, "frozen", False):
    CONFIG_PATH = Path(os.environ.get("FENIKZ_OCR_CONFIG", _user_data_dir() / "config.json"))
else:
    CONFIG_PATH = Path(os.environ.get("FENIKZ_OCR_CONFIG", ROOT / "config.json"))
DEFAULT_WIDGET_HTML = '<span data-kills></span>'
DEFAULT_WIDGET_CSS = """html,body{width:100%;height:100%}body{margin:0;background:transparent;overflow:hidden;color:#fff;font:900 64px/1 Arial,sans-serif;text-shadow:2px 3px 4px #000,0 0 8px #000}#custom-widget-root{display:grid;place-items:center;width:100%;height:100%;padding:8px}.count-changed{animation:count-pop 300ms ease-out}@keyframes count-pop{50%{transform:scale(1.2)}}"""
DEFAULT_WIDGET_JS = """// React to a newly confirmed count, for example:
// window.killWidget.onChange(count => console.log('Kills:', count));
window.killWidget.onChange(count => {
  document.querySelectorAll('[data-kills], #kills').forEach(element => {
    element.classList.remove('count-changed');
    void element.offsetWidth;
    element.classList.add('count-changed');
  });
});"""
MAX_WIDGET_FIELD_LENGTH = 50000
RESTART_EXIT_CODE = 23
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fenikz-ocr")
app = Flask(__name__)


def load_config() -> dict[str, Any]:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        bundled_config = RESOURCE_ROOT / "config.json"
        if not bundled_config.exists():
            bundled_config = ROOT / "config.json"
        CONFIG_PATH.write_text(bundled_config.read_text(encoding="utf-8"), encoding="utf-8")
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    packaged_device = os.environ.get("FENIKZ_OCR_DEVICE")
    if packaged_device in {"cpu", "cuda"}:
        config["ai_device"] = packaged_device
    return config


@dataclass
class ServiceState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_requested: threading.Event = field(default_factory=threading.Event)
    running: bool = False
    waiting_for_window: bool = False
    latest_raw: str = ""
    latest_candidate: int | None = None
    latest_candidate_note: str | None = None
    latest_confidence: float | None = None
    latest_ocr_support: int = 0
    latest_error: str | None = None
    latest_read_at: str | None = None
    window_title: str | None = None
    kills: int | None = None
    pending_value: int | None = None
    pending_count: int = 0
    latest_crop_png: bytes | None = None
    latest_source_image: Image.Image | None = None
    calibration_preview_requested: bool = False
    last_valid_candidate_at: float | None = None
    stale_after_seconds: float = 3.0
    recognition_device: str = "CPU"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            reading_fresh = (self.last_valid_candidate_at is not None and
                             time.time() - self.last_valid_candidate_at <= self.stale_after_seconds)
            return {"running": self.running, "waiting_for_window": self.waiting_for_window,
                    "latest_raw": self.latest_raw,
                    "latest_candidate": self.latest_candidate,
                    "latest_candidate_note": self.latest_candidate_note,
                    "latest_confidence": self.latest_confidence,
                    "latest_ocr_support": self.latest_ocr_support,
                    "latest_error": self.latest_error, "latest_read_at": self.latest_read_at,
                    "window_title": self.window_title,
                    "recognition_device": self.recognition_device,
                    "kills": self.kills if reading_fresh else None,
                    "last_confirmed_kills": self.kills,
                    "reading_fresh": reading_fresh}


state = ServiceState()
worker_thread: threading.Thread | None = None


def capture_crop(image, region: dict[str, float] | None):
    if not region:
        raise RuntimeError("No kill region is configured. Calibrate the counter before OCR can read it.")
    x, y, w, h = (float(region[k]) for k in ("x", "y", "width", "height"))
    if min(x, y, w, h) < 0 or x + w > 1 or y + h > 1 or w == 0 or h == 0:
        raise RuntimeError("kill_region must use normalized x, y, width, and height values between 0 and 1.")
    box = (round(x * image.width), round(y * image.height),
           round((x + w) * image.width), round((y + h) * image.height))
    return image.crop(box)


def read_number(image, minimum_confidence: float = 15,
                single_pass_minimum_confidence: float = 45) -> tuple[str, int | None, float | None, int]:
    import pytesseract
    # The HUD digits sit on a changing, textured background. Compare global
    # thresholds with a local-background mask; consensus avoids trusting a single
    # low-confidence guess. BGMI's zero can also be returned as the letter O.
    gray = ImageOps.grayscale(image).resize((image.width * 8, image.height * 8))
    contrast = ImageOps.autocontrast(gray, cutoff=1)
    local_background = gray.filter(ImageFilter.GaussianBlur(radius=24))
    local_contrast = ImageChops.subtract(gray, local_background, offset=128)
    variants = (
        ("fixed", gray.point(lambda pixel: 255 if pixel > 145 else 0)),
        ("contrast", contrast.point(lambda pixel: 255 if pixel > 128 else 0)),
        ("local", local_contrast.point(lambda pixel: 255 if pixel > 140 else 0)),
    )
    results: list[tuple[str, int | None, float | None]] = []
    def read_variant(name: str, processed) -> tuple[str, int | None, float | None]:
        processed = ImageOps.expand(processed, border=24, fill=0)
        data = pytesseract.image_to_data(
            processed, config=f"--psm {name.rsplit('_', 1)[-1]} -c tessedit_char_whitelist=0123456789OoIl|",
            output_type=pytesseract.Output.DICT,
        )
        tokens, confidences = [], []
        for token, confidence in zip(data["text"], data["conf"]):
            token = token.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"}))
            token = re.sub(r"\D", "", token)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                continue
            if token and confidence >= 0:
                tokens.append(token)
                confidences.append(confidence)
        value = int("".join(tokens)) if tokens else None
        confidence = sum(confidences) / len(confidences) if confidences else None
        return name, value, confidence

    for name, processed in variants:
        results.append(read_variant(f"{name}_13", processed))
    # PSM 8 treats the crop as one word and is often better at keeping adjacent
    # narrow glyphs (for example, "11") separate in this compact HUD font.
    results.append(read_variant("local_8", local_contrast.point(lambda pixel: 255 if pixel > 140 else 0)))

    raw = " | ".join(f"{name}:{value if value is not None else 'blank'}" for name, value, _ in results)
    votes: dict[int, list[float]] = {}
    for _, value, confidence in results:
        if value is not None:
            votes.setdefault(value, []).append(confidence if confidence is not None else 0.0)
    winner = max(votes, key=lambda value: len(votes[value])) if votes else None
    confidences = votes.get(winner, []) if winner is not None else []
    confidence = sum(confidences) / len(confidences) if confidences else None
    support = len(votes.get(winner, [])) if winner is not None else 0
    if winner is not None and confidence is not None:
        if support >= 2 and confidence >= minimum_confidence:
            return raw, winner, confidence, support
        # A high-confidence local-background result can survive when global
        # thresholds fail on a changing HUD. Its caller requires extra stable reads.
        strong_local = [(value, score) for name, value, score in results
                        if name.startswith("local_") and value is not None and score is not None
                        and score >= single_pass_minimum_confidence]
        if support == 1 and any(value == winner for value, _ in strong_local):
            local_score = next(score for value, score in strong_local if value == winner)
            return raw, winner, local_score, support
    return raw, None, confidence, support


def read_number_ai(image, engine, minimum_confidence: float = 0.60) -> tuple[str, int | None, float | None, int]:
    """Read the HUD crop with RapidOCR's local ONNX model (CPU inference)."""
    # The calibrated HUD numeral is often only a few dozen pixels tall. Enlarge
    # it and add a quiet border so the text detector can find it reliably.
    scale = max(4, min(10, 192 // max(1, image.height)))
    prepared = ImageOps.grayscale(image).resize(
        (max(1, image.width * scale), max(1, image.height * scale)), Image.Resampling.LANCZOS
    )
    prepared = ImageOps.autocontrast(prepared, cutoff=1).convert("RGB")
    prepared = ImageOps.expand(prepared, border=max(16, prepared.height // 4), fill="black")
    result = engine(prepared)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    texts = [] if texts is None else texts
    scores = [] if scores is None else scores
    parsed: list[tuple[str, float]] = []
    for text, score in zip(texts, scores):
        digits = re.sub(r"\D", "", str(text))
        try:
            confidence = float(score)
        except (TypeError, ValueError):
            continue
        # BGMI's kill counter is at most two digits. Ignore incidental OCR of
        # other UI text or scenery that may enter a slightly loose crop.
        if digits and len(digits) <= 2 and 0 <= int(digits) <= 99:
            parsed.append((digits, confidence))
    if not parsed:
        return "AI:blank", None, None, 0

    # A region calibrated tightly around the counter should contain one text
    # line. If it contains more, prefer the highest-confidence numeric line.
    digits, confidence = max(parsed, key=lambda item: item[1])
    raw = f"AI:{digits} ({confidence:.0%})"
    if confidence < minimum_confidence:
        return raw, None, confidence * 100, 0
    return raw, int(digits), confidence * 100, 1


def accept_candidate(candidate: int | None, config: dict[str, Any], support: int = 0) -> None:
    with state.lock:
        state.latest_candidate = candidate
        state.latest_candidate_note = None
        if candidate is None:
            state.pending_value, state.pending_count = None, 0
            return
        max_increase = max(1, int(config.get("max_kill_increase_per_update", 2)))
        if state.kills is not None and candidate < state.kills and candidate != 0:
            state.pending_value, state.pending_count = None, 0
            state.latest_candidate_note = (
                f"Ignored decrease {state.kills}→{candidate}; only a confirmed zero resets a match."
            )
            return
        if state.kills is not None and candidate > state.kills + max_increase:
            state.pending_value, state.pending_count = None, 0
            state.latest_candidate_note = (
                f"Ignored jump {state.kills}→{candidate}; maximum increase is {max_increase} per reading."
            )
            return
        state.last_valid_candidate_at = time.time()
        if candidate == state.kills:
            state.pending_value, state.pending_count = None, 0
            return
        if candidate == state.pending_value:
            state.pending_count += 1
        else:
            state.pending_value, state.pending_count = candidate, 1
        if state.kills is not None and candidate == 0 and candidate < state.kills:
            key = "stable_readings_to_accept_match_reset"
        elif support == 1:
            key = "stable_readings_to_accept_single_pass"
        else:
            key = "stable_readings_to_accept"
        if state.pending_count >= max(1, int(config.get(key, 2))):
            state.kills = candidate
            state.pending_value, state.pending_count = None, 0


def create_ai_engine(config: dict[str, Any], allow_cuda: bool = True):
    from rapidocr import RapidOCR
    import onnxruntime as ort

    cuda_requested = allow_cuda and str(config.get("ai_device", "cpu")).lower() == "cuda"
    cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
    use_cuda = cuda_requested and cuda_available
    if use_cuda:
        device_name = "CUDA (NVIDIA GPU)"
    elif cuda_requested:
        device_name = "CPU (CUDA provider unavailable)"
    else:
        device_name = "CPU"
    if use_cuda and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            log.exception("Could not preload CUDA/cuDNN libraries; CUDA inference may fail")
    ai_threads = max(1, int(config.get("ai_cpu_threads", 1)))
    engine = RapidOCR(params={
        # Keep the full recognition pipeline for accuracy.
        "EngineConfig.onnxruntime.use_cuda": use_cuda,
        "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": int(config.get("ai_cuda_device_id", 0)),
        # One shared CUDA stream is the closest CUDA equivalent to a single
        # execution lane; GPU kernels still use their normal hardware threads.
        "EngineConfig.onnxruntime.cuda_ep_cfg.use_ep_level_unified_stream": bool(
            config.get("ai_cuda_single_stream", True)
        ),
        "EngineConfig.onnxruntime.intra_op_num_threads": ai_threads,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })
    return engine, device_name


def actual_ai_device(engine, cuda_requested: bool) -> str:
    """Report the providers attached to RapidOCR's initialized ONNX sessions."""
    sessions = []
    for stage_name in ("text_det", "text_cls", "text_rec"):
        stage = getattr(engine, stage_name, None)
        inference_session = getattr(getattr(stage, "session", None), "session", None)
        if inference_session is not None:
            try:
                sessions.append(inference_session.get_providers())
            except Exception:
                continue
    if cuda_requested and sessions and all("CUDAExecutionProvider" in providers for providers in sessions):
        return "CUDA (NVIDIA GPU)"
    if cuda_requested:
        return "CPU fallback (CUDA libraries unavailable)"
    return "CPU"


def ocr_loop() -> None:
    with state.lock:
        state.running = True
    capture_source = None
    captured_title = None
    captured_interval_ms = None
    waiting_for_window = False
    ai_engine = None
    ai_device_active = "CPU"
    while not state.stop_requested.is_set():
        started = time.monotonic()
        try:
            config = load_config()
            with state.lock:
                state.stale_after_seconds = max(0.5, float(config.get("stale_after_seconds", 3.0)))
            title_hint = str(config.get("window_title_contains", "MobileStream"))
            capture_interval_ms = max(0, int(config.get("capture_minimum_update_interval_ms", 500)))
            if capture_source is not None and (
                    captured_title != title_hint or captured_interval_ms != capture_interval_ms):
                capture_source.stop()
                capture_source = None
                captured_title = None
                captured_interval_ms = None
            if capture_source is None:
                if not window_exists(title_hint):
                    with state.lock:
                        state.waiting_for_window = True
                        state.latest_error = None
                        state.window_title = title_hint
                    if not waiting_for_window:
                        log.info("Capture window containing '%s' not found; waiting for it to appear", title_hint)
                    waiting_for_window = True
                    state.stop_requested.wait(1.0)
                    continue
                try:
                    capture_source = WindowCaptureSource(title_hint, capture_interval_ms)
                    capture_source.start()
                    captured_title = title_hint
                    captured_interval_ms = capture_interval_ms
                except Exception as exc:
                    raise CaptureUnavailable(f"Could not start capture for '{title_hint}': {exc}") from exc
            try:
                source_image = capture_source.snapshot()
            except CaptureUnavailable:
                raise
            except Exception as exc:
                raise CaptureUnavailable(f"Capture failed for '{title_hint}': {exc}") from exc
            if waiting_for_window:
                log.info("Capture window containing '%s' is available; resuming recognition", title_hint)
                waiting_for_window = False
                with state.lock:
                    state.waiting_for_window = False
            with state.lock:
                if state.calibration_preview_requested:
                    state.latest_source_image = source_image
            crop_image = capture_crop(source_image, config.get("kill_region"))
            crop_buffer = io.BytesIO()
            crop_image.save(crop_buffer, format="PNG")
            with state.lock:
                state.latest_crop_png = crop_buffer.getvalue()
            engine_name = str(config.get("recognition_engine", "rapidocr")).lower()
            if engine_name == "rapidocr":
                if ai_engine is None:
                    ai_engine, ai_device_active = create_ai_engine(config)
                    with state.lock:
                        state.recognition_device = ai_device_active
                    log.info("RapidOCR is configured for %s", ai_device_active)
                try:
                    raw, candidate, confidence, support = read_number_ai(
                        crop_image, ai_engine,
                        float(config.get("ai_minimum_confidence", 0.60)),
                    )
                except Exception:
                    if not ai_device_active.startswith("CUDA"):
                        raise
                    log.exception("CUDA recognition failed; switching this run to CPU")
                    ai_engine, ai_device_active = create_ai_engine(config, allow_cuda=False)
                    with state.lock:
                        state.recognition_device = "CPU fallback (CUDA failed)"
                    raw, candidate, confidence, support = read_number_ai(
                        crop_image, ai_engine,
                        float(config.get("ai_minimum_confidence", 0.60)),
                    )
                if str(config.get("ai_device", "cpu")).lower() == "cuda":
                    confirmed_device = actual_ai_device(ai_engine, cuda_requested=True)
                    if confirmed_device != ai_device_active:
                        ai_device_active = confirmed_device
                        with state.lock:
                            state.recognition_device = confirmed_device
                        if confirmed_device.startswith("CPU fallback"):
                            log.warning("CUDA was requested, but RapidOCR is actually using CPU")
            elif engine_name == "tesseract":
                import pytesseract
                if config.get("tesseract_cmd"):
                    pytesseract.pytesseract.tesseract_cmd = config["tesseract_cmd"]
                raw, candidate, confidence, support = read_number(
                    crop_image,
                    float(config.get("minimum_ocr_confidence", 15)),
                    float(config.get("single_pass_minimum_confidence", 45)),
                )
            else:
                raise RuntimeError(f"Unknown recognition_engine: {engine_name}")
            accept_candidate(candidate, config, support)
            with state.lock:
                state.waiting_for_window = False
                state.latest_raw, state.latest_confidence = raw, confidence
                state.latest_ocr_support = support
                state.latest_error = None
                state.latest_read_at = datetime.now(timezone.utc).isoformat()
                state.window_title = title_hint
                if engine_name != "rapidocr":
                    state.recognition_device = engine_name
        except CaptureUnavailable as exc:
            if capture_source is not None:
                try:
                    capture_source.stop()
                except Exception:
                    pass
            capture_source = None
            captured_title = None
            captured_interval_ms = None
            if not waiting_for_window:
                log.info("%s Waiting to reconnect to the capture window.", exc)
            waiting_for_window = True
            with state.lock:
                state.waiting_for_window = True
                state.latest_error = None
                state.window_title = title_hint
            time.sleep(1.0)
            continue
        except Exception as exc:
            log.exception("Recognition cycle failed")
            if capture_source is not None:
                try:
                    capture_source.stop()
                except Exception:
                    log.exception("Could not stop the capture source after an error")
                capture_source = None
                captured_title = None
                captured_interval_ms = None
            with state.lock:
                state.waiting_for_window = False
                state.latest_error = str(exc)
                state.latest_read_at = datetime.now(timezone.utc).isoformat()
                state.window_title = None
        try:
            interval = max(0.2, float(load_config().get("poll_interval_seconds", 0.8)))
        except Exception:
            interval = 0.8
        state.stop_requested.wait(max(0.0, interval - (time.monotonic() - started)))
    if capture_source is not None:
        try:
            capture_source.stop()
        except Exception:
            log.exception("Could not stop the capture source during shutdown")
    with state.lock:
        state.running = False


def start_worker() -> None:
    global worker_thread
    if worker_thread is None or not worker_thread.is_alive():
        state.stop_requested.clear()
        worker_thread = threading.Thread(target=ocr_loop, name="ocr-worker", daemon=True)
        worker_thread.start()


def start_console_reload_watcher() -> None:
    if os.name != "nt" or not sys.stdin.isatty():
        return

    def watch_for_reload() -> None:
        import msvcrt

        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b"\x12":  # Ctrl+R
                    log.info("Ctrl+R pressed; asking the launcher to restart the server")
                    logging.shutdown()
                    os._exit(RESTART_EXIT_CODE)
            time.sleep(0.05)

    threading.Thread(target=watch_for_reload, name="console-reload", daemon=True).start()


@app.get("/control")
def control_page():
    return """<!doctype html><html><head><meta charset='utf-8'><title>FENIKZ OCR Control</title>
<style>body{font:16px system-ui;background:#10151d;color:#eef3fa;margin:32px;max-width:760px}article{background:#1a2330;padding:22px;border-radius:12px;margin:16px 0}strong{font-size:42px}button{padding:10px 14px;border:0;border-radius:7px;cursor:pointer}#error{color:#ffb078}small{color:#aab6c5}#crop{display:block;image-rendering:pixelated;max-width:100%;max-height:220px;background:#080a0e;margin-top:10px;padding:8px}</style></head>
<body><h1>BGMI Kill OCR</h1><article><small>CONFIRMED KILLS</small><br><strong id='kills'>—</strong><p id='service'></p><button onclick='resetCount()'>Reset match count</button></article>
<article><b>Calibrate in this browser</b><p>Choose the capture window and drag around only the kill number.</p><a href='/calibrate' style='color:#9dc8ff'>Open browser calibration</a></article>
<article><b>OBS number widget</b><p>Use this transparent page as a Browser Source in OBS.</p><a href='/widget' style='color:#9dc8ff'>Open number-only widget</a></article>
<article><b>Customize the OBS widget</b><p>Edit its HTML, CSS, and JavaScript.</p><a href='/customize' style='color:#9dc8ff'>Customize widget</a></article>
<article><b>Configuration</b><p>Your settings are stored in a readable JSON file. Export a backup or import a configuration file.</p><button onclick="location.href='/api/config/export'">Export configuration</button> <label style="cursor:pointer;background:#4c91e8;padding:10px 14px;border-radius:7px"><input id="configImport" type="file" accept=".json,application/json" hidden>Import configuration</label><p id="configMessage" role="status"></p><small id="configPath"></small></article>
<article><b>OCR diagnostics</b><p id='candidate'>No consensus yet</p><p id='raw'>Waiting for first reading</p><p id='meta'></p><p id='error'></p><small>Latest crop being sent to OCR</small><img id='crop' alt='OCR crop not available yet'></article>
<script>let lastRead='';async function refresh(){try{const s=await(await fetch('/api/status')).json();document.querySelector('#kills').textContent=s.kills??'—';document.querySelector('#service').textContent=s.waiting_for_window?`Waiting for window ${s.window_title||'MobileStream'}…`:!s.running?'Starting recognition service':(!s.reading_fresh&&s.last_confirmed_kills!==null?`No reliable HUD reading · last confirmed ${s.last_confirmed_kills}`:`Recognition running on ${s.recognition_device||'CPU'}`);document.querySelector('#candidate').textContent=s.latest_candidate_note||(s.latest_candidate==null?'No accepted candidate':`Candidate: ${s.latest_candidate}`);document.querySelector('#raw').textContent=s.latest_raw||'No digits detected';document.querySelector('#meta').textContent=[s.latest_confidence==null?'':`Confidence ${s.latest_confidence.toFixed(0)}%`,s.recognition_device||'',s.window_title||'',s.latest_read_at||''].filter(Boolean).join(' · ');document.querySelector('#error').textContent=s.latest_error||'';if(s.latest_read_at&&s.latest_read_at!==lastRead){lastRead=s.latest_read_at;document.querySelector('#crop').src='/api/crop.png?t='+Date.now()}}catch(e){document.querySelector('#error').textContent='Could not reach the local service.'}}document.querySelector('#configImport').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;const note=document.querySelector('#configMessage');note.textContent='Importing…';try{const response=await fetch('/api/config/import',{method:'PUT',headers:{'Content-Type':'application/json'},body:await file.text()});const data=await response.json();if(!response.ok)throw new Error(data.error||'Could not import configuration.');note.textContent='Configuration imported. Restart the service to apply all settings.'}catch(error){note.textContent=error.message||String(error)}finally{event.target.value=''}});fetch('/api/config/location').then(r=>r.json()).then(data=>document.querySelector('#configPath').textContent='Configuration file: '+data.path).catch(()=>{});async function resetCount(){await fetch('/api/reset',{method:'POST'});refresh()}refresh();setInterval(refresh,750)</script></body></html>"""


@app.get("/")
def main_page():
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>BGMI Kill Count</title>
<style>body{font:16px system-ui;background:#10151d;color:#eef3fa;margin:32px;max-width:520px}article{background:#1a2330;padding:24px;border-radius:12px}strong{font-size:56px}small{color:#aab6c5}a{color:#9dc8ff}</style></head>
<body><h1>BGMI Kill Count</h1><article><small>CONFIRMED KILLS</small><br><strong id='kills'>—</strong><p id='service'>Connecting to local service…</p><a href='/control'>Open control and diagnostics</a><br><a href='/customize'>Customize OBS widget</a></article>
<script>async function refresh(){try{const s=await(await fetch('/api/status')).json();document.querySelector('#kills').textContent=s.kills??'—';document.querySelector('#service').textContent=s.waiting_for_window?`Waiting for window ${s.window_title||'MobileStream'}…`:!s.running?'Starting recognition service':(!s.reading_fresh&&s.last_confirmed_kills!==null?`No reliable HUD reading · last confirmed ${s.last_confirmed_kills}`:`Recognition running on ${s.recognition_device||'CPU'}`)}catch(e){document.querySelector('#service').textContent='Could not reach the local service.'}}refresh();setInterval(refresh,750)</script></body></html>"""


@app.get("/calibrate")
def browser_calibration_page():
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Calibrate kill counter</title>
<style>body{font:16px system-ui;background:#10151d;color:#eef3fa;margin:24px auto;padding:0 16px;max-width:1100px}article{background:#1a2330;padding:18px;border-radius:12px;margin:14px 0}button,input{font:inherit;padding:9px 12px;border:0;border-radius:7px}button{cursor:pointer;margin:4px}button.primary{background:#4c91e8;color:#fff}input{background:#0d1219;color:#eef3fa;border:1px solid #465365;width:min(420px,90%)}#stage{position:relative;background:#080a0e;min-height:160px;height:62vh;display:flex;align-items:center;justify-content:center;overflow:hidden;border-radius:8px}#shareVideo{display:block;max-width:100%;max-height:62vh;object-fit:contain}#selectCanvas{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);max-width:100%;max-height:62vh;width:auto;height:auto;cursor:crosshair;touch-action:none}#status{color:#aab6c5;min-height:1.5em}#status.error{color:#ffb078}a{color:#9dc8ff}</style></head><body>
<h1>Calibrate the kill counter</h1><article><p>Choose which window to capture. The default is <b>MobileStream</b>; enter any distinctive part of the target window's title.</p>
<p><label for='windowTitleInput'>Capture window title contains</label><br><input id='windowTitleInput' maxlength='120' value='MobileStream'><button id='saveWindow'>Save window</button></p><p id='windowStatus' role='status'></p>
<p>Then drag a tight rectangle around only the kill numeral in the preview.</p>
<p><button class='primary' id='startPreview'>Start preview</button><button id='stopPreview' disabled>Stop preview</button></p><div id='stage'><canvas id='selectCanvas'></canvas></div>
<p>Keep the `Finished` label and other numbers outside the box.</p><button class='primary' id='saveRegion' disabled>Save calibration</button><a href='/control' style='margin-left:12px'>Back to control</a><p id='status'>Start the server preview, then drag around the number.</p></article>
<script>
const canvas=document.querySelector('#selectCanvas'), ctx=canvas.getContext('2d'), stage=document.querySelector('#stage'), frame=new Image();
let timer=null, selection=null, dragOrigin=null, previewActive=false;
function say(text,error=false){const e=document.querySelector('#status');e.textContent=text;e.classList.toggle('error',error)}
async function loadConfig(){try{const config=await(await fetch('/api/config')).json();document.querySelector('#windowTitleInput').value=config.window_title_contains||'MobileStream'}catch{}}
function draw(){if(!frame.complete||!frame.naturalWidth)return;const w=frame.naturalWidth,h=frame.naturalHeight;if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}const scale=Math.min(stage.clientWidth/w,stage.clientHeight/h);canvas.style.width=`${w*scale}px`;canvas.style.height=`${h*scale}px`;ctx.clearRect(0,0,w,h);ctx.drawImage(frame,0,0,w,h);if(selection){const x=Math.min(selection.x1,selection.x2),y=Math.min(selection.y1,selection.y2),rw=Math.abs(selection.x2-selection.x1),rh=Math.abs(selection.y2-selection.y1);ctx.fillStyle='rgba(0,0,0,.48)';ctx.fillRect(0,0,w,h);ctx.clearRect(x,y,rw,rh);ctx.drawImage(frame,x,y,rw,rh,x,y,rw,rh);ctx.strokeStyle='#ff365d';ctx.lineWidth=Math.max(2,w/900);ctx.strokeRect(x,y,rw,rh)}}
function point(e){const r=canvas.getBoundingClientRect();return{x:Math.max(0,Math.min(canvas.width,(e.clientX-r.left)*canvas.width/r.width)),y:Math.max(0,Math.min(canvas.height,(e.clientY-r.top)*canvas.height/r.height))}}
async function refreshFrame(){try{const r=await fetch('/api/source.jpg?t='+Date.now());if(!r.ok)throw new Error(await r.text()||'Waiting for a server-captured frame…');const blob=await r.blob(),url=URL.createObjectURL(blob);frame.onload=()=>{URL.revokeObjectURL(url);draw()};frame.src=url}catch(e){say(e.message||String(e))}}
canvas.addEventListener('pointerdown',e=>{if(!previewActive||!frame.naturalWidth)return;canvas.setPointerCapture(e.pointerId);dragOrigin=point(e);selection={x1:dragOrigin.x,y1:dragOrigin.y,x2:dragOrigin.x,y2:dragOrigin.y};draw()});
canvas.addEventListener('pointermove',e=>{if(!dragOrigin)return;const p=point(e);selection.x2=p.x;selection.y2=p.y;document.querySelector('#saveRegion').disabled=Math.abs(selection.x2-selection.x1)<5||Math.abs(selection.y2-selection.y1)<5;draw()});
canvas.addEventListener('pointerup',()=>{dragOrigin=null});
addEventListener('resize',draw);
document.querySelector('#startPreview').onclick=async()=>{try{const r=await fetch('/api/calibration-preview/start',{method:'POST'});if(!r.ok)throw new Error('Could not start the server preview.');previewActive=true;selection=null;document.querySelector('#saveRegion').disabled=true;document.querySelector('#stopPreview').disabled=false;document.querySelector('#startPreview').disabled=true;say('Requesting a captured frame from the server…');await refreshFrame();timer=setInterval(refreshFrame,750)}catch(e){say(e.message||String(e),true)}};
async function stopPreview(){if(timer){clearInterval(timer);timer=null}previewActive=false;document.querySelector('#stopPreview').disabled=true;document.querySelector('#startPreview').disabled=false;await fetch('/api/calibration-preview/stop',{method:'POST'}).catch(()=>{});say('Server preview stopped.')}
document.querySelector('#stopPreview').onclick=stopPreview;
document.querySelector('#saveWindow').onclick=async()=>{const title=document.querySelector('#windowTitleInput').value.trim(),status=document.querySelector('#windowStatus');status.textContent='';if(!title){status.textContent='Enter part of the window title.';return}try{const r=await fetch('/api/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({window_title_contains:title})});const data=await r.json();if(!r.ok)throw new Error(data.error||'Could not save the window title.');status.textContent=`Saved. Waiting for a window containing “${data.window_title_contains}”; capture switches automatically when it appears.`}catch(e){status.textContent=e.message||String(e)}};
document.querySelector('#saveRegion').onclick=async()=>{if(!selection||!frame.naturalWidth)return;const x=Math.min(selection.x1,selection.x2),y=Math.min(selection.y1,selection.y2),w=Math.abs(selection.x2-selection.x1),h=Math.abs(selection.y2-selection.y1);try{const r=await fetch('/api/calibration',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({kill_region:{x:x/frame.naturalWidth,y:y/frame.naturalHeight,width:w/frame.naturalWidth,height:h/frame.naturalHeight}})});const data=await r.json();if(!r.ok)throw new Error(data.error||'Could not save calibration.');document.querySelector('#saveRegion').disabled=true;await stopPreview();say('Calibration saved on the server. The same window capture continues for OCR.')}catch(e){say(e.message||String(e),true)}};
addEventListener('pagehide',()=>{if(previewActive)navigator.sendBeacon('/api/calibration-preview/stop')});
loadConfig();
</script></body></html>"""


@app.get("/customize")
def customize_widget_page():
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Customize OBS widget</title>
<style>body{font:16px system-ui;background:#10151d;color:#eef3fa;margin:24px auto;padding:0 16px;max-width:1000px}article{background:#1a2330;padding:18px;border-radius:12px;margin:14px 0}button{font:inherit;padding:9px 13px;border:0;border-radius:7px;cursor:pointer;margin:4px}textarea{display:block;width:100%;min-height:130px;box-sizing:border-box;background:#0d1219;color:#eef3fa;border:1px solid #465365;border-radius:7px;padding:12px;font:14px/1.45 Consolas,monospace;resize:vertical}label{display:block;font-weight:700;margin:14px 0 7px}small,#message{color:#b7c3d2}a{color:#9dc8ff}iframe{width:100%;height:180px;border:1px solid #465365;border-radius:8px;background:repeating-conic-gradient(#202938 0 25%,#151b25 0 50%) 50%/24px 24px}</style></head><body>
<h1>Customize OBS widget</h1><p>Edit the widget's HTML, CSS, and JavaScript. Changes are saved only when you select <b>Save and preview</b>; the page in OBS stays transparent.</p>
<article><b>Live preview</b><p><iframe id='preview' title='Widget preview' src='/widget'></iframe></p><button id='save'>Save and preview</button><button id='defaults'>Restore defaults</button><span id='message' role='status'></span></article>
<article><label for='html'>HTML</label><small>Use <code>{{kills}}</code> or an element with <code>data-kills</code> where the number should appear.</small><textarea id='html' spellcheck='false'></textarea>
<label for='css'>CSS</label><textarea id='css' spellcheck='false'></textarea>
<label for='js'>JavaScript</label><small>Use <code>window.killWidget.onChange(count =&gt; { ... })</code> to run an animation when the confirmed count changes. The latest value is in <code>window.killWidget.count</code>.</small><textarea id='js' spellcheck='false'></textarea>
<p><button id='saveBottom'>Save and preview</button><span id='bottomMessage' role='status'></span></p>
<p><a href='/'>Home</a> · <a href='/control'>Control page</a> · <a href='/widget' target='_blank'>Open widget URL</a></p></article>
<script>
const fields={html:document.querySelector('#html'),css:document.querySelector('#css'),js:document.querySelector('#js')};let defaults={};
async function load(){try{const r=await fetch('/api/widget-settings');const data=await r.json();defaults=data.defaults;for(const key of Object.keys(fields))fields[key].value=data[key]}catch(e){document.querySelector('#message').textContent='Could not load widget settings.'}}
async function save(useDefaults=false){const messages=[document.querySelector('#message'),document.querySelector('#bottomMessage')],body=useDefaults?defaults:Object.fromEntries(Object.entries(fields).map(([key,field])=>[key,field.value]));messages.forEach(message=>message.textContent='Saving…');try{const r=await fetch('/api/widget-settings',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),data=await r.json();if(!r.ok)throw new Error(data.error||'Could not save widget settings.');if(useDefaults)for(const key of Object.keys(fields))fields[key].value=data[key];defaults=data.defaults;document.querySelector('#preview').src='/widget?preview='+Date.now();messages.forEach(message=>message.textContent='Saved. Refresh the OBS Browser Source if it is already open.')}catch(e){messages.forEach(message=>message.textContent=e.message||String(e))}}
document.querySelector('#save').onclick=()=>save();document.querySelector('#saveBottom').onclick=()=>save();document.querySelector('#defaults').onclick=()=>save(true);load();
</script></body></html>"""


@app.get("/overlay")
@app.get("/widget")
def overlay_page():
    config = load_config()
    html = str(config.get("widget_html", DEFAULT_WIDGET_HTML))
    css = str(config.get("widget_css", DEFAULT_WIDGET_CSS))
    javascript = str(config.get("widget_js", DEFAULT_WIDGET_JS))
    encoded_html = base64.b64encode(html.encode("utf-8")).decode("ascii")
    encoded_css = base64.b64encode(css.encode("utf-8")).decode("ascii")
    encoded_js = base64.b64encode(javascript.encode("utf-8")).decode("ascii")
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Kill count widget</title></head><body><div id='custom-widget-root'></div><script>
const decodeWidgetPart=value=>new TextDecoder().decode(Uint8Array.from(atob(value),character=>character.charCodeAt(0)));
const root=document.querySelector('#custom-widget-root');root.style.visibility='hidden';
const style=document.createElement('style');style.textContent='*{box-sizing:border-box}html,body{margin:0;background:transparent;overflow:hidden}'+decodeWidgetPart('""" + encoded_css + """');document.head.append(style);
root.innerHTML=decodeWidgetPart('""" + encoded_html + """').replaceAll('{{kills}}','<span data-kills></span>');
let currentCount=null;window.killWidget={get count(){return currentCount},onChange(handler){window.addEventListener('killcountchange',event=>handler(event.detail.count))}};
function setCount(value){const next=Number.isInteger(value)?value:null;if(next===currentCount)return;currentCount=next;root.querySelectorAll('[data-kills],#kills').forEach(element=>element.textContent=next===null?'':String(next));root.style.visibility=next===null?'hidden':'visible';window.dispatchEvent(new CustomEvent('killcountchange',{detail:{count:next}}))}
async function refreshCount(){try{const status=await(await fetch('/api/status')).json();setCount(status.kills)}catch{setCount(null)}}
const customScript=document.createElement('script');customScript.textContent=decodeWidgetPart('""" + encoded_js + """');document.body.append(customScript);refreshCount();setInterval(refreshCount,750);
</script></body></html>"""


@app.get("/api/widget-settings")
def api_get_widget_settings():
    config = load_config()
    defaults = {"html": DEFAULT_WIDGET_HTML, "css": DEFAULT_WIDGET_CSS, "js": DEFAULT_WIDGET_JS}
    return jsonify({"html": config.get("widget_html", defaults["html"]),
                    "css": config.get("widget_css", defaults["css"]),
                    "js": config.get("widget_js", defaults["js"]),
                    "defaults": defaults})


@app.put("/api/widget-settings")
def api_save_widget_settings():
    payload = request.get_json(silent=True) or {}
    values = {key: payload.get(key, "") for key in ("html", "css", "js")}
    if any(not isinstance(value, str) or len(value) > MAX_WIDGET_FIELD_LENGTH for value in values.values()):
        return jsonify({"error": f"Each widget section must be text no longer than {MAX_WIDGET_FIELD_LENGTH} characters."}), 400
    try:
        config = load_config()
        for key, value in values.items():
            config[f"widget_{key}"] = value
        temp_path = CONFIG_PATH.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_path, CONFIG_PATH)
    except Exception as exc:
        log.exception("Could not save widget customization")
        return jsonify({"error": f"Could not save widget customization: {exc}"}), 500
    return jsonify({**values, "defaults": {"html": DEFAULT_WIDGET_HTML,
                                             "css": DEFAULT_WIDGET_CSS,
                                             "js": DEFAULT_WIDGET_JS}})


@app.post("/api/reset")
def api_reset():
    with state.lock:
        state.kills, state.pending_value, state.pending_count = 0, None, 0
        state.latest_candidate_note = None
        state.last_valid_candidate_at = time.time()
    return {"ok": True, "kills": 0}


@app.get("/api/status")
def api_status():
    return jsonify(state.snapshot())


@app.get("/api/config/location")
def api_config_location():
    return jsonify({"path": str(CONFIG_PATH)})


@app.get("/api/config/export")
def api_export_config():
    try:
        data = json.dumps(load_config(), indent=2, ensure_ascii=False) + "\n"
        return Response(data, mimetype="application/json",
                        headers={"Content-Disposition": 'attachment; filename="fenikz-ocr-config.json"'})
    except Exception as exc:
        return jsonify({"error": f"Could not export configuration: {exc}"}), 500


@app.put("/api/config/import")
def api_import_config():
    incoming = request.get_json(silent=True)
    if not isinstance(incoming, dict):
        return jsonify({"error": "Choose a valid JSON configuration object."}), 400
    try:
        # Retain host/port from this installation so imported settings cannot
        # unexpectedly expose the local service or change the OBS URL.
        current = load_config()
        title = incoming.get("window_title_contains", current.get("window_title_contains"))
        if not isinstance(title, str) or not title.strip() or len(title) > 120:
            raise ValueError("window_title_contains must be text.")
        if incoming.get("recognition_engine", current.get("recognition_engine")) not in {"rapidocr", "tesseract"}:
            raise ValueError("recognition_engine must be rapidocr or tesseract.")
        if incoming.get("ai_device", current.get("ai_device")) not in {"cpu", "cuda"}:
            raise ValueError("ai_device must be cpu or cuda.")
        region = incoming.get("kill_region", current.get("kill_region"))
        if region is not None:
            if not isinstance(region, dict):
                raise ValueError("kill_region must be an object or null.")
            x, y, width, height = (float(region[key]) for key in ("x", "y", "width", "height"))
            if min(x, y, width, height) < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
                raise ValueError("kill_region coordinates must fit within the captured image.")
        numeric_ranges = {
            "ai_cuda_device_id": (0, 32), "ai_cpu_threads": (1, 64),
            "ai_minimum_confidence": (0, 1), "poll_interval_seconds": (0.2, 60),
            "capture_minimum_update_interval_ms": (0, 10000), "stale_after_seconds": (0.5, 300),
            "stable_readings_to_accept": (1, 30), "stable_readings_to_accept_single_pass": (1, 30),
            "stable_readings_to_accept_match_reset": (1, 30), "max_kill_increase_per_update": (1, 100),
            "minimum_ocr_confidence": (0, 100), "single_pass_minimum_confidence": (0, 100),
        }
        for key, (minimum, maximum) in numeric_ranges.items():
            try:
                value = float(incoming.get(key, current.get(key, minimum)))
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number.") from None
            if not minimum <= value <= maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}.")
        for key in ("widget_html", "widget_css", "widget_js"):
            value = incoming.get(key, current.get(key, ""))
            if not isinstance(value, str) or len(value) > MAX_WIDGET_FIELD_LENGTH:
                raise ValueError(f"{key} must be text no longer than {MAX_WIDGET_FIELD_LENGTH} characters.")
        imported = dict(current)
        imported.update(incoming)
        imported["host"] = current.get("host", "127.0.0.1")
        imported["port"] = current.get("port", 8765)
        # A packaged CPU/GPU build always keeps its selected inference backend.
        packaged_device = os.environ.get("FENIKZ_OCR_DEVICE")
        if packaged_device in {"cpu", "cuda"}:
            imported["ai_device"] = packaged_device
        temp_path = CONFIG_PATH.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(imported, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temp_path, CONFIG_PATH)
        return jsonify({"ok": True})
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return jsonify({"error": f"Configuration was not imported: {exc}"}), 400


@app.get("/api/config")
def api_public_config():
    config = load_config()
    return jsonify({"window_title_contains": config.get("window_title_contains", "MobileStream")})


@app.put("/api/config")
def api_update_config():
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("window_title_contains", "")).strip()
    if not title or len(title) > 120:
        return jsonify({"error": "Enter a window title between 1 and 120 characters."}), 400
    try:
        config = load_config()
        config["window_title_contains"] = title
        temp_path = CONFIG_PATH.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_path, CONFIG_PATH)
    except Exception as exc:
        log.exception("Could not save the capture window title")
        return jsonify({"error": f"Could not save the window title: {exc}"}), 500
    return jsonify({"ok": True, "window_title_contains": title})


@app.post("/api/calibration-preview/start")
def api_start_calibration_preview():
    with state.lock:
        state.calibration_preview_requested = True
        state.latest_source_image = None
    return jsonify({"ok": True})


@app.post("/api/calibration-preview/stop")
def api_stop_calibration_preview():
    with state.lock:
        state.calibration_preview_requested = False
        state.latest_source_image = None
    return jsonify({"ok": True})


@app.get("/api/source.jpg")
def api_source_preview():
    with state.lock:
        source_image = state.latest_source_image
        capture_error = state.latest_error
    if source_image is None:
        return capture_error or "Waiting for the first server-captured frame.", 503
    image_buffer = io.BytesIO()
    source_image.save(image_buffer, format="JPEG", quality=82)
    return Response(image_buffer.getvalue(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.put("/api/calibration")
def api_save_calibration():
    payload = request.get_json(silent=True) or {}
    region = payload.get("kill_region")
    title = str(payload.get("window_title_contains", "")).strip()
    try:
        x, y, width, height = (float(region[key]) for key in ("x", "y", "width", "height"))
    except (TypeError, KeyError, ValueError):
        return jsonify({"error": "The crop selection is incomplete."}), 400
    if (len(title) > 120 or not all(map(math.isfinite, (x, y, width, height)))
            or min(x, y) < 0 or width <= 0 or height <= 0
            or x + width > 1 or y + height > 1):
        return jsonify({"error": "Select a valid area inside the server preview."}), 400
    try:
        config = load_config()
        if title:
            config["window_title_contains"] = title
        config["kill_region"] = {"x": round(x, 6), "y": round(y, 6),
                                 "width": round(width, 6), "height": round(height, 6)}
        temp_path = CONFIG_PATH.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_path, CONFIG_PATH)
    except Exception as exc:
        log.exception("Could not save browser calibration")
        return jsonify({"error": f"Could not save calibration: {exc}"}), 500
    return jsonify({"ok": True, "kill_region": config["kill_region"]})


@app.get("/api/crop.png")
def api_crop():
    with state.lock:
        crop_png = state.latest_crop_png
    if crop_png is None:
        return "No OCR crop has been captured yet.", 503
    return Response(crop_png, mimetype="image/png", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    start_worker()
    start_console_reload_watcher()
    if os.name == "nt" and sys.stdin.isatty():
        log.info("Press Ctrl+R to restart the server, or Ctrl+C to stop it")
    config = load_config()
    app.run(host=config.get("host", "127.0.0.1"), port=int(config.get("port", 8765)), threaded=True, use_reloader=False)
