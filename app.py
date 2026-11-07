from __future__ import annotations

import json
import io
import base64
import ipaddress
import logging
import math
import os
import re
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, stream_with_context
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
DEFAULT_WIDGET_HTML = '<span data-kills>—</span>'
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
DEFAULT_CUSTOM_COUNT_RULES = {
    "limit_increase": False,
    "ignore_decreases": False,
    "reset_on_zero": False,
}
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
    config.setdefault("lan_access_enabled", False)
    config.setdefault("lan_access_code", "")
    packaged_device = os.environ.get("FENIKZ_OCR_DEVICE")
    if packaged_device in {"cpu", "cuda"}:
        config["ai_device"] = packaged_device
    return config


def _area_slug(name: str) -> str:
    value = re.sub(r"[^\w]+", "-", name.strip().lower(), flags=re.UNICODE).strip("-_-")
    return value[:48].strip("-_-") or "area"


def configured_areas(config: dict[str, Any]) -> list[dict[str, Any]]:
    saved = config.get("capture_areas")
    if not isinstance(saved, list):
        return [{"slug": "kill", "name": "Kill", "region": config.get("kill_region")}]
    areas = []
    seen = set()
    for item in saved:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug", ""))
        name = str(item.get("name", "")).strip()
        if not re.fullmatch(r"[\w]+(?:-[\w]+)*", slug, flags=re.UNICODE) or not name or slug in seen:
            continue
        seen.add(slug)
        areas.append({**item, "slug": slug, "name": name[:60], "region": item.get("region")})
    kill = next((item for item in areas if item["slug"] == "kill"), None)
    if kill is None:
        areas.insert(0, {"slug": "kill", "name": "Kill", "region": config.get("kill_region")})
    elif kill.get("region") is None and config.get("kill_region") is not None:
        kill["region"] = config["kill_region"]
    return areas


def write_config(config: dict[str, Any]) -> None:
    temp_path = CONFIG_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp_path, CONFIG_PATH)


@dataclass
class AreaReading:
    kills: int | None = None
    pending_value: int | None = None
    pending_count: int = 0
    latest_raw: str = ""
    latest_candidate: int | None = None
    latest_candidate_note: str | None = None
    latest_confidence: float | None = None
    latest_ocr_support: int = 0
    latest_read_at: str | None = None
    latest_crop_png: bytes | None = None
    last_valid_candidate_at: float | None = None


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
    area_readings: dict[str, AreaReading] = field(default_factory=dict)
    update_condition: threading.Condition = field(default_factory=threading.Condition)
    update_version: int = 0

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


def notify_live_clients() -> None:
    with state.update_condition:
        state.update_version += 1
        state.update_condition.notify_all()


@app.before_request
def protect_lan_access():
    remote = request.remote_addr or "127.0.0.1"
    try:
        address = ipaddress.ip_address(remote)
    except ValueError:
        return jsonify({"error": "Access is limited to local devices."}), 403
    if address.is_loopback:
        return None
    config = load_config()
    if not config.get("lan_access_enabled", False) or not address.is_private:
        return jsonify({"error": "LAN access is disabled or this device is outside the local network."}), 403
    expected = str(config.get("lan_access_code", ""))
    authorization = request.authorization
    supplied = authorization.password if authorization and authorization.username == "fenikz" else ""
    if len(expected) < 8 or not secrets.compare_digest(supplied or "", expected):
        return Response("Local network access requires the access code.", 401,
                        {"WWW-Authenticate": 'Basic realm="FENIKZ OCR local access"'})
    return None


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
        reading = state.area_readings.setdefault("kill", AreaReading())
        _accept_reading(reading, candidate, config, support)
        state.kills = reading.kills
        state.pending_value, state.pending_count = reading.pending_value, reading.pending_count
        state.latest_candidate = reading.latest_candidate
        state.latest_candidate_note = reading.latest_candidate_note
        state.latest_confidence = reading.latest_confidence
        state.latest_ocr_support = reading.latest_ocr_support
        state.latest_raw = reading.latest_raw
        state.latest_read_at = reading.latest_read_at
        state.last_valid_candidate_at = reading.last_valid_candidate_at


def _accept_reading(reading: AreaReading, candidate: int | None,
                    config: dict[str, Any], support: int = 0,
                    count_rules: dict[str, Any] | None = None) -> None:
    reading.latest_candidate = candidate
    reading.latest_candidate_note = None
    if candidate is None:
        reading.pending_value, reading.pending_count = None, 0
        return
    max_increase = max(1, int(config.get("max_kill_increase_per_update", 2)))
    custom_rules = count_rules is not None
    limit_increase = (bool(count_rules.get("limit_increase", False))
                      if custom_rules else True)
    ignore_decreases = (bool(count_rules.get("ignore_decreases", False))
                        if custom_rules else True)
    reset_on_zero = (bool(count_rules.get("reset_on_zero", False))
                     if custom_rules else True)
    if (reading.kills is not None and candidate < reading.kills
            and ((candidate != 0 and ignore_decreases)
                 or (candidate == 0 and ignore_decreases and not reset_on_zero))):
        reading.pending_value, reading.pending_count = None, 0
        reading.latest_candidate_note = f"Ignored decrease {reading.kills}→{candidate}; only a confirmed zero resets a match."
        return
    if (limit_increase and reading.kills is not None
            and candidate > reading.kills + max_increase):
        reading.pending_value, reading.pending_count = None, 0
        reading.latest_candidate_note = f"Ignored jump {reading.kills}→{candidate}; maximum increase is {max_increase} per reading."
        return
    reading.last_valid_candidate_at = time.time()
    if candidate == reading.kills:
        reading.pending_value, reading.pending_count = None, 0
        return
    if candidate == reading.pending_value:
        reading.pending_count += 1
    else:
        reading.pending_value, reading.pending_count = candidate, 1
    if (reading.kills is not None and candidate == 0 and candidate < reading.kills
            and reset_on_zero):
        key = "stable_readings_to_accept_match_reset"
    elif support == 1:
        key = "stable_readings_to_accept_single_pass"
    else:
        key = "stable_readings_to_accept"
    if reading.pending_count >= max(1, int(config.get(key, 2))):
        reading.kills = candidate
        reading.pending_value, reading.pending_count = None, 0


def accept_named_candidate(area_slug: str, candidate: int | None,
                           config: dict[str, Any], support: int = 0,
                           count_rules: dict[str, Any] | None = None) -> None:
    with state.lock:
        reading = state.area_readings.setdefault(area_slug, AreaReading())
        _accept_reading(reading, candidate, config, support,
                        count_rules if count_rules is not None else DEFAULT_CUSTOM_COUNT_RULES)


def area_reading_snapshot(reading: AreaReading, stale_after: float) -> dict[str, Any]:
    fresh = (reading.last_valid_candidate_at is not None and
             time.time() - reading.last_valid_candidate_at <= stale_after)
    return {"kills": reading.kills if fresh else None,
            "last_confirmed_kills": reading.kills, "reading_fresh": fresh,
            "latest_raw": reading.latest_raw, "latest_candidate": reading.latest_candidate,
            "latest_candidate_note": reading.latest_candidate_note,
            "latest_confidence": reading.latest_confidence,
            "latest_ocr_support": reading.latest_ocr_support,
            "latest_read_at": reading.latest_read_at}


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
        log.warning("CUDA was requested but CUDAExecutionProvider is not registered. Available ONNX Runtime providers: %s",
                    ", ".join(ort.get_available_providers()))
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
                sessions.append((stage_name, inference_session.get_providers()))
            except Exception:
                continue
    if cuda_requested and sessions and all("CUDAExecutionProvider" in providers for _, providers in sessions):
        return "CUDA (NVIDIA GPU)"
    if cuda_requested:
        details = "; ".join(f"{name}={','.join(providers)}" for name, providers in sessions)
        return f"CPU fallback (CUDA provider inactive: {details or 'sessions not initialized'})"
    return "CPU"


def recognize_crop(crop_image, config: dict[str, Any], engine_name: str,
                   ai_engine, ai_device_active: str):
    if engine_name == "rapidocr":
        if ai_engine is None:
            ai_engine, ai_device_active = create_ai_engine(config)
            with state.lock:
                state.recognition_device = ai_device_active
            log.info("RapidOCR is configured for %s", ai_device_active)
        try:
            raw, candidate, confidence, support = read_number_ai(
                crop_image, ai_engine, float(config.get("ai_minimum_confidence", 0.60)))
        except Exception:
            if not ai_device_active.startswith("CUDA"):
                raise
            log.exception("CUDA recognition failed; switching this run to CPU")
            ai_engine, ai_device_active = create_ai_engine(config, allow_cuda=False)
            with state.lock:
                state.recognition_device = "CPU fallback (CUDA failed)"
            raw, candidate, confidence, support = read_number_ai(
                crop_image, ai_engine, float(config.get("ai_minimum_confidence", 0.60)))
        if str(config.get("ai_device", "cpu")).lower() == "cuda":
            confirmed_device = actual_ai_device(ai_engine, cuda_requested=True)
            if confirmed_device != ai_device_active:
                ai_device_active = confirmed_device
                with state.lock:
                    state.recognition_device = confirmed_device
                if confirmed_device.startswith("CPU fallback"):
                    log.warning("CUDA was requested, but RapidOCR provider status is %s", confirmed_device)
        return raw, candidate, confidence, support, ai_engine, ai_device_active
    if engine_name == "tesseract":
        import pytesseract
        if config.get("tesseract_cmd"):
            pytesseract.pytesseract.tesseract_cmd = config["tesseract_cmd"]
        raw, candidate, confidence, support = read_number(
            crop_image,
            float(config.get("minimum_ocr_confidence", 15)),
            float(config.get("single_pass_minimum_confidence", 45)),
        )
        return raw, candidate, confidence, support, ai_engine, ai_device_active
    raise RuntimeError(f"Unknown recognition_engine: {engine_name}")


def ocr_loop() -> None:
    with state.lock:
        state.running = True
    notify_live_clients()
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
                    notify_live_clients()
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
            engine_name = str(config.get("recognition_engine", "rapidocr")).lower()
            for area in configured_areas(config):
                area_slug = area["slug"]
                try:
                    crop_image = capture_crop(source_image, area.get("region"))
                except Exception as exc:
                    note = "This area has not been calibrated yet." if area.get("region") is None else str(exc)
                    with state.lock:
                        reading = state.area_readings.setdefault(area_slug, AreaReading())
                        reading.latest_candidate = None
                        reading.latest_candidate_note = note
                        if area_slug == "kill":
                            state.latest_candidate = None
                            state.latest_candidate_note = note
                            state.latest_error = note
                            state.window_title = title_hint
                    continue
                crop_buffer = io.BytesIO()
                crop_image.save(crop_buffer, format="PNG")
                crop_png = crop_buffer.getvalue()
                with state.lock:
                    state.area_readings.setdefault(area_slug, AreaReading()).latest_crop_png = crop_png
                    if area_slug == "kill":
                        state.latest_crop_png = crop_png
                try:
                    raw, candidate, confidence, support, ai_engine, ai_device_active = recognize_crop(
                        crop_image, config, engine_name, ai_engine, ai_device_active)
                except Exception as exc:
                    with state.lock:
                        reading = state.area_readings.setdefault(area_slug, AreaReading())
                        reading.latest_candidate = None
                        reading.latest_candidate_note = f"OCR error: {exc}"
                        if area_slug == "kill":
                            state.latest_candidate = None
                            state.latest_candidate_note = reading.latest_candidate_note
                            state.latest_error = reading.latest_candidate_note
                    continue
                if area_slug == "kill":
                    accept_candidate(candidate, config, support)
                else:
                    accept_named_candidate(area_slug, candidate, config, support,
                                           area.get("count_rules", DEFAULT_CUSTOM_COUNT_RULES))
                read_at = datetime.now(timezone.utc).isoformat()
                with state.lock:
                    reading = state.area_readings.setdefault(area_slug, AreaReading())
                    reading.latest_raw = raw
                    reading.latest_confidence = confidence
                    reading.latest_ocr_support = support
                    reading.latest_read_at = read_at
                    if area_slug == "kill":
                        state.waiting_for_window = False
                        state.latest_raw, state.latest_confidence = raw, confidence
                        state.latest_ocr_support = support
                        state.latest_error = None
                        state.latest_read_at = read_at
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
            notify_live_clients()
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
        notify_live_clients()
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
    notify_live_clients()


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


@app.get("/")
def control_page():
    return """<!doctype html><html><head><meta charset='utf-8'><title>FENIKZ OCR Control</title>
<style>:root{color-scheme:dark;--bg:#080d18;--card:#0e1626;--line:#202c40;--text:#edf4ff;--muted:#91a4c0;--cyan:#17b9ef;--green:#21d6a0}*{box-sizing:border-box}body{font:15px/1.5 system-ui;background:var(--bg);color:var(--text);margin:0 auto;padding:26px 24px;max-width:1050px}header{display:flex;align-items:center;justify-content:space-between;gap:24px;padding-bottom:20px;border-bottom:1px solid var(--line)}h1{font-size:26px;margin:8px 0}.eyebrow,a{color:#28bff3}a{text-decoration:none}a:hover{text-decoration:underline}.back{font-size:14px;font-weight:650}.badge{display:inline-flex;align-items:center;gap:8px;background:#0c2729;border:1px solid #174743;color:var(--green);border-radius:99px;padding:4px 12px;font-weight:650}.badge:before{content:'';width:8px;height:8px;background:currentColor;border-radius:50%}.stats{display:flex;gap:20px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 16px;color:var(--muted)}.stats b{display:block;color:var(--text)}main{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(280px,.8fr);gap:16px;margin-top:24px}article{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}article:first-child{grid-row:span 2}strong{font-size:56px;line-height:1.1}small,.muted{color:var(--muted)}button,.button{display:inline-flex;align-items:center;justify-content:center;padding:10px 14px;border:1px solid #31415c;border-radius:9px;background:#1a2639;color:var(--text);font:inherit;cursor:pointer;text-decoration:none}button:hover,.button:hover{border-color:var(--cyan);background:#20324a}.primary{background:#078cc2;border-color:#078cc2;color:white}#service{color:var(--muted)}#configMessage{color:var(--green);min-height:1.3em}#configPath{overflow-wrap:anywhere}@media(max-width:720px){body{padding:18px 14px}header{align-items:flex-start;flex-direction:column}.stats{width:100%;justify-content:space-between}main{grid-template-columns:1fr}article:first-child{grid-row:auto}}</style></head>
<body><header><div><a class='back' href='/'>◆ FENIKZ OCR</a><h1>Control Center</h1><span class='badge' id='serviceBadge'>Starting service</span></div><div class='stats'><span>Confirmed kills<b id='headerKills'>—</b></span><span>Capture source<b id='headerSource'>MobileStream</b></span></div></header><main><article><small>CONFIRMED KILLS</small><br><strong id='kills'>—</strong><p id='service'></p><button class='primary' onclick='resetCount()'>Reset match count</button></article>
<article><b>Capture Areas, Calibration and Customization</b><p class='muted'>Add, calibrate, or recapture named regions, and customize each area's counter widget.</p><a class='button primary' href='/areas'>Open capture areas</a></article>
<article><b>OBS number widget</b><p class='muted'>Use this transparent page as a Browser Source in OBS.</p><a class='button' href='/widget/kill' target='_blank'>Open Kill widget</a></article>
<article><b>Configuration</b><p class='muted'>Export a backup or import a configuration file.</p><button onclick="location.href='/api/config/export'">Export configuration</button> <label class='button' style="cursor:pointer"><input id="configImport" type="file" accept=".json,application/json" hidden>Import configuration</label><p id="configMessage" role="status"></p><small id="configPath"></small></article>
<article><b>Phone access on local Wi-Fi</b><p class='muted'>Optional. Enable this to open the control page from a phone on your private Wi-Fi. Restart the app after saving. The browser will ask for username <b>fenikz</b> and this access code.</p><label><input id='lanEnabled' type='checkbox'> Allow access from local network</label><p><input id='lanCode' type='text' minlength='8' maxlength='128' autocomplete='off' placeholder='Access code (8+ characters)' style='width:min(100%,420px);padding:9px;background:#080d18;color:var(--text);border:1px solid #263650;border-radius:8px'></p><button id='saveLan' class='primary'>Save phone access</button><p id='lanMessage' role='status'></p><small>Use only on a trusted, password-protected Wi-Fi network. Local HTTP does not encrypt the access code.</small></article></main>
<script>function renderStatus(s){const count=s.kills??'—',text=s.waiting_for_window?`Waiting for ${s.window_title||'MobileStream'}…`:!s.running?'Starting recognition service':(!s.reading_fresh&&s.last_confirmed_kills!==null?`No reliable reading · last confirmed ${s.last_confirmed_kills}`:`Recognition running on ${s.recognition_device||'CPU'}`);document.querySelector('#kills').textContent=count;document.querySelector('#headerKills').textContent=count;document.querySelector('#headerSource').textContent=s.window_title||'MobileStream';document.querySelector('#service').textContent=text;document.querySelector('#serviceBadge').textContent=s.running?'OCR Service Active':'Service Starting'}const events=new EventSource('/api/events');events.onmessage=e=>{try{renderStatus(JSON.parse(e.data))}catch{}};events.onerror=()=>{document.querySelector('#service').textContent='Reconnecting to the local service…'};document.querySelector('#configImport').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;const note=document.querySelector('#configMessage');note.textContent='Importing…';try{const response=await fetch('/api/config/import',{method:'PUT',headers:{'Content-Type':'application/json'},body:await file.text()});const data=await response.json();if(!response.ok)throw new Error(data.error||'Could not import configuration. No settings were applied.');note.textContent='Configuration imported. Restart the service to apply all settings. '+(data.notice||'')}catch(error){note.textContent=error.message||String(error)}finally{event.target.value=''}});fetch('/api/config/location').then(r=>r.json()).then(data=>document.querySelector('#configPath').textContent='Configuration file: '+data.path).catch(()=>{});fetch('/api/lan-access').then(r=>r.json()).then(data=>{document.querySelector('#lanEnabled').checked=data.enabled;document.querySelector('#lanCode').value=data.access_code||''}).catch(()=>{});document.querySelector('#saveLan').onclick=async()=>{const message=document.querySelector('#lanMessage');message.textContent='Saving…';try{const response=await fetch('/api/lan-access',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:document.querySelector('#lanEnabled').checked,access_code:document.querySelector('#lanCode').value})}),data=await response.json();if(!response.ok)throw new Error(data.error||'Could not save.');document.querySelector('#lanCode').value=data.access_code;message.textContent=data.enabled?'Saved. Restart the app, then use http://<this-PC-IP>:8765/ and sign in as fenikz with this code.':'Phone access disabled. Restart the app to keep the service local.'}catch(error){message.textContent=error.message||String(error)}};async function resetCount(){await fetch('/api/reset',{method:'POST'})}</script></body></html>"""


@app.get("/areas")
def capture_areas_page():
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Capture areas</title>
<style>:root{color-scheme:dark;--bg:#080d18;--card:#0e1626;--line:#202c40;--text:#edf4ff;--muted:#91a4c0;--cyan:#17b9ef;--green:#21d6a0}*{box-sizing:border-box}body{font:14px/1.5 system-ui;background:var(--bg);color:var(--text);margin:0;padding:24px 18px;max-width:none}header{display:flex;align-items:center;justify-content:space-between;gap:20px;padding-bottom:16px;border-bottom:1px solid var(--line)}h1{font-size:22px;margin:0}.back{color:var(--cyan);font-weight:650;text-decoration:none}.intro,.addbar{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:16px 0;padding:14px 16px;color:#afbdd2}.addbar{display:flex;align-items:center;gap:14px}.add-form{display:flex;align-items:center;gap:10px;flex:1}.add-form label{font-weight:650;white-space:nowrap}.add-form input{flex:1;min-width:120px;background:#080d18;color:var(--text);border:1px solid #263650;border-radius:8px;padding:9px 13px;font:inherit}.total{white-space:nowrap;color:var(--muted)}.total b{color:var(--text)}button{font:inherit;padding:8px 11px;border:1px solid #31415c;border-radius:7px;background:#1a2639;color:var(--text);cursor:pointer;white-space:nowrap}button:hover{border-color:var(--cyan);background:#20324a}.primary{background:#078cc2;border-color:#078cc2;color:white}a{color:#29baf0;overflow-wrap:anywhere}.message{min-height:1.3em;margin:7px 2px;color:var(--muted)}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:11px;background:var(--card)}table{width:100%;min-width:1080px;border-collapse:collapse;table-layout:fixed}th{background:#080d18;color:#8fa4c0;font-size:12px;text-transform:uppercase;text-align:left;padding:12px 14px;border-bottom:1px solid var(--line)}td{padding:12px 14px;border-bottom:1px solid #1b2739;vertical-align:middle}tr:last-child td{border-bottom:0}.col-name{width:150px}.col-count{width:82px}.col-confidence{width:120px}.col-url{width:290px}.col-crop{width:130px}.col-actions{width:auto;text-align:right}.area-name{font-weight:700}.area-default{font-size:11px;margin-left:6px;padding:2px 7px;background:#1a2639;border:1px solid #31415c;border-radius:5px;color:#b4c5dc}.area-state{display:block;color:var(--green);font-size:12px}.area-state.missing{color:#ffc47a}.area-count{font-weight:700}.area-confidence{color:#afbdd2;font-style:italic}.area-link{display:inline-block;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.area-crop{display:block;image-rendering:pixelated;width:80px;height:55px;object-fit:contain;background:#050912;border:1px solid #1f2b3e;border-radius:5px;padding:3px}.area-crop[hidden]{display:none}.area-actions{display:flex;justify-content:flex-end;gap:6px;flex-wrap:wrap}.area-actions button{font-size:12px;padding:7px 9px}.message-row{color:var(--muted)}@media(max-width:650px){body{padding:16px 10px}header{align-items:flex-start;gap:8px;flex-direction:column}.addbar{align-items:stretch;flex-direction:column}.add-form{flex-wrap:wrap}.add-form label{width:100%}.add-form input{flex-basis:100%}.total{align-self:flex-end}}</style></head><body>
<header><h1>Capture Areas, Calibration and Customization</h1><a class='back' href='/'>← Back to control page</a></header>
<p class='intro'>Each area reads a separate rectangle from the captured window with independent counts, AI confidence, crop preview, widget design, and OBS URL.</p>
<section class='addbar'><form class='add-form' id='addForm'><label for='newName'>Add Area:</label><input id='newName' maxlength='60' placeholder='Name, for example Headshots'><button class='primary' type='submit'>Add</button></form><span class='total'>Total: <b id='totalAreas'>—</b></span></section><p id='message' role='status' class='message'></p>
<div class='table-wrap'><table><thead><tr><th class='col-name'>Area name</th><th class='col-count'>Count</th><th class='col-confidence'>AI confidence</th><th class='col-url'>OBS widget URL</th><th class='col-crop'>Latest crop</th><th class='col-actions'>Actions</th></tr></thead><tbody id='areas'></tbody></table></div><p><a href='/'>Back to control page</a></p>
<script>
const list=document.querySelector('#areas'),message=document.querySelector('#message'),cards=new Map();
function makeButton(label,action){const b=document.createElement('button');b.textContent=label;b.onclick=action;return b}
async function request(url,method='GET',body){const response=await fetch(url,{method,headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});const data=await response.json();if(!response.ok)throw new Error(data.error||'Request failed.');return data}
function cell(className){const td=document.createElement('td');if(className)td.className=className;return td}
function loadCrop(view){if(!view.latestReadAt)return;view.crop.dataset.loadedAt=view.latestReadAt;view.crop.src='/api/crop/'+encodeURIComponent(view.slug)+'.png?read='+encodeURIComponent(view.latestReadAt);view.crop.hidden=false;view.cropNote.hidden=true}
function hideCrop(view){clearTimeout(view.hideTimer);view.cropVisible=false;view.crop.hidden=true;view.crop.removeAttribute('src');view.crop.dataset.loadedAt='';view.cropNote.textContent=view.latestReadAt?'Screenshot hidden':'Waiting for the first OCR crop';view.cropNote.hidden=false;view.cropToggle.textContent='👁 Show';view.cropToggle.setAttribute('aria-label','Show latest OCR crop for five seconds')}
function showCrop(view){if(!view.latestReadAt)return;view.cropVisible=true;view.cropToggle.textContent='◉ Hide';view.cropToggle.setAttribute('aria-label','Hide OCR crop');if(view.crop.dataset.loadedAt!==view.latestReadAt)loadCrop(view);else view.crop.hidden=false;view.cropNote.hidden=true;clearTimeout(view.hideTimer);view.hideTimer=setTimeout(()=>hideCrop(view),5000)}
function createCard(area){const row=document.createElement('tr'),nameCell=cell('col-name'),title=document.createElement('span'),defaultTag=document.createElement('span'),state=document.createElement('span'),countCell=cell('col-count'),count=document.createElement('b'),confidenceCell=cell('col-confidence'),confidence=document.createElement('span'),urlCell=cell('col-url'),link=document.createElement('a'),cropCell=cell('col-crop'),cropNote=document.createElement('small'),cropToggle=document.createElement('button'),crop=document.createElement('img'),actionsCell=cell('col-actions'),actions=document.createElement('div');title.className='area-name';defaultTag.className='area-default';defaultTag.textContent='Default';state.className='area-state';count.className='area-count';confidence.className='area-confidence';link.className='area-link';link.target='_blank';cropNote.className='muted';cropNote.textContent='Waiting for the first OCR crop';cropToggle.className='crop-toggle';cropToggle.type='button';cropToggle.textContent='👁 Show';cropToggle.title='Show the latest crop for five seconds';cropToggle.setAttribute('aria-label','Show latest OCR crop for five seconds');cropToggle.disabled=true;crop.className='area-crop';crop.alt='Latest OCR crop sent to OCR';crop.hidden=true;actions.className='area-actions';nameCell.append(title,defaultTag,state);countCell.append(count);confidenceCell.append(confidence);urlCell.append(link);cropCell.append(cropToggle,cropNote,crop);actions.append(makeButton('Calibrate',()=>location.href='/calibrate?area='+encodeURIComponent(area.slug)));actions.append(makeButton('Customize widget',()=>location.href='/customize/'+encodeURIComponent(area.slug)));actions.append(makeButton('Reset count',async()=>{try{await request('/api/areas/'+encodeURIComponent(area.slug)+'/reset','POST');await refresh()}catch(e){message.textContent=e.message}}));if(area.slug!=='kill'){actions.append(makeButton('Rename',async()=>{const name=prompt('New name for this capture area:',area.name);if(name===null)return;try{await request('/api/areas/'+encodeURIComponent(area.slug),'PUT',{name});await refresh()}catch(e){message.textContent=e.message}}));actions.append(makeButton('Remove',async()=>{if(!confirm('Remove '+area.name+' and its capture area?'))return;try{await request('/api/areas/'+encodeURIComponent(area.slug),'DELETE');await refresh()}catch(e){message.textContent=e.message}}))}actionsCell.append(actions);const view={card:row,title,defaultTag,state,count,confidence,link,crop,cropNote,cropToggle,slug:area.slug,latestReadAt:null,cropVisible:false,hideTimer:null};cropToggle.onclick=()=>view.cropVisible?hideCrop(view):showCrop(view);row.append(nameCell,countCell,confidenceCell,urlCell,cropCell,actionsCell);return view}
function renderAreas(areas){const active=new Set();document.querySelector('#totalAreas').textContent=areas.length;for(const area of areas){active.add(area.slug);let view=cards.get(area.slug);if(!view){view=createCard(area);cards.set(area.slug,view)}view.title.textContent=area.name;view.defaultTag.hidden=area.slug!=='kill';view.state.textContent=area.region?'Crop saved':'Calibration needed';view.state.classList.toggle('missing',!area.region);view.link.href='/widget/'+encodeURIComponent(area.slug);view.link.textContent=location.origin+view.link.pathname;view.count.textContent=area.last_confirmed_kills??'—';view.confidence.textContent=area.latest_confidence==null?'unavailable':`${area.latest_confidence.toFixed(0)}%`;view.latestReadAt=area.latest_read_at||null;view.cropToggle.disabled=!view.latestReadAt;view.cropNote.textContent=view.latestReadAt?'Screenshot hidden':'Waiting for the first OCR crop';if(view.cropVisible&&view.latestReadAt&&view.crop.dataset.loadedAt!==view.latestReadAt)loadCrop(view);if(view.card.parentElement!==list)list.append(view.card)}for(const [slug,view] of cards){if(!active.has(slug)){hideCrop(view);view.card.remove();cards.delete(slug)}}}
async function refresh(){try{const data=await request('/api/areas');renderAreas(data.areas)}catch(e){message.textContent=e.message}}
document.querySelector('#addForm').addEventListener('submit',async event=>{event.preventDefault();const input=document.querySelector('#newName');try{await request('/api/areas','POST',{name:input.value});input.value='';message.textContent='Area added. Calibrate it, then add its widget URL to OBS.';await refresh()}catch(e){message.textContent=e.message}});const areaEvents=new EventSource('/api/events');areaEvents.onmessage=event=>{try{const status=JSON.parse(event.data);renderAreas(Object.values(status.areas||{}))}catch{}};areaEvents.onerror=()=>{message.textContent='Reconnecting to live counter updates…'};refresh();
</script></body></html>"""


@app.get("/calibrate")
def browser_calibration_page():
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Calibrate capture area</title>
<style>:root{color-scheme:dark;--bg:#080d18;--card:#0e1626;--line:#202c40;--text:#edf4ff;--muted:#91a4c0;--cyan:#17b9ef;--green:#21d6a0}*{box-sizing:border-box}body{font:14px/1.5 system-ui;background:var(--bg);color:var(--text);margin:0 auto;padding:24px 24px;max-width:1200px}header{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;padding-bottom:16px;border-bottom:1px solid var(--line)}h1{font-size:24px;margin:7px 0}.back{color:var(--cyan);text-decoration:none;font-weight:650}.cal-subtitle,.muted{color:var(--muted)}article,.settings-panel{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:16px}.cal-panel{margin:18px 0}.cal-intro{margin:0 0 14px;color:#afbdd2}.cal-settings{display:grid;grid-template-columns:minmax(280px,1fr) minmax(280px,1fr);gap:12px;margin-bottom:14px}.settings-panel h2{font-size:15px;margin:0 0 8px}.settings-panel p{color:var(--muted);margin:5px 0 10px}.window-entry{display:flex;gap:8px;flex-wrap:wrap}.window-entry input{flex:1;min-width:200px}.count-rules label{display:block;margin:7px 0;color:#c7d5e8}button,input{font:inherit;padding:9px 12px;border-radius:8px}button{border:1px solid #31415c;background:#1a2639;color:var(--text);cursor:pointer}button:hover{border-color:var(--cyan)}button.primary{background:#078cc2;border-color:#078cc2;color:#fff}input:not([type=checkbox]){background:#080d18;color:var(--text);border:1px solid #263650}input[type=checkbox]{vertical-align:middle;accent-color:var(--cyan)}#windowStatus{color:var(--green)}.preview-panel{border:1px solid var(--line);border-radius:12px;background:#0a111e;padding:12px}.preview-head{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:10px}.preview-head p{margin:2px 0;color:var(--muted)}.preview-actions{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.zoom-control{display:flex;align-items:center;gap:8px;color:var(--muted)}.zoom-control input{width:130px;padding:0;border:0;background:transparent;accent-color:var(--cyan)}#stage{position:relative;background:#03070d;min-height:180px;height:58vh;display:block;overflow:auto;border:1px solid #263650;border-radius:9px}#selectCanvas{display:block;margin:0 auto;cursor:crosshair;touch-action:none}.cal-footer{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-top:12px}.cal-footer p{margin:0;color:var(--muted);min-height:1.5em}#status.error{color:#ffb078}a{color:#29baf0}@media(max-width:700px){body{padding:16px 12px}header{flex-direction:column;gap:6px}.cal-settings{grid-template-columns:1fr}.preview-head,.cal-footer{align-items:flex-start;flex-direction:column}.preview-actions{flex-wrap:wrap}.zoom-control input{width:90px}#stage{height:50vh}}</style></head><body>
<header><div><a class='back' href='/areas'>← Back to Capture Areas</a><h1>Calibrate: <span id='areaName'>Kill</span></h1><p class='cal-subtitle'>Select the exact number region from the captured stream.</p></div><span class='muted'>MobileStream window capture</span></header>
<article class='cal-panel'><p class='cal-intro'>Choose the capture window, set optional safeguards for this custom counter, then drag a tight rectangle around only its number.</p>
<div class='cal-settings'><section class='settings-panel'><h2>Capture window</h2><p>The default is <b>MobileStream</b>. Enter any distinctive part of the target window title.</p><div class='window-entry'><input id='windowTitleInput' maxlength='120' value='MobileStream'><button id='saveWindow'>Save window</button></div><p id='windowStatus' role='status'></p></section>
<section id='countRules' class='settings-panel count-rules' hidden><h2>Count safeguards</h2><p>Optional protections for this custom counter. All are off by default; Kill keeps its current safeguards.</p><label><input type='checkbox' id='limitIncrease'> Limit increases to the configured maximum per reading</label><label><input type='checkbox' id='ignoreDecreases'> Ignore decreases</label><label><input type='checkbox' id='resetOnZero'> Require repeated confirmation before resetting to zero</label><p id='rulesStatus' role='status'></p></section></div>
<section class='preview-panel'><div class='preview-head'><div><b>Captured stream screenshot</b><p>Capture one still image, then drag around the number. Retake it only if the source changes.</p></div><div class='preview-actions'><button class='primary' id='startPreview'>Capture screenshot</button><button id='refreshPreview' disabled>Retake screenshot</button><label class='zoom-control' for='zoomRange'>Zoom <input id='zoomRange' type='range' min='1' max='4' step='0.25' value='1'><span id='zoomValue'>100%</span></label></div></div><div id='stage'><canvas id='selectCanvas'></canvas></div><div class='cal-footer'><p id='status'>Capture a screenshot to begin calibration.</p><button class='primary' id='saveRegion' disabled>Save calibration</button></div></section></article>
<script>
const canvas=document.querySelector('#selectCanvas'), ctx=canvas.getContext('2d'), stage=document.querySelector('#stage'), frame=new Image();
const areaSlug=new URLSearchParams(location.search).get('area')||'kill';
let selection=null, dragOrigin=null, previewActive=false, zoom=1;
function say(text,error=false){const e=document.querySelector('#status');e.textContent=text;e.classList.toggle('error',error)}
async function loadConfig(){try{const config=await(await fetch('/api/config')).json();document.querySelector('#windowTitleInput').value=config.window_title_contains||'MobileStream';const data=await(await fetch('/api/areas')).json();const area=data.areas.find(item=>item.slug===areaSlug);if(!area){say('Capture area not found.',true);return}document.querySelector('#areaName').textContent=area.name;if(areaSlug!=='kill'){document.querySelector('#countRules').hidden=false;const rules=area.count_rules||{};document.querySelector('#limitIncrease').checked=!!rules.limit_increase;document.querySelector('#ignoreDecreases').checked=!!rules.ignore_decreases;document.querySelector('#resetOnZero').checked=!!rules.reset_on_zero}}catch{}}
async function saveCountRules(){const status=document.querySelector('#rulesStatus'),count_rules={limit_increase:document.querySelector('#limitIncrease').checked,ignore_decreases:document.querySelector('#ignoreDecreases').checked,reset_on_zero:document.querySelector('#resetOnZero').checked};status.textContent='Saving…';try{const r=await fetch('/api/areas/'+encodeURIComponent(areaSlug)+'/count-rules',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(count_rules)}),data=await r.json();if(!r.ok)throw new Error(data.error||'Could not save count safeguards.');status.textContent='Saved.'}catch(e){status.textContent=e.message||String(e)}}
function draw(){if(!frame.complete||!frame.naturalWidth)return;const w=frame.naturalWidth,h=frame.naturalHeight;if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}const fitScale=Math.min(stage.clientWidth/w,stage.clientHeight/h),scale=fitScale*zoom;canvas.style.width=`${w*scale}px`;canvas.style.height=`${h*scale}px`;ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);ctx.drawImage(frame,0,0,w,h);if(selection){const x=Math.min(selection.x1,selection.x2),y=Math.min(selection.y1,selection.y2),rw=Math.abs(selection.x2-selection.x1),rh=Math.abs(selection.y2-selection.y1);ctx.fillStyle='rgba(0,0,0,.48)';ctx.fillRect(0,0,w,h);ctx.clearRect(x,y,rw,rh);ctx.drawImage(frame,x,y,rw,rh,x,y,rw,rh);ctx.strokeStyle='#ff365d';ctx.lineWidth=Math.max(2,w/900);ctx.strokeRect(x,y,rw,rh)}}
function point(e){const r=canvas.getBoundingClientRect();return{x:Math.max(0,Math.min(canvas.width,(e.clientX-r.left)*canvas.width/r.width)),y:Math.max(0,Math.min(canvas.height,(e.clientY-r.top)*canvas.height/r.height))}}
async function captureStill(){const captureButton=document.querySelector('#startPreview'),retakeButton=document.querySelector('#refreshPreview');captureButton.disabled=true;retakeButton.disabled=true;selection=null;document.querySelector('#saveRegion').disabled=true;previewActive=false;say('Capturing one screenshot…');try{const started=await fetch('/api/calibration-preview/start',{method:'POST'});if(!started.ok)throw new Error('Could not request a screenshot.');let response=null;for(let attempt=0;attempt<24;attempt++){response=await fetch('/api/source.jpg?t='+Date.now());if(response.ok)break;if(response.status!==503)throw new Error(await response.text()||'Could not capture the source.');await new Promise(resolve=>setTimeout(resolve,250))}if(!response||!response.ok)throw new Error(await response.text()||'No screenshot arrived. Try again.');const blob=await response.blob(),url=URL.createObjectURL(blob);await new Promise((resolve,reject)=>{frame.onload=()=>{URL.revokeObjectURL(url);draw();resolve()};frame.onerror=()=>{URL.revokeObjectURL(url);reject(new Error('Could not display the captured screenshot.'))};frame.src=url});previewActive=true;retakeButton.disabled=false;say('Still screenshot ready. Zoom as needed, then drag around the number.')}catch(e){say(e.message||String(e),true)}finally{await fetch('/api/calibration-preview/stop',{method:'POST'}).catch(()=>{});captureButton.disabled=false;retakeButton.disabled=!frame.naturalWidth}}
canvas.addEventListener('pointerdown',e=>{if(!previewActive||!frame.naturalWidth)return;canvas.setPointerCapture(e.pointerId);dragOrigin=point(e);selection={x1:dragOrigin.x,y1:dragOrigin.y,x2:dragOrigin.x,y2:dragOrigin.y};draw()});
canvas.addEventListener('pointermove',e=>{if(!dragOrigin)return;const p=point(e);selection.x2=p.x;selection.y2=p.y;document.querySelector('#saveRegion').disabled=Math.abs(selection.x2-selection.x1)<5||Math.abs(selection.y2-selection.y1)<5;draw()});
canvas.addEventListener('pointerup',()=>{dragOrigin=null});
addEventListener('resize',draw);
document.querySelector('#startPreview').onclick=captureStill;document.querySelector('#refreshPreview').onclick=captureStill;
document.querySelector('#zoomRange').addEventListener('input',event=>{zoom=Number(event.target.value);document.querySelector('#zoomValue').textContent=Math.round(zoom*100)+'%';draw()});
for(const id of ['limitIncrease','ignoreDecreases','resetOnZero'])document.querySelector('#'+id).addEventListener('change',saveCountRules);
document.querySelector('#saveWindow').onclick=async()=>{const title=document.querySelector('#windowTitleInput').value.trim(),status=document.querySelector('#windowStatus');status.textContent='';if(!title){status.textContent='Enter part of the window title.';return}try{const r=await fetch('/api/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({window_title_contains:title})});const data=await r.json();if(!r.ok)throw new Error(data.error||'Could not save the window title.');status.textContent=`Saved. Waiting for a window containing “${data.window_title_contains}”; capture switches automatically when it appears.`}catch(e){status.textContent=e.message||String(e)}};
document.querySelector('#saveRegion').onclick=async()=>{if(!selection||!frame.naturalWidth)return;const x=Math.min(selection.x1,selection.x2),y=Math.min(selection.y1,selection.y2),w=Math.abs(selection.x2-selection.x1),h=Math.abs(selection.y2-selection.y1);try{const r=await fetch('/api/calibration',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({area_slug:areaSlug,kill_region:{x:x/frame.naturalWidth,y:y/frame.naturalHeight,width:w/frame.naturalWidth,height:h/frame.naturalHeight}})});const data=await r.json();if(!r.ok)throw new Error(data.error||'Could not save calibration.');document.querySelector('#saveRegion').disabled=true;previewActive=false;say('Calibration saved for '+document.querySelector('#areaName').textContent+'.')}catch(e){say(e.message||String(e),true)}};
addEventListener('pagehide',()=>navigator.sendBeacon('/api/calibration-preview/stop'));
loadConfig();
</script></body></html>"""


@app.get("/customize")
@app.get("/customize/<area_slug>")
def customize_widget_page(area_slug: str = "kill"):
    config = load_config()
    area = next((item for item in configured_areas(config) if item["slug"] == area_slug), None)
    if area is None:
        return "Capture area not found.", 404
    area_name = json.dumps(area["name"]).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    area_slug_json = json.dumps(area_slug)
    page = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Customize OBS widget</title>
<style>body{font:16px system-ui;background:#10151d;color:#eef3fa;margin:24px auto;padding:0 16px;max-width:1000px}article{background:#1a2330;padding:18px;border-radius:12px;margin:14px 0}button{font:inherit;padding:9px 13px;border:0;border-radius:7px;cursor:pointer;margin:4px}textarea{display:block;width:100%;min-height:130px;box-sizing:border-box;background:#0d1219;color:#eef3fa;border:1px solid #465365;border-radius:7px;padding:12px;font:14px/1.45 Consolas,monospace;resize:vertical}label{display:block;font-weight:700;margin:14px 0 7px}small,#message{color:#b7c3d2}a{color:#9dc8ff}iframe{width:100%;height:180px;border:1px solid #465365;border-radius:8px;background:repeating-conic-gradient(#202938 0 25%,#151b25 0 50%) 50%/24px 24px}</style></head><body>
<h1>Customize <span id='areaName'></span> widget</h1><p>Edit this capture area's HTML, CSS, and JavaScript. Changes are saved only when you select <b>Save and preview</b>; other capture areas keep their own designs.</p>
<article><b>Live preview</b><p><iframe id='preview' title='Widget preview'></iframe></p><button id='save'>Save and preview</button><button id='defaults'>Restore defaults</button><span id='message' role='status'></span></article>
<article><label for='html'>HTML</label><small>Use <code>{{kills}}</code> or an element with <code>data-kills</code> where the number should appear.</small><textarea id='html' spellcheck='false'></textarea>
<label for='css'>CSS</label><textarea id='css' spellcheck='false'></textarea>
<label for='js'>JavaScript</label><small>Use <code>window.killWidget.onChange(count =&gt; { ... })</code> to run an animation when the confirmed count changes. The latest value is in <code>window.killWidget.count</code>.</small><textarea id='js' spellcheck='false'></textarea>
<p><button id='saveBottom'>Save and preview</button><span id='bottomMessage' role='status'></span></p>
<p><a href='/'>Control page</a> · <a id='widgetLink' target='_blank'>Open widget URL</a> · <a href='/areas'>Capture areas</a></p></article>
<script>
const fields={html:document.querySelector('#html'),css:document.querySelector('#css'),js:document.querySelector('#js')};let defaults={};const areaSlug=__AREA_SLUG__,settingsUrl='/api/areas/'+encodeURIComponent(areaSlug)+'/widget-settings',widgetUrl='/widget/'+encodeURIComponent(areaSlug);document.querySelector('#areaName').textContent=__AREA_NAME__;document.querySelector('#preview').src=widgetUrl;document.querySelector('#widgetLink').href=widgetUrl;
async function load(){try{const r=await fetch(settingsUrl);const data=await r.json();if(!r.ok)throw new Error(data.error||'Could not load widget settings.');defaults=data.defaults;for(const key of Object.keys(fields))fields[key].value=data[key]}catch(e){document.querySelector('#message').textContent=e.message||'Could not load widget settings.'}}
async function save(useDefaults=false){const messages=[document.querySelector('#message'),document.querySelector('#bottomMessage')],body=useDefaults?defaults:Object.fromEntries(Object.entries(fields).map(([key,field])=>[key,field.value]));messages.forEach(message=>message.textContent='Saving…');try{const r=await fetch(settingsUrl,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),data=await r.json();if(!r.ok)throw new Error(data.error||'Could not save widget settings.');if(useDefaults)for(const key of Object.keys(fields))fields[key].value=data[key];defaults=data.defaults;document.querySelector('#preview').src=widgetUrl+'?preview='+Date.now();messages.forEach(message=>message.textContent='Saved. Refresh the OBS Browser Source if it is already open.')}catch(e){messages.forEach(message=>message.textContent=e.message||String(e))}}
document.querySelector('#save').onclick=()=>save();document.querySelector('#saveBottom').onclick=()=>save();document.querySelector('#defaults').onclick=()=>save(true);load();
</script></body></html>"""
    return page.replace("__AREA_SLUG__", area_slug_json).replace("__AREA_NAME__", area_name)


@app.get("/overlay")
@app.get("/widget")
@app.get("/widget/kill")
def overlay_page():
    return render_widget_page("kill")


@app.get("/widget/<area_slug>")
def named_widget_page(area_slug: str):
    config = load_config()
    area = next((item for item in configured_areas(config) if item["slug"] == area_slug), None)
    if area is None:
        return "Capture area not found.", 404
    return render_widget_page(area_slug)


def render_widget_page(area_slug: str):
    config = load_config()
    area = next((item for item in configured_areas(config) if item["slug"] == area_slug), {})
    html = str(area.get("widget_html", config.get("widget_html", DEFAULT_WIDGET_HTML)))
    css = str(area.get("widget_css", config.get("widget_css", DEFAULT_WIDGET_CSS)))
    javascript = str(area.get("widget_js", config.get("widget_js", DEFAULT_WIDGET_JS)))
    encoded_html = base64.b64encode(html.encode("utf-8")).decode("ascii")
    encoded_css = base64.b64encode(css.encode("utf-8")).decode("ascii")
    encoded_js = base64.b64encode(javascript.encode("utf-8")).decode("ascii")
    area_key = json.dumps(area_slug)
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Kill count widget</title></head><body><div id='custom-widget-root'></div><script>
const decodeWidgetPart=value=>new TextDecoder().decode(Uint8Array.from(atob(value),character=>character.charCodeAt(0)));
const root=document.querySelector('#custom-widget-root');
const style=document.createElement('style');style.textContent='*{box-sizing:border-box}html,body{margin:0;background:transparent;overflow:hidden}'+decodeWidgetPart('""" + encoded_css + """');document.head.append(style);
root.innerHTML=decodeWidgetPart('""" + encoded_html + """').replaceAll('{{kills}}','<span data-kills></span>');
const areaSlug=""" + area_key + """;
let currentCount=null;window.killWidget={get count(){return currentCount},onChange(handler){window.addEventListener('killcountchange',event=>handler(event.detail.count))}};
function setCount(value){const next=Number.isInteger(value)?value:null;if(next===currentCount)return;currentCount=next;root.querySelectorAll('[data-kills],#kills').forEach(element=>element.textContent=next===null?'—':String(next));root.style.visibility='visible';window.dispatchEvent(new CustomEvent('killcountchange',{detail:{count:next}}))}
async function refreshCount(){try{const status=await(await fetch('/api/status')).json();setCount(status.areas?.[areaSlug]?.kills??null)}catch{setCount(null)}}
const customScript=document.createElement('script');customScript.textContent=decodeWidgetPart('""" + encoded_js + """');document.body.append(customScript);refreshCount();setInterval(refreshCount,750);
</script></body></html>"""


@app.get("/api/areas/<area_slug>/widget-settings")
def api_get_area_widget_settings(area_slug: str):
    config = load_config()
    area = next((item for item in configured_areas(config) if item["slug"] == area_slug), None)
    if area is None:
        return jsonify({"error": "Capture area not found."}), 404
    defaults = {"html": DEFAULT_WIDGET_HTML, "css": DEFAULT_WIDGET_CSS, "js": DEFAULT_WIDGET_JS}
    return jsonify({key: area.get(f"widget_{key}", config.get(f"widget_{key}", value))
                    for key, value in defaults.items()} | {"defaults": defaults})


@app.put("/api/areas/<area_slug>/widget-settings")
def api_save_area_widget_settings(area_slug: str):
    payload = request.get_json(silent=True) or {}
    values = {key: payload.get(key, "") for key in ("html", "css", "js")}
    if any(not isinstance(value, str) or len(value) > MAX_WIDGET_FIELD_LENGTH for value in values.values()):
        return jsonify({"error": f"Each widget section must be text no longer than {MAX_WIDGET_FIELD_LENGTH} characters."}), 400
    try:
        config = load_config()
        areas = configured_areas(config)
        target = next((item for item in areas if item["slug"] == area_slug), None)
        if target is None:
            return jsonify({"error": "Capture area not found."}), 404
        for key, value in values.items():
            target[f"widget_{key}"] = value
        config["capture_areas"] = areas
        write_config(config)
    except Exception as exc:
        log.exception("Could not save area widget customization")
        return jsonify({"error": f"Could not save widget customization: {exc}"}), 500
    return jsonify({**values, "defaults": {"html": DEFAULT_WIDGET_HTML,
                                             "css": DEFAULT_WIDGET_CSS,
                                             "js": DEFAULT_WIDGET_JS}})


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


@app.get("/api/areas")
def api_get_areas():
    config = load_config()
    with state.lock:
        areas = []
        for area in configured_areas(config):
            reading = state.area_readings.get(area["slug"], AreaReading())
            area_snapshot = {**area_reading_snapshot(reading, state.stale_after_seconds),
                             "slug": area["slug"], "name": area["name"],
                             "region": area.get("region") is not None}
            if area["slug"] != "kill":
                area_snapshot["count_rules"] = area.get("count_rules", DEFAULT_CUSTOM_COUNT_RULES)
            areas.append(area_snapshot)
    return jsonify({"areas": areas})


@app.post("/api/areas")
def api_add_area():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    if not name or len(name) > 60:
        return jsonify({"error": "Enter an area name between 1 and 60 characters."}), 400
    slug = _area_slug(name)
    if not slug:
        return jsonify({"error": "Use at least one letter or number in the name."}), 400
    try:
        config = load_config()
        areas = configured_areas(config)
        slugs = {area["slug"] for area in areas}
        base_slug = slug
        suffix = 2
        while slug in slugs:
            slug = f"{base_slug}-{suffix}"
            suffix += 1
        areas.append({"slug": slug, "name": name, "region": None,
                      "count_rules": dict(DEFAULT_CUSTOM_COUNT_RULES),
                      "widget_html": config.get("widget_html", DEFAULT_WIDGET_HTML),
                      "widget_css": config.get("widget_css", DEFAULT_WIDGET_CSS),
                      "widget_js": config.get("widget_js", DEFAULT_WIDGET_JS)})
        config["capture_areas"] = areas
        write_config(config)
        with state.lock:
            state.area_readings.setdefault(slug, AreaReading())
        return jsonify({"ok": True, "area": {"slug": slug, "name": name}}), 201
    except Exception as exc:
        log.exception("Could not add capture area")
        return jsonify({"error": f"Could not add capture area: {exc}"}), 500


@app.route("/api/areas/<area_slug>", methods=["PUT", "DELETE"])
def api_update_area(area_slug: str):
    try:
        config = load_config()
        areas = configured_areas(config)
        area = next((item for item in areas if item["slug"] == area_slug), None)
        if area is None:
            return jsonify({"error": "Capture area not found."}), 404
        if request.method == "DELETE":
            if area_slug == "kill":
                return jsonify({"error": "The default Kill area cannot be removed."}), 400
            areas = [item for item in areas if item["slug"] != area_slug]
            with state.lock:
                state.area_readings.pop(area_slug, None)
        else:
            payload = request.get_json(silent=True) or {}
            name = str(payload.get("name", "")).strip()
            if not name or len(name) > 60:
                return jsonify({"error": "Enter an area name between 1 and 60 characters."}), 400
            new_slug = "kill" if area_slug == "kill" else _area_slug(name)
            if not new_slug:
                return jsonify({"error": "Use at least one letter or number in the name."}), 400
            if any(item["slug"] == new_slug and item["slug"] != area_slug for item in areas):
                return jsonify({"error": "That name is already in use."}), 409
            area["name"] = name
            area["slug"] = new_slug
            if new_slug != area_slug:
                with state.lock:
                    state.area_readings[new_slug] = state.area_readings.pop(area_slug, AreaReading())
        config["capture_areas"] = areas
        write_config(config)
        return jsonify({"ok": True})
    except Exception as exc:
        log.exception("Could not update capture area")
        return jsonify({"error": f"Could not update capture area: {exc}"}), 500


@app.put("/api/areas/<area_slug>/count-rules")
def api_save_area_count_rules(area_slug: str):
    if area_slug == "kill":
        return jsonify({"error": "The default Kill counter keeps its existing safeguards."}), 400
    payload = request.get_json(silent=True) or {}
    rules = {}
    for key in DEFAULT_CUSTOM_COUNT_RULES:
        value = payload.get(key, False)
        if not isinstance(value, bool):
            return jsonify({"error": "Count safeguard options must be true or false."}), 400
        rules[key] = value
    try:
        config = load_config()
        areas = configured_areas(config)
        area = next((item for item in areas if item["slug"] == area_slug), None)
        if area is None:
            return jsonify({"error": "Capture area not found."}), 404
        area["count_rules"] = rules
        config["capture_areas"] = areas
        write_config(config)
    except Exception as exc:
        log.exception("Could not save custom counter safeguards")
        return jsonify({"error": f"Could not save count safeguards: {exc}"}), 500
    return jsonify({"ok": True, "count_rules": rules})


@app.post("/api/areas/<area_slug>/reset")
def api_reset_area(area_slug: str):
    config = load_config()
    if not any(area["slug"] == area_slug for area in configured_areas(config)):
        return jsonify({"error": "Capture area not found."}), 404
    now = time.time()
    with state.lock:
        reading = state.area_readings.setdefault(area_slug, AreaReading())
        reading.kills, reading.pending_value, reading.pending_count = 0, None, 0
        reading.latest_candidate_note = None
        reading.last_valid_candidate_at = now
        if area_slug == "kill":
            state.kills, state.pending_value, state.pending_count = 0, None, 0
            state.latest_candidate_note = None
            state.last_valid_candidate_at = now
    notify_live_clients()
    return jsonify({"ok": True, "kills": 0})


@app.post("/api/reset")
def api_reset():
    return api_reset_area("kill")


@app.get("/api/status")
def api_status():
    return jsonify(status_payload())


def status_payload() -> dict[str, Any]:
    snapshot = state.snapshot()
    config = load_config()
    with state.lock:
        area_snapshots = {}
        for area in configured_areas(config):
            reading = state.area_readings.get(area["slug"], AreaReading())
            area_snapshots[area["slug"]] = {
                **area_reading_snapshot(reading, state.stale_after_seconds),
                "slug": area["slug"], "name": area["name"],
                "region": area.get("region") is not None,
                "count_rules": area.get("count_rules", DEFAULT_CUSTOM_COUNT_RULES),
            }
    snapshot["areas"] = area_snapshots
    return snapshot


@app.get("/api/events")
def api_live_events():
    @stream_with_context
    def stream():
        last_version = -1
        while True:
            with state.update_condition:
                changed = state.update_condition.wait_for(
                    lambda: state.update_version != last_version, timeout=15.0
                )
                version = state.update_version
            if not changed:
                yield ": keep-alive\n\n"
                continue
            last_version = version
            yield "data: " + json.dumps(status_payload(), separators=(",", ":")) + "\n\n"

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/config/location")
def api_config_location():
    return jsonify({"path": str(CONFIG_PATH)})


@app.get("/api/lan-access")
def api_get_lan_access():
    config = load_config()
    return jsonify({"enabled": bool(config.get("lan_access_enabled", False)),
                    "access_code": str(config.get("lan_access_code", ""))})


@app.put("/api/lan-access")
def api_set_lan_access():
    payload = request.get_json(silent=True) or {}
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        return jsonify({"error": "Choose whether phone access is enabled."}), 400
    code = str(payload.get("access_code", "")).strip()
    if enabled and not code:
        code = secrets.token_urlsafe(12)
    if enabled and (len(code) < 8 or len(code) > 128):
        return jsonify({"error": "The access code must be between 8 and 128 characters."}), 400
    try:
        config = load_config()
        config["lan_access_enabled"] = enabled
        config["lan_access_code"] = code if enabled else ""
        write_config(config)
    except Exception as exc:
        log.exception("Could not save local network access settings")
        return jsonify({"error": f"Could not save local network settings: {exc}"}), 500
    notify_live_clients()
    return jsonify({"ok": True, "enabled": enabled, "access_code": code,
                    "restart_required": True})


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
        reason = "Choose a valid JSON configuration object."
        return jsonify({"error": reason, "rejected": [reason], "applied": False}), 400
    try:
        # Retain host/port from this installation so imported settings cannot
        # unexpectedly expose the local service or change the OBS URL.
        current = load_config()
        unknown_keys = sorted(set(incoming) - (set(current) | {"lan_access_enabled", "lan_access_code"}))
        if unknown_keys:
            return jsonify({"error": "Configuration contains unsupported settings: " + ", ".join(unknown_keys),
                            "rejected": unknown_keys, "applied": False}), 400
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
        if "capture_areas" in incoming:
            imported_areas = incoming["capture_areas"]
            if not isinstance(imported_areas, list) or len(imported_areas) > 30:
                raise ValueError("capture_areas must be a list with no more than 30 areas.")
            area_slugs = set()
            for item in imported_areas:
                if not isinstance(item, dict):
                    raise ValueError("Each capture area must be an object.")
                slug, area_name = item.get("slug"), item.get("name")
                if (not isinstance(slug, str) or not re.fullmatch(r"[\w]+(?:-[\w]+)*", slug, flags=re.UNICODE)
                        or slug in area_slugs or not isinstance(area_name, str)
                        or not area_name.strip() or len(area_name) > 60):
                    raise ValueError("Capture areas need unique URL names and display names up to 60 characters.")
                area_slugs.add(slug)
                area_region = item.get("region")
                if area_region is not None:
                    if not isinstance(area_region, dict):
                        raise ValueError(f"The crop for {area_name} must be an object or null.")
                    ax, ay, aw, ah = (float(area_region[key]) for key in ("x", "y", "width", "height"))
                    if min(ax, ay, aw, ah) < 0 or aw <= 0 or ah <= 0 or ax + aw > 1 or ay + ah > 1:
                        raise ValueError(f"The crop for {area_name} is outside the captured image.")
                for key in ("widget_html", "widget_css", "widget_js"):
                    if key in item and (not isinstance(item[key], str) or len(item[key]) > MAX_WIDGET_FIELD_LENGTH):
                        raise ValueError(f"{key} for {area_name} must be text no longer than {MAX_WIDGET_FIELD_LENGTH} characters.")
                if "count_rules" in item:
                    count_rules = item["count_rules"]
                    if (not isinstance(count_rules, dict)
                            or any(not isinstance(count_rules.get(key, False), bool)
                                   for key in DEFAULT_CUSTOM_COUNT_RULES)):
                        raise ValueError(f"Count safeguard options for {area_name} must be true or false.")
            if "kill" not in area_slugs:
                raise ValueError("capture_areas must include the default Kill area.")
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
        # Network exposure is a local-machine choice and is never copied from
        # an imported backup onto another installation.
        imported["lan_access_enabled"] = bool(current.get("lan_access_enabled", False))
        imported["lan_access_code"] = str(current.get("lan_access_code", ""))
        # A packaged CPU/GPU build always keeps its selected inference backend.
        packaged_device = os.environ.get("FENIKZ_OCR_DEVICE")
        if packaged_device in {"cpu", "cuda"}:
            imported["ai_device"] = packaged_device
        temp_path = CONFIG_PATH.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(imported, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temp_path, CONFIG_PATH)
        notify_live_clients()
        return jsonify({"ok": True, "applied": True,
                        "notice": "Network access settings remain unchanged on this device."})
    except (KeyError, TypeError, ValueError, OSError) as exc:
        reason = str(exc)
        return jsonify({"error": f"Configuration was not imported: {reason}",
                        "rejected": [reason], "applied": False}), 400


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
        if source_image is not None:
            # Keep one still for the browser; calibration does not need a
            # continuous second copy of every OCR capture frame.
            state.calibration_preview_requested = False
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
    area_slug = str(payload.get("area_slug", "kill"))
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
        areas = configured_areas(config)
        area = next((item for item in areas if item["slug"] == area_slug), None)
        if area is None:
            return jsonify({"error": "Capture area not found."}), 404
        if title:
            config["window_title_contains"] = title
        saved_region = {"x": round(x, 6), "y": round(y, 6),
                        "width": round(width, 6), "height": round(height, 6)}
        area["region"] = saved_region
        config["capture_areas"] = areas
        if area_slug == "kill":
            config["kill_region"] = saved_region
        write_config(config)
    except Exception as exc:
        log.exception("Could not save browser calibration")
        return jsonify({"error": f"Could not save calibration: {exc}"}), 500
    return jsonify({"ok": True, "area_slug": area_slug, "kill_region": saved_region})


@app.get("/api/crop.png")
def api_crop():
    with state.lock:
        crop_png = state.latest_crop_png
    if crop_png is None:
        return "No OCR crop has been captured yet.", 503
    return Response(crop_png, mimetype="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/crop/<area_slug>.png")
def api_area_crop(area_slug: str):
    with state.lock:
        reading = state.area_readings.get(area_slug)
        crop_png = reading.latest_crop_png if reading is not None else None
    if crop_png is None:
        return "No OCR crop has been captured for this area yet.", 503
    return Response(crop_png, mimetype="image/png", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    start_worker()
    start_console_reload_watcher()
    if os.name == "nt" and sys.stdin.isatty():
        log.info("Press Ctrl+R to restart the server, or Ctrl+C to stop it")
    config = load_config()
    host = "0.0.0.0" if config.get("lan_access_enabled", False) else config.get("host", "127.0.0.1")
    app.run(host=host, port=int(config.get("port", 8765)), threaded=True, use_reloader=False)
