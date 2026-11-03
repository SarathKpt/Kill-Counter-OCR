# FENIKZ BGMI Kill OCR (Windows + MobileStream + OBS)

This helper reads the **BGMI in-game kill number** from the MobileStream window using a local AI OCR model. It does not read game memory, inject into BGMI, or interact with the game. A small local web server provides a transparent OBS Browser Source and a control page.

## What this package assumes

- BGMI runs on your Android device/tablet.
- The device is mirrored to your Windows streaming PC in the **MobileStream** window.
- The MobileStream window remains open and is not minimized. Other apps, including Streamlabs, can cover it because capture targets the window directly.
- OBS runs on the same Windows PC.
- The BGMI HUD has a stable visible `Finished` / kills number during the match.

The capture uses Windows Graphics Capture to read the MobileStream window independently of the desktop pixels shown in front. Windows' application-window capture API is designed to provide frames from an application window, and the Python `windows-capture` package exposes title-based window capture for this purpose. [Microsoft documentation](https://learn.microsoft.com/en-us/windows/apps/develop/media-authoring-processing/screen-capture) · [windows-capture package](https://pypi.org/project/windows-capture/)

## Current build status

The service provides a lightweight count page, a separate control and diagnostics page, a customizable transparent OBS widget, AI recognition loop, stable-count logic, and browser-based calibration. The server captures the configured window through Windows Graphics Capture; the browser calibration page can change the title match (default `MobileStream`) and crop region. The widget customization page lets you edit HTML, CSS, and JavaScript, with a live preview. The browser only displays a local preview and sends the selected settings back to the server. The preview is available only while calibrating, and normal recognition continues to capture the window in the background even when other windows cover it. A `null` region intentionally leaves recognition waiting with a clear message. The default RapidOCR model runs locally through ONNX Runtime; game frames and crops are not sent to an online service. CPU inference is limited to one thread by default (`ai_cpu_threads`) while keeping the full recognition pipeline enabled. In GPU mode, CUDA uses one shared execution stream (`ai_cuda_single_stream`); GPU kernels still run across the GPU's hardware threads. Tesseract remains available as an optional fallback by setting `recognition_engine` to `tesseract`.

## 1. Install Python dependencies

1. Install Python for Windows and confirm `python` works in Command Prompt. The installer uses that Python to create the private environment, with the Python Launcher (`py`) as a fallback.
2. Keep this folder somewhere permanent, for example `C:\FENIKZ\BGMI_Kill_OCR`.
3. Double-click `install.bat`. It creates or updates the private `.venv` and installs the recognition model and Python libraries. Run it again after receiving project updates that change dependencies.

## 1a. Build the CPU or GPU Windows app

The packaged release runs in the Windows notification area (system tray). Select its tray icon to open the control page; the menu also opens the OBS widget and the configuration folder/file. The control page can export a JSON backup and import a previously exported JSON configuration. Packaged settings are stored at `%LOCALAPPDATA%\FENIKZ\BGMI Kill OCR\config.json`; importing settings keeps this installation's local host and port, and the chosen CPU/GPU build keeps its inference mode. The first launch copies the bundled defaults into that folder.

Build on Windows from the project folder after installing its matching environment:

- `install.bat`, then `build_cpu.bat` creates `releases\apps\FENIKZ Kill OCR CPU`.
- `install_gpu.bat`, then `build_gpu.bat` creates `releases\apps\FENIKZ Kill OCR GPU`.

Each output is a folder-based app; distribute the complete folder (or wrap that folder in an installer). The GPU output needs a compatible NVIDIA driver and runtime libraries. Test the GPU build on a PC without the development Python environment before publishing. The editable source setup (`run.bat` / `run_gpu.bat`) continues to use the project-folder `config.json`.

### Try NVIDIA GPU recognition

The normal `run.bat` environment uses CPU inference. To try CUDA without replacing it, stop the current service, double-click `install_gpu.bat` once, then start `run_gpu.bat`. This creates a separate `.venv-gpu` with ONNX Runtime's NVIDIA package and installs the CUDA runtime/cuDNN Python packages. The ONNX Runtime GPU package still needs the CUDA 13 toolkit libraries (including cuBLAS) and a compatible NVIDIA driver; the display driver alone does not supply these DLLs. Install the CUDA 13 toolkit from NVIDIA, then reopen the command window before starting `run_gpu.bat`. The control page reports whether CUDA is active or the app has fallen back to CPU. Keep Streamlabs and the game on the same scene/settings while comparing GPU use and dropped frames; use `run.bat` to return to the CPU setup. See the [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) and [NVIDIA cuDNN Windows instructions](https://docs.nvidia.com/cudnn/installation/backend/windows/).

## 2. (Optional) Install the Tesseract OCR engine

RapidOCR is the default and does not need a separate OCR program. To use the optional Tesseract fallback, install Tesseract:

1. Open the Tesseract OCR documentation's Windows installation instructions:
   https://github.com/tesseract-ocr/tessdoc/blob/main/Installation.md
2. Follow its link to the UB Mannheim Windows installer and install the current 64-bit Tesseract 5 build.
3. The default install path is usually `C:\Program Files\Tesseract-OCR\tesseract.exe`.
4. Open Command Prompt and verify:

   ```bat
   "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
   ```

If you installed it elsewhere, put the full `tesseract.exe` path in `config.json` under `tesseract_cmd`.

## 3. Start and calibrate in the browser

1. Start MobileStream and leave its window open (it may be covered by Streamlabs or other apps, but do not minimize it).
2. Double-click `run.bat` (or `run_gpu.bat` for the GPU environment). The control page opens automatically.
3. In the control page, click **Open browser calibration**, then **Start preview**. The server captures MobileStream directly and sends a local preview to the page.
4. Drag a tight rectangle around only the kill number beside `Finished`, then save. The preview stops; background capture continues through Windows Graphics Capture.
5. Leave the service's console running during the stream. The local control page is:

   http://127.0.0.1:8765/control

Press **Ctrl+R** in the server console to restart the service and reload its settings; press **Ctrl+C** to stop it. Start the service through `run.bat` or `run_gpu.bat` for the restart shortcut to relaunch it automatically.

It shows the latest AI reading, confidence, detected window, current confirmed kills, and service status. The browser never requests screen-sharing permission; window capture and AI processing stay on the local server.
The control page also shows the exact crop sent to OCR and its current numeric candidate, which helps distinguish a capture problem from a recognition problem. The OBS widget stays transparent until a fresh count is available and hides again if the local status service becomes unreachable.

## 4. Add the counter to OBS

1. In OBS, open the scene containing your portrait livestream layout.
2. Add **Browser** as a new source.
3. Set URL to:

   `http://127.0.0.1:8765/widget`
4. Set an initial width of `520` and height of `100` pixels.
5. Position it directly over the `LIVE KILL COUNT` number area in your existing overlay.
6. Keep this browser source above the static overlay image in the Sources list.

To customize the counter design or animate value changes, open `http://127.0.0.1:8765/customize`. HTML can use `{{kills}}` or an element with `data-kills`; JavaScript can listen with `window.killWidget.onChange(count => { ... })`. The current value is available as `window.killWidget.count`. The older `/overlay` URL points to this same customizable widget.

The page, overlay, and AI recognition loop refresh every 0.75 seconds. A number is accepted only after stable readings. During a match the count can only rise, by at most two per accepted reading; a larger jump is treated as an OCR error.

## 5. Match resets and false readings

- Increases require two stable readings by default.
- With RapidOCR selected, a number must have at least `ai_minimum_confidence` (0.60 by default) and pass the existing stable-reading check. Because each AI result is one recognition pass, three consecutive matching readings are required before changing the displayed count.
- With Tesseract selected, the older threshold and multi-pass consensus settings apply.
- Readings below the selected engine's confidence threshold are ignored so weak guesses cannot change the count.
- If no reliable number is read for `stale_after_seconds` (3 seconds by default), the control page and overlay hide the last confirmed count until a valid HUD reading returns. This avoids showing an old match count in the lobby.
- A nonzero decrease is ignored. A match reset is accepted only when zero is read three times in a row, so one mistaken digit cannot reset the count.
- A jump greater than `max_kill_increase_per_update` (2 by default) is ignored. If several kills occur between readings, the counter catches up only as the OCR sees plausible intermediate values.
- You can use **Reset match count** on the control panel after the new BGMI match has started and the HUD visibly shows zero.
- If OCR reads the wrong number, recalibrate more tightly around the kill numeral. The most common mistake is including the `Remaining` count in the crop.

You can tune `capture_minimum_update_interval_ms` (500 ms by default, about two capture frames per second), `poll_interval_seconds`, `ai_cpu_threads`, the stable-reading thresholds, and the confidence settings in `config.json`. The capture interval and polling rate reload while the service is running; restart `run.bat` after changing `ai_cpu_threads` or `recognition_engine`.

## Privacy / network

The HTTP service binds to `127.0.0.1` only. It is intended for OBS on the same PC and is not exposed to your LAN or the public internet.

## Troubleshooting

**“No captured frame arrived from MobileStream”**
- Confirm the selected capture window is open and not minimized.
- On the browser calibration page, change **Capture window title contains** to a distinctive part of the window title. The service will wait for and switch to that window automatically.

**OCR does not change**
- Open `/control` and check the live crop preview.
- Ensure the crop contains only the `Finished` kill numeral.
- MobileStream may be covered, but should not be minimized. This build captures the application window itself rather than the visible desktop.
- Recalibrate after changing BGMI HUD layout or MobileStream resolution/aspect ratio.

**OBS shows a blank browser source**
- Confirm `run.bat` is still running.
- Open `http://127.0.0.1:8765/widget` in a normal browser on the same PC. The older `/overlay` URL remains available too.
- In OBS, refresh the Browser Source.

**Port 8765 is already in use**
- Close the other local service or change `port` in `config.json`, then use the matching URL in OBS.
