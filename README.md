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

Build on Windows from the project folder after installing its matching environment. The packaged apps start with clean defaults (including LAN access disabled and an uncalibrated Kill area), not the developer's local capture regions or access code:

- `install.bat`, then `build_cpu.bat` creates `releases\apps\FENIKZ Kill OCR CPU`.
- `install_gpu.bat`, then `build_gpu.bat` creates `releases\apps\FENIKZ Kill OCR GPU`.

Each output is a folder-based app; distribute the complete folder (or wrap that folder in an installer). The GPU output needs a compatible NVIDIA driver and runtime libraries. Test the GPU build on a PC without the development Python environment before publishing. The editable source setup (`run.bat` / `run_gpu.bat`) continues to use the project-folder `config.json`.

### Try NVIDIA GPU recognition

The normal `run.bat` environment uses CPU inference. To try CUDA without replacing it, stop the current service, double-click `install_gpu.bat` once, then start `run_gpu.bat`. This creates a separate `.venv-gpu` with ONNX Runtime's NVIDIA package plus CUDA and cuDNN runtime wheels. Keep a compatible NVIDIA display driver installed. With these Python runtime packages, a separate CUDA Toolkit installation is usually unnecessary; the control page reports whether CUDA is active or the app has fallen back to CPU. Keep Streamlabs and the game on the same scene/settings while comparing GPU use and dropped frames; use `run.bat` to return to the CPU setup. See the [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) and [ONNX Runtime DLL preload guidance](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls).

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
3. In the control page, open **Manage capture areas** and choose **Calibrate** for Kill. Select **Capture screenshot** to load one still from MobileStream; the browser does not continuously refresh the calibration image.
4. Zoom in as needed, then drag a tight rectangle around only the kill number beside `Finished` and save. Select **Retake screenshot** only if the source has changed. To recalibrate later, return to **Manage capture areas** and choose **Calibrate** again. Normal background capture continues through Windows Graphics Capture.
5. Leave the service's console running during the stream. The local control page is:

   http://127.0.0.1:8765/

Press **Ctrl+R** in the server console to restart the service and reload its settings; press **Ctrl+C** to stop it. Start the service through `run.bat` or `run_gpu.bat` for the restart shortcut to relaunch it automatically.

It shows the latest AI reading, confidence, detected window, current confirmed kills, and service status. The browser never requests screen-sharing permission; window capture and AI processing stay on the local server.
The control page also shows the exact crop sent to OCR and its current numeric candidate, which helps distinguish a capture problem from a recognition problem. The OBS widget stays transparent until a fresh count is available and hides again if the local status service becomes unreachable.

## 4. Add the counter to OBS

1. In OBS, open the scene containing your portrait livestream layout.
2. Add **Browser** as a new source.
3. Set URL to:

   `http://127.0.0.1:8765/widget/kill`
4. Set an initial width of `520` and height of `100` pixels.
5. Position it directly over the `LIVE KILL COUNT` number area in your existing overlay.
6. Keep this browser source above the static overlay image in the Sources list.

To customize the counter design or animate value changes, choose **Customize widget** on that area's card in **Manage capture areas**. Every capture area has independent HTML, CSS, and JavaScript settings. The Kill editor is also available at `http://127.0.0.1:8765/customize`. HTML can use `{{kills}}` or an element with `data-kills`; JavaScript can listen with `window.killWidget.onChange(count => { ... })`. The current value is available as `window.killWidget.count`. The older `/overlay` URL points to the Kill widget.

The default Kill widget is `http://127.0.0.1:8765/widget/kill`. Add more named capture areas from **Control page → Manage capture areas**; each area gets its own URL under `/widget/<name>`, independent count, and customizable widget design. Calibrate each area around only its number. The older `/widget` and `/overlay` URLs remain aliases for the default Kill widget.

The AI recognition loop runs every 0.75 seconds. The control and capture-area pages update when the service publishes a new status; OBS widgets use their own status refresh. A number is accepted only after stable readings. During a match the count can only rise, by at most two per accepted reading; a larger jump is treated as an OCR error.

## 5. Match resets and false readings

- Increases require two stable readings by default.
- With RapidOCR selected, a number must have at least `ai_minimum_confidence` (0.60 by default) and pass the existing stable-reading check. Because each AI result is one recognition pass, three consecutive matching readings are required before changing the displayed count.
- With Tesseract selected, the older threshold and multi-pass consensus settings apply.
- Readings below the selected engine's confidence threshold are ignored so weak guesses cannot change the count.
- If no reliable number is read for `stale_after_seconds` (3 seconds by default), the control page and overlay hide the last confirmed count until a valid HUD reading returns. This avoids showing an old match count in the lobby.
- The default Kill counter keeps its safeguards: nonzero decreases are ignored, a reset requires repeated zero readings, and jumps above `max_kill_increase_per_update` are ignored.
- Custom capture areas have their own optional safeguards on their calibration page: limit increases to the configured maximum, ignore decreases, and require repeated confirmation before resetting to zero. All three are off by default for custom counters.
- You can use **Reset match count** on the control panel after the new BGMI match has started and the HUD visibly shows zero.
- If OCR reads the wrong number, recalibrate more tightly around the kill numeral. The most common mistake is including the `Remaining` count in the crop.

You can tune `capture_minimum_update_interval_ms` (500 ms by default, about two capture frames per second), `poll_interval_seconds`, `ai_cpu_threads`, the stable-reading thresholds, and the confidence settings in `config.json`. The capture interval and polling rate reload while the service is running; restart `run.bat` after changing `ai_cpu_threads` or `recognition_engine`.

## Privacy / network

By default the service binds to `127.0.0.1` only and is available only on this PC. To open the control page from a phone on the same private Wi-Fi, use **Phone access on local Wi-Fi** on the control page, set an access code, save, and restart the tray app. Find this PC's Wi-Fi IPv4 address in Windows Network settings, then open `http://<PC-IP>:8765/` on the phone. At the browser sign-in prompt, use username `fenikz` and the configured access code. Phone access is optional and disabled by default; disabling it and restarting returns the service to local-only binding.

LAN access is for a trusted, password-protected home network only. The service uses plain local HTTP, so the access code is not encrypted in transit. Do not use this on public Wi-Fi, forward the port on your router, or expose it to the public internet. Windows Firewall may ask to allow the app on private networks; allow it only if you intend to use phone access.

The control and capture-area pages receive live status through one pushed event connection instead of repeatedly requesting status. OBS widgets keep their own lightweight status refresh so they remain usable in OBS Browser Source.

## Troubleshooting

**“No captured frame arrived from MobileStream”**
- Confirm the selected capture window is open and not minimized.
- On the browser calibration page, change **Capture window title contains** to a distinctive part of the window title. The service will wait for and switch to that window automatically.

**OCR does not change**
- Open `/` and check the capture areas page for each area's latest crop and AI confidence.
- Ensure the crop contains only the `Finished` kill numeral.
- MobileStream may be covered, but should not be minimized. This build captures the application window itself rather than the visible desktop.
- Recalibrate after changing BGMI HUD layout or MobileStream resolution/aspect ratio.

**OBS shows a blank browser source**
- Confirm `run.bat` is still running.
- Open `http://127.0.0.1:8765/widget/kill` in a normal browser on the same PC. The older `/widget` and `/overlay` URLs remain aliases for Kill.
- In OBS, refresh the Browser Source.

**Port 8765 is already in use**
- Close the other local service or change `port` in `config.json`, then use the matching URL in OBS.
