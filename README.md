# AUTONIQ Cafe Monitor — English Guide

Arabic guide: [Egyptian Arabic README](README_AR_EG.md)  
Live NVR handover: [LIVE_CONNECTION_CHECKLIST.md](LIVE_CONNECTION_CHECKLIST.md)  
Technical status: [docs/TECHNICAL_STATUS.md](docs/TECHNICAL_STATUS.md)

## 1. Package contents and limitations

This is a local proof-of-concept for the café requirements: an eight-channel CCTV dashboard, selectable per-camera AI analytics, visible-person estimates, an entrance/exit counter, approximate dwell time in one camera zone, provisional table-status regions, generic cup/glass candidates, diagnostics, and session/event storage.

**No video or image files are included in this lightweight ZIP.** You must add your own authorized recordings before Demo Mode can display video. Exact paths are in Section 3. The application does not generate fake footage and does not substitute recordings when Live Mode is selected.

Several client requirements remain incomplete: reliable cup subtype/size/colour classification, clean/dirty table classification, cash-drawer detection, monetary reconciliation, POS integration, cross-camera identity, and end-to-end customer dwell. Read the status table below before presenting capabilities to the client.

## 2. Test PC and prerequisites

The target PC supplied for this project has an Intel Core i3-5005U CPU (2 cores), 8 GB RAM, SSD storage, and Intel HD 5500 graphics. Inference runs on the CPU; the Intel graphics chip is not a supported NVIDIA CUDA device. Defaults are conservative.

Required on Windows:

1. Docker Desktop with the Linux container engine running and Docker Compose v2.
2. Internet access for the first Docker image build and initial download of `yolov8n.pt`. The model is stored in the project's named Docker volume for reuse.
3. Approximately 5 GB of free disk space is recommended for image layers, Python/PyTorch dependencies, and model weights.

Host Python 3.12.1 and Git are not required to launch the Docker application. The backend container uses Python 3.12.

## 3. Add your recordings and reference photos

Before running Demo Mode, copy each video to the exact path below. Every file must be named `sample.mp4` (lowercase). The app runs in a Linux container, so path capitalization matters. MP4/H.264 is recommended.

| NVR channel | Intended view (confirm with client) | Exact local path | AI role |
|---|---|---|---|
| 01 | Side Seating Area | `data/client_footage/camera_01_seating/sample.mp4` | Visible people |
| 02 | Kitchen / Prep Area | `data/client_footage/camera_02_kitchen/sample.mp4` | Visible people |
| 03 | Main Hall | `data/client_footage/camera_03_main_hall/sample.mp4` | People count / zone dwell estimate |
| 04 | Wide Seating Area | `data/client_footage/camera_04_seating_wide/sample.mp4` | Provisional table regions |
| 05 | Service / Prep Counter | `data/client_footage/camera_05_service_counter/sample.mp4` | Visible people |
| 06 | Beverage Counter / Cups | `data/client_footage/camera_06_beverage_counter/sample.mp4` | Generic cup/glass candidates |
| 07 | Entrance / Doorway | `data/client_footage/camera_07_entrance/sample.mp4` | Entry/exit counter |
| 08 | Area not confirmed | `data/client_footage/camera_08_unmapped/sample.mp4` | Video only until configured |

Example: copy your kitchen recording to `data/client_footage/camera_02_kitchen/sample.mp4`. If only seven recordings are available, leave Camera 08 without a file; it will be unavailable in Demo Mode until a recording is added. Do not rename one camera's clip as another channel unless its channel mapping has been confirmed.

Optional cup reference photos go in `data/cup_dataset/raw/`. Supported extensions are `.jpg`, `.jpeg`, `.png`, and `.webp`; image filenames may be anything. Those images are used only by the diagnostics reference-image test and label planning. They are not used to train a model automatically.

## 4. Start the demo on Windows

1. Extract the ZIP into a normal folder, such as `D:\Projects\cafe-monitor-demo`. Do not extract it into `C:\Windows\System32`.
2. Add the video files as specified in Section 3.
3. Start Docker Desktop and wait until its engine reports **Running**.
4. Double-click `START_DEMO.bat`. It applies CPU-friendly values to `.env` without changing camera credentials, builds the backend/frontend images, starts the services, creates low-rate preview copies in `runtime/demo_proxy/`, and attempts an Intel OpenVINO export. On the first run, downloading weights, preparing proxies, and exporting the model can take several minutes. Original recordings in `data/` are never changed. If OpenVINO setup fails, the app keeps a PyTorch fallback. Missing Camera 08 footage is optional.
5. If Windows blocks the batch file, open PowerShell inside the extracted folder and run:

   ```powershell
   Copy-Item .env.example .env -ErrorAction SilentlyContinue
   docker compose up -d --build
   ```

6. Open:
   - Main dashboard: http://localhost:8080
   - Diagnostics: http://localhost:8090
   - API health: http://localhost:8000/health
   - API docs: http://localhost:8000/docs

7. Choose **Demo Mode** and click **Start monitoring**. Only channels with valid recordings can play. Missing channels show as unavailable; no footage is fabricated.

The first image build, preview conversion, and OpenVINO export can each take several minutes. The launcher tries to finish these one-time preparations before opening the dashboard so later playbacks do less CPU work. If OpenVINO preparation fails, check `runtime/openvino_export_status.txt` and `docker compose logs --tail 120 backend`; PyTorch is the fallback. If weight download fails, video playback may be available but AI remains unavailable until weights are installed.

## 5. Recommended settings for this laptop

The camera selector chooses **which cameras run AI**, not which camera tiles are visible. A fresh browser selects Camera 03 (Main Hall) for AI; other configured feeds remain video-only. Use the profile buttons to switch to Camera 07 for entrance/occupancy counting. This two-camera focus is intended to reduce load on the dual-core CPU.

For a kitchen-only demonstration, stop monitoring, select Camera 02, then start. For entrance counts, select Camera 07 only. For cup candidates, select Camera 06 only. For table regions, select Camera 04. For main-hall people/dwell estimates, select Camera 03. Selecting **All** may overload this PC, slow inference, or cause missed entrance counts.

A playing tile is not proof that AI inference succeeded. Check the inference counters and Diagnostics. If a camera reports errors, run:

```powershell
docker compose logs --tail 120 backend
```

## 6. Implemented status

| Requirement | Current status |
|---|---|
| Eight-channel dashboard | UI supports eight channels; add one recording per camera for Demo Mode, or configured RTSP URLs for Live Mode. |
| Visible people in seating, kitchen, and service counter | Generic person detection on selected cameras; cannot distinguish staff from customers. |
| People count and dwell | Approximate tracking within Camera 03 only; not total café visit time and no cross-camera identity. |
| Entry/exit and estimated people inside | Door-line tracker on Camera 07; direction and line placement need calibration against a manual count. |
| Occupied/empty tables | Provisional normalized regions on Camera 04; review them against the actual camera view. |
| Clean/dirty table classification | **Not implemented.** “Needs check” is a prompt for staff review, not proof of dirt or cleaning. |
| Generic drinkware detection | COCO `cup` and `wine glass` candidates on Camera 06; false positives and missed cups are possible. |
| Cup type, size, shape, colour, takeaway vs dine-in | **Not implemented as reliable classification.** Collect and label frames from the actual CCTV angles, train a café-specific model, and validate it on separate recordings. |
| Cash drawer detection / reconciliation | **Not implemented.** Person presence near the counter does not prove that a cash drawer opened or that money was counted. Reconciliation needs suitable footage and authorized POS/API/database data. |
| Live Hikvision RTSP | Connection code exists, but the actual café NVR has not yet been validated. Confirm exact URLs and permissions with café IT. |

## 7. Connect the Hikvision NVR

1. Ask café IT for the NVR IP/hostname, confirmation that RTSP is enabled, a restricted view-only account, and the exact RTSP sub-stream URL for channels 01–08.
2. Confirm the PC can reach the NVR over the café LAN or authorized VPN. Do not expose RTSP port 554 or this dashboard directly to the public internet.
3. Stop the app with `STOP_DEMO.bat`. Open `.env` in Notepad and fill `CAMERA_01_RTSP` through `CAMERA_08_RTSP`. Leave unknown channels empty until IT confirms them. This is only an example format, not a guaranteed path:

   ```text
   rtsp://USER:URL_ENCODED_PASSWORD@NVR_IP:554/Streaming/Channels/102
   ```

4. Recreate the relevant services so environment changes take effect:

   ```powershell
   docker compose up -d --force-recreate backend diagnostics
   ```

5. Select **Live Mode**, select one or two cameras for AI, and click **Start monitoring**. Live Mode does not fall back to local recordings. An empty RTSP setting remains offline.
6. Validate one camera at a time. Check video continuity, resolution/FPS, and manual counts before adding more feeds.

Use [LIVE_CONNECTION_CHECKLIST.md](LIVE_CONNECTION_CHECKLIST.md) during handover. Never share NVR credentials in screenshots, chat groups, GitHub, logs, or public URLs.

## 8. Useful commands

Run these in PowerShell from the project folder:

```powershell
docker compose ps
docker compose logs -f backend
docker compose exec -T backend sh -c "cd /workspace/backend && python -m pytest -q"
docker compose restart backend
docker compose down
```

Do not use `docker compose down -v` unless you intentionally want to delete the project's named Docker volumes, including cached model weights. This Compose project is named `cafe-monitor-work` and is separate from `tg-migrator`; run commands from the café project directory only.

## 9. Common problems

- **Docker daemon unavailable:** open Docker Desktop and wait for the engine to report Running. `docker --version` alone is not sufficient.
- **Camera tile is unavailable in Demo Mode:** confirm that the recording exists at the exact path in Section 3 and is named `sample.mp4` using lowercase characters.
- **No AI boxes or model-not-found error:** check internet access and `docker compose logs backend`; retry `START_DEMO.bat` after fixing network or model-volume problems. Check `runtime/openvino_export_status.txt` if OpenVINO preparation failed.
- **Small or distant people/cups are missed:** analyze one camera at a time. Raising `YOLO_IMGSZ` from 512 to 640 may improve recall but costs more CPU. Higher confidence can reduce false positives while increasing missed detections. No configuration change converts the generic COCO model into a café-specific cup classifier.
- **Live camera is offline:** verify the exact RTSP URL, view-only account, network route, RTSP setting, firewall, and sub-stream path with café IT.
- **Entrance counts are inaccurate:** analyze Camera 07 alone, calibrate line position/direction on the real view, and compare at least 30 manually labelled crossings before reporting accuracy.
- **Kitchen has no boxes:** confirm Camera 02 is selected for AI and the footage has sufficient lighting, resolution, and visible people. The model cannot identify staff roles.
- **Cup types are wrong:** the current model is not a cup subtype classifier. Gather CCTV frames, annotate target classes, train a suitable model, and test it on separate recordings.
- **Cup reference-image test has no files:** add reference images to `data/cup_dataset/raw/` and rerun the diagnostic.

## 10. Privacy and acceptance

This is a local proof of concept, not a production security or cash-accounting system. Handle recordings and cup photos as confidential client data. Agree access permissions, retention/deletion, authorized viewers, and manual acceptance criteria before deployment. Do not use AI estimates alone to accuse staff or reconcile cash.


## Low-spec playback and AI evidence

The default profile is tuned for a dual-core CPU with 8 GB RAM: 640-pixel person inference, 4 FPS cached demo previews, 1.5 dashboard JPEG updates per second, 640-pixel display width, and about one Main Hall analysis every five seconds. Camera 03 (Main Hall) uses 2x2 overlapping tiles to make distant/seated people larger at inference time; this costs up to four detector passes per sampled frame, so it runs only when Camera 03 is selected. OpenVINO is attempted for Intel CPU acceleration; PyTorch remains the fallback. These changes improve the chance of finding small people but cannot guarantee a count of 19 until checked against the exact footage.

Video playback starts independently of model loading. The launcher starts the services before any model download, so a slow first model download should not block the dashboard from opening. AI may remain unavailable until weights load; check the camera processing status and backend logs.

In Demo Mode, the `PEOPLE` value on presence cameras is the number of people visible in the sampled inference frame, not a cumulative visitor total; it can legitimately become zero when no person is visible. If an inference completes after the 5-second freshness limit, the Main Hall can show the observed count as `STALE` evidence, but the delayed result does not update entrance counters or live tracking. Entry/exit counts on a looping recorded clip are per playback cycle and reset when the same clip rewinds, to avoid counting the same prerecorded crossings repeatedly. Live RTSP does not loop the recording, so its session counters do not reset for that reason.

The live video tiles deliberately do not carry detector boxes forward onto newer frames. Reusing a previous detection on a newer frame made people appear marked at their old positions. Use **LAST AI EVIDENCE** to inspect the most recent successful inference frame; the boxes are drawn on the exact frame processed, and the UI labels its timestamp and source age. Return to **LIVE VIDEO** to see the current stream. Evidence is a delayed diagnostic snapshot, not a live overlay.

The dashboard requests each tile snapshot once per refresh (the previous implementation made a probe request and then loaded the same image again). This reduces browser requests on an eight-tile page.

`START_DEMO.bat` applies the low-spec performance values to the matching keys in `.env` while preserving RTSP credentials, database settings, and other unrelated entries. To change these tuned values permanently, edit `scripts/apply_low_spec_env.ps1` and then run the launcher.

## Updating the project without touching recordings

The performance-fix ZIP intentionally contains **no `data/` directory**. Extract it into the existing `cafe-monitor-final-demo` folder and approve replacing code files. The archive does not replace or delete `data/`, so your existing CCTV recordings and cup images stay where they are. Keep a separate backup of `.env` and `runtime/` before larger project changes.


## Offline Recording Analyzer

After the dashboard is running, stop monitoring in the UI and double-click `RUN_OFFLINE_ANALYSIS.bat`. The script samples each configured recording every two seconds of video time, applies the currently configured model, saves annotated screenshots with short movement trails for temporary person-track IDs, and generates `frames.csv`, `detections.csv`, `summary.json`, `contact_sheet.jpg`, and `index.html` under `runtime/offline_analysis/<run-id>/`. Camera 08 is skipped if no recording is present. This analysis uses CPU and should be run with monitoring stopped. It reports model outputs, not proven accuracy; sampled people counts are not unique customer counts, tracking IDs may switch, and the generic cup class is not a custom cup classifier.


## Main Hall count and occupancy workflow

See [MAIN_HALL_COUNT_FIX.md](MAIN_HALL_COUNT_FIX.md) for the distinction between Camera 03 visible people and Camera 07 total occupancy, including how to set a verified starting count of 16. The default AI selection is Camera 03, and other camera feeds stay video-only unless selected.
