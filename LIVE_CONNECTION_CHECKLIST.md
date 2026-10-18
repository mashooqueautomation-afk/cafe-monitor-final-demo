# Live Hikvision NVR handover checklist

Complete this with the cafe IT contact before switching from Demo Mode to Live Mode.

## Information to collect

- [ ] NVR make/model and firmware if available.
- [ ] NVR IP/hostname and the network/VPN path reachable from the demo PC.
- [ ] RTSP enabled; firewall route allowed from this PC; port confirmed by IT.
- [ ] Least-privilege view-only user. Avoid sharing the NVR administrator password.
- [ ] Exact RTSP sub-stream URLs for channels 01-08, verified against this NVR's configuration.
- [ ] Confirm which channel covers entrance/exit, dining tables, kitchen, service/cashier, beverage/cups, and any cash drawer.
- [ ] Camera resolution, FPS, angle, lighting, night mode, and whether each person/object of interest is large enough to detect.
- [ ] Cashier/POS vendor, available API/database/export, and written authorization before any integration.
- [ ] A representative recording for each camera and separate manually labelled test windows.

## URL entry

1. Stop the application (`STOP_DEMO.bat`).
2. Edit `.env`, setting `CAMERA_01_RTSP` through `CAMERA_08_RTSP` only with URLs verified by IT.
3. Do not paste credentials into tickets, screenshots, chat groups, source control, or public URLs. URL-encode reserved characters if required.
4. Recreate the services so the new environment is loaded:

   ```powershell
   docker compose up -d --force-recreate backend diagnostics
   ```

5. Select **Live Mode**, then start with one camera. Confirm a live image arrives and status is `playing`; then inspect logs/Diagnostics.
6. Compare entry/exit events with a written manual count and calibrate `config/cameras.yaml` before reporting accuracy.
7. Increase the number of AI-selected cameras gradually while observing CPU, memory, inference time, dropped/missed detections, and video continuity.

The usual Hikvision URL shape seen in some configurations is `rtsp://USER:PASSWORD@NVR_IP:554/Streaming/Channels/102`, but exact paths and channel suffixes are device/configuration dependent. This is a format example only; do not assume it is correct for this NVR.

## Go-live acceptance checks

- [ ] Every intended channel produces live video for at least 30 minutes without repeated reconnects.
- [ ] Entry/exit line and direction validated against at least 30 labelled crossings, including both directions.
- [ ] Camera 04 table regions are reviewed at the actual resolution and angle.
- [ ] Cup-count results reviewed on representative day/night/occluded clips; cup subtype/size/color is not accepted until a cafe-specific model is evaluated.
- [ ] Cash/POS integration remains marked unavailable until authorized source data and reconciliation rules are agreed and tested.
- [ ] Retention, operator access, network exposure, and client approval for CCTV handling are documented.
