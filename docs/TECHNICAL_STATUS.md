# Technical Status and Validation Limits

## Included in this package

- Eight camera channels are configured in `config/cameras.yaml`. Demo paths expect user-provided `sample.mp4` files in the corresponding `data/client_footage/camera_*` folders. No video or image media is bundled.
- Channels 01, 02, and 05 have explicit generic people-presence analytics. Channel 03 is configured for people counting and approximate within-zone dwell. Channel 04 has provisional table regions. Channel 06 produces generic cup/glass candidate counts. Channel 07 uses an entrance line counter. Channel 08 is video-only until its purpose is confirmed.
- Camera selection limits which feeds run AI inference, while configured video tiles remain visible. Default selection is Camera 02 and Camera 07 to reduce load on the dual-core CPU.
- YOLO inference runs on CPU. PyTorch thread limits, sampled demo playback, conservative inference settings, and diagnostics are included to reduce contention.
- The diagnostic dashboard and its messages are in English. The package includes an English README and an Egyptian Arabic README.

## Important limits

- Generic COCO classes do not reliably separate café cup types, sizes, colours, takeaway vs dine-in, or brands. Reference images in `data/cup_dataset/raw/` are used by a diagnostic image test only; they are not used to train a model automatically.
- Clean/dirty classification, cash-drawer state detection, cash reconciliation, POS/API integration, cross-camera re-identification, and end-to-end customer dwell are not implemented.
- Table regions and entrance-line settings are provisional and must be calibrated against the real camera view and checked against manual labels.
- Live RTSP integration code exists, but the target café NVR and its specific URLs have not been validated in this package.

## Test environment note

Backend tests should be run inside the built container using `docker compose exec -T backend sh -c "cd /workspace/backend && python -m pytest -q"`. Test results verify code paths and mocked behavior only; they do not establish detection accuracy or prove compatibility with the customer's NVR.
