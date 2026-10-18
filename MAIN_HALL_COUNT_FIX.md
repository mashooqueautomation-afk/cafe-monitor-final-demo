# Main Hall Count and Total Occupancy — Quick Guide

## Why Camera 03 may show video but no count

The dashboard plays video for every camera, but AI inference runs only on selected cameras. The default selection is now Camera 03 (Main Hall), using overlapping 2×2 image tiles for better recall of small/far seated people. This costs more CPU than a single full-frame inference, so keep only Camera 03 selected while validating the visual count.

## Validate the Main Hall visible count

1. Stop monitoring.
2. In **Analyze cameras**, select **03 only**.
3. Start monitoring and wait for the first successful inference.
4. Open **LAST AI EVIDENCE** on Camera 03; compare the boxes against a manually counted frame.
5. Check **Processing**, inference milliseconds, result age, and the note below Main Hall. A stale/old result is not a current count.

The model's count is a detection estimate, not a guaranteed exact census. People hidden behind furniture, strongly occluded, or represented by very few pixels can still be missed. If the annotated evidence misses people, collect several representative frames and label them for a café-specific training/validation set. Do not claim exact accuracy from one successful API response.

## Show 16 initially inside, then 3 new entries = 19

For an occupancy estimate, do **not** add the Main Hall count to entry events automatically: the same person may appear on both cameras. Instead:

1. Stop monitoring.
2. Click the **Entrance occupancy · 07** profile.
3. Enter the staff-verified starting occupancy `16` in the **Verified opening occupancy** field that appears before starting.
4. Start monitoring. The backend installs the baseline before the first frames are read, avoiding a race where early demo entries could be counted before the baseline was saved.
5. The `Inside now` metric follows `16 + valid entries − valid exits`. Three correctly detected entry events with no exits should make it 19.
6. Manually audit Camera 07's line and direction on the provided recording. If the three crossings are missed or counted twice, line position, direction, sampling rate, or tracking needs camera-specific calibration. While monitoring is already running, the separate correction field can still set the current staff-verified count.

The dual-core CPU is configured to prioritize one AI camera at a time. This is deliberate: all video panels can play, but Main Hall detection and entrance counting should be validated in separate runs on this machine. In a real live deployment, the starting occupancy must be verified at the moment the counter is initialized.
