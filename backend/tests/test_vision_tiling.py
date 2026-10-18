import numpy as np

from app.vision import _suppress_duplicate_boxes, _tile_regions


def test_two_by_two_tiles_cover_frame_edges_and_have_overlap():
    frame = np.zeros((1080, 960, 3), dtype=np.uint8)
    regions = _tile_regions(frame, divisions=2, overlap=0.08)
    assert len(regions) == 4
    # Each tile carries its source-frame offset and image dimensions.
    assert regions[0][:4] == (0, 0, 518, 583)
    assert regions[-1][2] == 960
    assert regions[-1][3] == 1080
    # Neighbouring tiles overlap so a person on a tile boundary is not lost.
    assert regions[0][2] > regions[1][0]
    assert regions[0][3] > regions[2][1]


def test_duplicate_detections_in_overlapping_tiles_are_suppressed():
    items = [
        {"class_id": 0, "confidence": 0.8, "xyxy": [100, 100, 180, 260]},
        {"class_id": 0, "confidence": 0.7, "xyxy": [104, 102, 182, 261]},
        {"class_id": 0, "confidence": 0.6, "xyxy": [350, 100, 420, 260]},
        {"class_id": 41, "confidence": 0.9, "xyxy": [100, 100, 180, 260]},
    ]
    kept = _suppress_duplicate_boxes(items, iou_threshold=0.45)
    assert len(kept) == 3
    assert kept[0]["class_id"] == 41
    assert [x["confidence"] for x in kept if x["class_id"] == 0] == [0.8, 0.6]


def test_tiling_can_be_disabled_for_fast_single_frame_path():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert len(_tile_regions(frame, divisions=2, overlap=0.0)) == 4
