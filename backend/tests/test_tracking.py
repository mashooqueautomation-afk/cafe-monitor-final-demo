"""Door-line counter tests. These reproduce the failures of the old nearest-centroid counter
(a standing person near the door created ~19 fake events; a normal walker created none)."""
import pytest

from app.tracking import LineCounter, PersonTracker

W, H = 1280, 720


def box(cx, y2=400, w=80, h=300):
    return (cx - w / 2, y2 - h, cx + w / 2, y2)


def run(counter, frames, dt):
    """frames: list of lists of boxes, one list per inference tick."""
    t, events = 100.0, []
    for boxes in frames:
        ev, _ = counter.update(boxes, W, H, t)
        events += ev
        t += dt
    return events


def test_walker_at_low_fps_is_counted_once():
    # 1.5 FPS analytics: a normal walker moves ~170 px between ticks. The old code counted 0.
    c = LineCounter(entry_direction="left_to_right")
    run(c, [[box(200 + 167 * i)] for i in range(8)], dt=1 / 1.5)
    assert (c.entries, c.exits) == (1, 0)


def test_walker_at_entrance_fps_is_counted_once():
    c = LineCounter()
    run(c, [[box(200 + 30 * i)] for i in range(30)], dt=1 / 8)
    assert (c.entries, c.exits) == (1, 0)


def test_person_standing_on_the_line_creates_no_events():
    # The old code produced 10 entries + 9 exits here.
    c = LineCounter()
    run(c, [[box(640 + (10 if i % 2 else -10))] for i in range(40)], dt=1 / 8)
    assert (c.entries, c.exits) == (0, 0)


def test_dithering_in_and_out_of_the_band_counts_nothing():
    c = LineCounter()
    run(c, [[box(560)]] * 3 + [[box(x)] for x in (600, 650, 610, 660, 600)] + [[box(560)]] * 3, dt=1 / 8)
    assert (c.entries, c.exits) == (0, 0)


def test_right_to_left_is_an_exit_when_entry_is_left_to_right():
    c = LineCounter(entry_direction="left_to_right")
    run(c, [[box(1000 - 30 * i)] for i in range(30)], dt=1 / 8)
    assert (c.entries, c.exits) == (0, 1)


def test_entry_direction_can_be_reversed():
    c = LineCounter(entry_direction="right_to_left")
    run(c, [[box(1000 - 30 * i)] for i in range(30)], dt=1 / 8)
    assert (c.entries, c.exits) == (1, 0)


def test_two_people_crossing_in_opposite_directions():
    c = LineCounter()
    frames = [[box(400 + 40 * i), box(900 - 40 * i)] for i in range(16)]
    run(c, frames, dt=1 / 8)
    assert (c.entries, c.exits) == (1, 1)


def test_one_missed_detection_does_not_double_count_or_lose_the_person():
    c = LineCounter()
    xs = [200 + 30 * i for i in range(30)]
    frames = [[box(x)] if i not in (9, 10, 11) else [] for i, x in enumerate(xs)]  # 3 ticks lost mid-walk
    run(c, frames, dt=1 / 8)
    assert (c.entries, c.exits) == (1, 0)


def test_walk_in_then_walk_out_nets_to_zero():
    c = LineCounter()
    path = [200 + 30 * i for i in range(30)] + [1100 - 30 * i for i in range(30)]
    run(c, [[box(x)] for x in path], dt=1 / 8)
    assert (c.entries, c.exits) == (1, 1)


def test_person_who_first_appears_inside_the_band_is_not_counted_on_the_spot():
    c = LineCounter()
    run(c, [[box(645)], [box(650)], [box(700)], [box(760)], [box(820)]], dt=1 / 8)
    assert (c.entries, c.exits) == (0, 0)   # no known starting side -> no guess


def test_horizontal_line_uses_feet_and_top_to_bottom():
    c = LineCounter(orientation="horizontal", entry_direction="top_to_bottom")
    run(c, [[box(600, y2=200 + 25 * i)] for i in range(30)], dt=1 / 8)
    assert (c.entries, c.exits) == (1, 0)


def test_orientation_and_direction_must_match():
    with pytest.raises(ValueError):
        LineCounter(orientation="vertical", entry_direction="top_to_bottom")
    with pytest.raises(ValueError):
        LineCounter(orientation="diagonal")


def test_tracker_keeps_ids_and_drops_old_tracks():
    t = PersonTracker(max_age_seconds=2.0)
    a = t.update([(100, 300, 300)], 0.0)[0].id
    b = t.update([(260, 300, 300)], 0.7)[0].id   # moved 160px in 0.7 s
    assert a == b
    t.update([], 10.0)
    assert t.tracks == {}
