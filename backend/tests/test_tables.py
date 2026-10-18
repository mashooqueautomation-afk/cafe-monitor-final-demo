from app.tables import EMPTY, NEEDS_CHECK, OCCUPIED, TableMonitor

REGION = [{"id": "t1", "name": "Table 1", "x1": 0, "y1": 0, "x2": 1, "y2": 1}]


def feed(tm, seq, step=1.0, start=0.0):
    t, out = start, []
    for person, clutter in seq:
        out.append(tm.update("t1", person, clutter, t))
        t += step
    return out


def test_empty_by_default():
    assert feed(TableMonitor(REGION), [(False, False)] * 5)[-1] == EMPTY


def test_person_must_stay_before_table_is_occupied():
    s = feed(TableMonitor(REGION, occupy_seconds=3), [(True, False)] * 6)
    assert s[0] == EMPTY and s[-1] == OCCUPIED


def test_passer_by_does_not_occupy_table():
    s = feed(TableMonitor(REGION, occupy_seconds=3), [(True, False)] * 2 + [(False, False)] * 5 + [(True, False)] * 2)
    assert OCCUPIED not in s


def test_short_detector_dropout_keeps_table_occupied():
    s = feed(TableMonitor(REGION, occupy_seconds=3, leave_seconds=20), [(True, False)] * 5 + [(False, False)] * 4)
    assert s[-1] == OCCUPIED


def test_leftovers_after_person_leaves_become_needs_check():
    tm = TableMonitor(REGION, occupy_seconds=3, leave_seconds=5, clutter_seconds=5)
    s = feed(tm, [(True, True)] * 6 + [(False, True)] * 30)
    assert s[5] == OCCUPIED and s[-1] == NEEDS_CHECK


def test_leftovers_while_someone_sits_stay_occupied():
    s = feed(TableMonitor(REGION, occupy_seconds=3, clutter_seconds=5), [(True, True)] * 20)
    assert s[-1] == OCCUPIED


def test_table_without_leftovers_goes_back_to_empty():
    s = feed(TableMonitor(REGION, occupy_seconds=3, leave_seconds=5), [(True, False)] * 6 + [(False, False)] * 15)
    assert s[-1] == EMPTY


def test_staff_confirmation_clears_needs_check_until_items_disappear():
    tm = TableMonitor(REGION, occupy_seconds=3, leave_seconds=5, clutter_seconds=5)
    feed(tm, [(True, True)] * 6 + [(False, True)] * 30)
    assert tm.confirm_cleaned("t1") is True
    assert feed(tm, [(False, True)] * 10, start=100)[-1] == EMPTY       # leftovers still seen, but staff said clean
    feed(tm, [(False, False)] * 3, start=120)                            # items really gone -> ack resets
    assert feed(tm, [(False, True)] * 10, start=140)[-1] == NEEDS_CHECK  # new items -> needs check again
    assert tm.confirm_cleaned("nope") is False
