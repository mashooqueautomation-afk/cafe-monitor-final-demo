"""Table state machine: occupied / empty / needs_check.

Each configured table region gets one of three states:
  occupied     a person has been inside the region for a few seconds
  needs_check  nobody is there any more, but cups/plates/etc. are still detected
               (a REVIEW PROMPT for staff - it is not proof that the table is dirty)
  empty        nobody there and nothing detected

Time-based debounce stops the state flickering when the detector misses a person
for a frame or two. Pure Python, unit-tested with fake detections.
"""
from __future__ import annotations

from dataclasses import dataclass

OCCUPIED, EMPTY, NEEDS_CHECK = "occupied", "empty", "needs_check"


@dataclass
class _Table:
    id: str
    name: str
    state: str = EMPTY
    person_since: float | None = None      # when a person was first seen continuously
    person_last: float | None = None       # last time a person was seen
    clutter_since: float | None = None
    cleaned_ack: bool = False              # staff said "cleaned": ignore leftover detections until they vanish once


class TableMonitor:
    def __init__(self, regions: list[dict], occupy_seconds: float = 3.0,
                 leave_seconds: float = 20.0, clutter_seconds: float = 10.0):
        self.occupy_seconds, self.leave_seconds, self.clutter_seconds = occupy_seconds, leave_seconds, clutter_seconds
        self.tables: dict[str, _Table] = {}
        for i, r in enumerate(regions):
            tid = str(r.get("id", r.get("name", f"table_{i + 1}")))
            self.tables[tid] = _Table(id=tid, name=str(r.get("name", f"Table {i + 1}")))

    def update(self, table_id: str, person_present: bool, clutter_present: bool, now: float) -> str:
        t = self.tables[table_id]
        gap = 2.0   # a person missing for longer than this breaks "continuous presence" (unless already seated)
        if person_present:
            if t.person_since is None:
                t.person_since = now
            t.person_last = now
        elif t.person_last is not None:
            away = now - t.person_last
            if (t.state != OCCUPIED and away > gap) or away > self.leave_seconds:
                t.person_since = None          # a passer-by, or the seated person really left
        if clutter_present:
            if t.clutter_since is None:
                t.clutter_since = now
        else:
            t.clutter_since = None
            t.cleaned_ack = False              # leftovers are gone, so a new "cleaned" ack is needed next time

        if t.person_since is not None and t.person_last is not None and (
                t.state == OCCUPIED or t.person_last - t.person_since >= self.occupy_seconds):
            t.state = OCCUPIED
        elif (clutter_present and t.clutter_since is not None and not t.cleaned_ack
              and now - t.clutter_since >= self.clutter_seconds):
            t.state = NEEDS_CHECK
        else:
            t.state = EMPTY
        return t.state

    def confirm_cleaned(self, table_id: str) -> bool:
        """Staff pressed 'cleaned'. Returns False if the table id is unknown."""
        for t in self.tables.values():
            if table_id in (t.id, t.name):
                t.cleaned_ack = True
                if t.state == NEEDS_CHECK:
                    t.state = EMPTY
                return True
        return False

    def counts(self) -> dict[str, int]:
        c = {OCCUPIED: 0, EMPTY: 0, NEEDS_CHECK: 0}
        for t in self.tables.values():
            c[t.state] += 1
        return c

    def states(self) -> dict[str, str]:
        return {t.id: t.state for t in self.tables.values()}
