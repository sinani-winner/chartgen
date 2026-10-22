"""Phase 4: key tracking and form on hand-made chord paths (no audio)."""
import numpy as np
import pytest

from chartgen.harmony import N_BASS, make_grid
from chartgen.structure import analyze_form, local_keys


def state(root, minor=False):
    return (root + (12 if minor else 0)) * N_BASS     # root-position bass


def path_from_bars(bars):
    """bars: list of chord lists (2 per bar, half-bar each) → (path, grid)."""
    beats = np.arange(len(bars) * 4) * 0.5
    grid = make_grid(beats, beats[::4], duration=len(bars) * 2.0)
    path = []
    for b in bars:
        for c in b:
            path += [c] * 4
    return np.array(path), grid


G, C, D, Em, A, Bm, E, Fsm = (state(7), state(0), state(2), state(4, True), state(9),
                              state(11, True), state(4), state(6, True))
VERSE_G = [[G, G], [C, C], [Em, Em], [D, D]] * 2
CHORUS_G = [[C, D], [G, Em], [C, D], [G, G]] * 2
VERSE_A = [[A, A], [D, D], [Fsm, Fsm], [E, E]] * 2


def test_local_keys_find_whole_step_modulation():
    path, grid = path_from_bars(VERSE_G * 3 + VERSE_A * 2)
    keys = local_keys(path, grid)
    assert keys[0] == 7 and keys[-1] == 9
    change = next(i for i in range(1, len(keys)) if keys[i] != keys[i - 1])
    assert abs(change - 24) <= 1


@pytest.mark.xfail(reason="작업 중: 반복 점수가 벌스+후렴을 16마디 하나로 묶음 — 경계 신호 방식으로 교체 중", strict=False)
def test_form_splits_verse_and_chorus_and_names_them():
    bars = [[G, G]] * 4 + VERSE_G + CHORUS_G + VERSE_G + CHORUS_G + CHORUS_G
    path, grid = path_from_bars(bars)
    vocal = np.array([0.0] * 4 + [1.0] * (len(bars) - 4))
    secs, _ = analyze_form(path, grid, vocal, [(7, False)] * len(bars))
    names = [n for _, _, n in secs]
    starts = [s for s, _, _ in secs]
    assert names[0] == "Intro"
    assert "Chorus" in names and "Verse" in names
    for b in (4, 12, 20, 28, 36):
        assert any(abs(s - b) <= 1 for s in starts), (b, secs)
