import pytest

from chartgen.chords import ChordParseError, parse_chord, parse_key


@pytest.mark.parametrize("a,b", [
    ("Cmaj7", "CM7"), ("Cmaj7", "CΔ7"), ("D#m", "Ebm"), ("Cm7b5", "Cø"),
    ("C6/9", "C69"), ("Csus", "Csus4"), ("Cadd2", "Cadd9"), ("C/C", "C"),
    ("Cmin7", "C-7"), ("Cdim7", "C°7"), ("Cm(maj7)", "CmM7"), ("C2", "Cadd9"),
])
def test_equivalent(a, b):
    assert parse_chord(a) == parse_chord(b)


@pytest.mark.parametrize("a,b", [
    ("C", "Cmaj7"), ("C", "C/E"), ("Cadd9", "C9"), ("C7", "Cmaj7"), ("Csus4", "Csus2"),
    ("Cm7b5", "Cdim7"), ("G/B", "B"),
])
def test_different(a, b):
    assert parse_chord(a) != parse_chord(b)


@pytest.mark.parametrize("sym,intervals", [
    ("C", {0, 4, 7}), ("Am", {0, 3, 7}), ("G7", {0, 4, 7, 10}), ("Fmaj7", {0, 4, 7, 11}),
    ("Em7", {0, 3, 7, 10}), ("Dsus4", {0, 5, 7}), ("Dsus2", {0, 2, 7}), ("Cadd9", {0, 2, 4, 7}),
    ("Bdim", {0, 3, 6}), ("Bm7b5", {0, 3, 6, 10}), ("C9", {0, 2, 4, 7, 10}),
    ("D7sus4", {0, 5, 7, 10}), ("C5", {0, 7}), ("Caug", {0, 4, 8}), ("C6", {0, 4, 7, 9}),
    ("G7b9", {0, 1, 4, 7, 10}), ("Am11", {0, 2, 3, 5, 7, 10}),
])
def test_intervals(sym, intervals):
    assert parse_chord(sym).intervals == frozenset(intervals)


def test_root_and_bass():
    c = parse_chord("D/F#")
    assert (c.root, c.bass_pc) == (2, 6)
    assert parse_chord("Bb").root == 10
    assert parse_chord("G/B").root == 7


def test_tensions():
    assert parse_chord("C").tensions() == set()
    assert parse_chord("Dsus4").tensions() == set()
    assert parse_chord("Cmaj7").tensions() == {"maj7"}
    assert parse_chord("C9").tensions() == {"b7", "9"}


@pytest.mark.parametrize("bad", ["H", "Cxyz", "c", "C/H", ""])
def test_bad(bad):
    with pytest.raises(ChordParseError):
        parse_chord(bad)


@pytest.mark.parametrize("sym", ["C", "Am", "G7", "Fmaj7", "Bm7b5", "Cdim7", "C9", "Cm9", "D7sus4",
                                 "Cadd9", "C6/9", "Cm(maj7)", "G7b9", "Csus2", "C13"])
def test_canonical_name_roundtrip(sym):
    c = parse_chord(sym)
    assert parse_chord(c.name()) == c


def test_key():
    assert parse_key("Ab") == parse_key("G#")
    assert parse_key("Em").minor and not parse_key("E").minor
    with pytest.raises(ChordParseError):
        parse_key("X")
