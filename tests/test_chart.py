from pathlib import Path

import pytest

from chartgen.chart import ChartError, build_timeline, chart_from_dict, load_chart, parse_chart
from chartgen.chords import parse_chord, parse_key

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "example.chart"


def bars_of(text):
    return parse_chart(text).expand()


def test_header_one_line():
    c = parse_chart("Key: F#m   BPM ≈ 72 (가변)   3/4\n| F#m |")
    assert c.meta["key"] == "F#m" and c.meta["bpm"] == 72 and c.meta["time"] == (3, 4)
    assert c.meta["bpm_variable"]


def test_beat_positions():
    b = bars_of("| C G | C . . G | C G@4 | C G A F | C G@2.5 |")
    assert [e.beat for e in b[0].events] == [1, 3]
    assert [e.beat for e in b[1].events] == [1, 4]
    assert [e.beat for e in b[2].events] == [1, 4]
    assert [e.beat for e in b[3].events] == [1, 2, 3, 4]
    assert [e.beat for e in b[4].events] == [1, 2.5]


def test_ambiguous_bar_is_error():
    with pytest.raises(ChartError):
        parse_chart("| C D E |")


def test_repeat_and_percent():
    b = bars_of("[Verse]\n| C | % |\n[Chorus] x3\n| F | G |")
    assert len(b) == 2 + 6
    assert str(b[1].events[0].chord) == "C"
    assert len({x.section_instance for x in b}) == 4
    assert [x.section_type for x in b[:3]] == ["verse", "verse", "chorus"]


def test_key_change_between_sections_applies_to_next():
    b = bars_of("Key: G\n[Chorus] x2\n| G |\n[Key Up → Ab]\n[Chorus]\n| Ab |")
    assert [str(x.key) for x in b] == ["G", "G", "Ab"]


def test_key_change_inside_section_and_relative():
    b = bars_of("Key: G\n[Chorus]\n| G | D |\n[Key +1]\n| Ab |")
    assert b[0].key == parse_key("G") and b[2].key == parse_key("Ab")


def test_push_moves_change_into_previous_bar():
    tl = build_timeline(bars_of("| C | >G |"))
    assert tl.changes() == [7]          # 4&-of-bar-1 in 8th-note frames
    assert tl.frames[7] == parse_chord("G")


def test_free_and_flags():
    b = bars_of("[Intro] (free)\n| C | G |\n[Verse]\n| C (stop) |")
    assert b[0].free and b[1].free and not b[2].free
    assert "stop" in b[2].flags


def test_low_confidence_marker():
    b = bars_of("| Cmaj7? |")
    assert b[0].events[0].conf is not None and b[0].events[0].conf < 0.5


def test_comments_and_sharps():
    b = bars_of("# 메모\nKey: F#   # 키\n| F# C#m7 |   # 끝 메모")
    assert str(b[0].key) == "F#"
    assert str(b[0].events[1].chord) == "C#m7"


def test_time_change():
    b = bars_of("4/4\n| C |\n[Time 3/4]\n| C | {2/4} D |")
    assert [x.time for x in b] == [(4, 4), (3, 4), (2, 4)]
    tl = build_timeline(b)
    assert tl.n_frames == 8 + 6 + 4


def test_example_roundtrip_text_and_json():
    c = load_chart(EXAMPLE)
    for c2 in (parse_chart(c.to_text()), chart_from_dict(c.to_dict())):
        a, b = c.expand(), c2.expand()
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert x.key == y.key and x.free == y.free and x.section_name == y.section_name
            assert [(e.beat, e.chord, e.push) for e in x.events] == [(e.beat, e.chord, e.push) for e in y.events]


@pytest.mark.parametrize("bad", [
    "| C | | D |", "| Cxx |", "[Key Up]\n| C |", "| C >  D |", "hello\n| C |", "| C@5 |", "| C C@1 |",
])
def test_errors_have_line_numbers(bad):
    with pytest.raises(ChartError):
        parse_chart(bad)
