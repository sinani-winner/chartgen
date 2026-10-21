from pathlib import Path

from chartgen.chart import load_chart, parse_chart
from chartgen.scoring import score

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "example.chart"


def m(ref, pred):
    return score(parse_chart(ref), parse_chart(pred)).metrics


def test_identical_is_perfect():
    c = load_chart(EXAMPLE)
    s = score(c, c).metrics
    for k in ("root", "timing_f", "timing_f_exact", "quality", "bass", "form_boundary_f",
              "form_seq", "key", "mod_f", "free_f", "stop_f", "chord_full"):
        assert s[k] == 1.0, k
    assert s["false_changes"] == 0 and s["false_tensions"] == 0


def test_enharmonic_spelling_is_not_an_error():
    s = m("Key: Ab\n| Ab | Db/F |", "Key: G#\n| G# | C#/E# |")
    assert s["chord_full"] == 1.0 and s["key"] == 1.0


def test_root_quality_bass_are_separate():
    s = m("| C | G/B | Am7 | F |", "| Cmaj7 | G | Am7 | F |")
    assert s["root"] == 1.0
    assert s["quality"] == 0.75          # Cmaj7 vs C wrong on 1 of 4 bars
    assert s["bass"] == 0.75             # G vs G/B
    assert s["false_tensions"] == 1      # maj7 added on C


def test_missing_bar_does_not_shift_everything():
    ref = "| C | G | Am | F | C | G | F | C |"
    pred = "| C | Am | F | C | G | F | C |"   # dropped the 2nd bar
    s = m(ref, pred)
    assert abs(s["root"] - 7 / 8) < 1e-9
    assert s["bars_pred"] == 7


def test_adlib_passing_chord_is_false_change():
    s = m("| C | F |", "| C E7@4.5 | F |")
    assert s["false_changes"] == 1
    assert s["timing_r"] == 1.0


def test_push_timing_tolerance():
    ref, pred = "| C | >G |", "| C | G |"
    s = m(ref, pred)
    assert s["timing_f"] == 1.0           # within one 8th note
    assert s["timing_f_exact"] == 0.0     # but the push was missed


def test_form_repeat_count():
    ref = "[V]\n| C | G |\n[C] x2\n| F | G |"
    pred = "[V]\n| C | G |\n[C]\n| F | G |"
    s = m(ref, pred)
    assert s["form_seq"] < 1.0
    assert s["bars_pred"] == 4 and s["bars_ref"] == 6


def test_modulation_detection():
    ref = "Key: G\n[C]\n| G | D |\n[Key: Ab]\n[C]\n| Ab | Eb |"
    good = m(ref, ref)
    miss = m(ref, "Key: G\n[C]\n| G | D |\n[C]\n| Ab | Eb |")
    assert good["mod_f"] == 1.0 and miss["mod_f"] == 0.0
    assert miss["key"] == 0.5


def test_free_intro_timing_is_lenient():
    ref = "[Intro] (free)\n| C G | F |\n[V]\n| C |"
    pred = "[Intro] (free)\n| C . . G | F |\n[V]\n| C |"
    s = m(ref, pred)
    assert s["timing_f"] == 1.0


def test_modulation_after_skipped_repeat_still_counts():
    # prediction plays the modulated chorus once instead of twice
    ref = "Key: G\n[B]\n| Am | D |\n[Key: A]\n[C] x2\n| A | E | D | E |"
    pred = "Key: G\n[B]\n| Am | D |\n[Key: A]\n[C]\n| A | E | D | E |"
    s = m(ref, pred)
    assert s["mod_f"] == 1.0
    assert s["form_boundary_f"] < 1.0      # the missing repeat is still penalized
