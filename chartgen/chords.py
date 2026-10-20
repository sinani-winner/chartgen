"""Chord symbol parsing.

A chord is reduced to: root pitch class, triad family, seventh type,
extension intervals, and bass pitch class. Two spellings that sound the same
(Cmaj7 / CM7 / CΔ7, D#m / Ebm) compare equal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOTE_RE = r"[A-G](?:#|b)?"

TRIADS = {
    "maj": frozenset({0, 4, 7}),
    "min": frozenset({0, 3, 7}),
    "dim": frozenset({0, 3, 6}),
    "aug": frozenset({0, 4, 8}),
    "sus4": frozenset({0, 5, 7}),
    "sus2": frozenset({0, 2, 7}),
    "5": frozenset({0, 7}),
}
SEVENTHS = {"b7": 10, "maj7": 11, "dim7": 9, "6": 9}
# extension name -> interval above root
EXT_INTERVAL = {"9": 2, "b9": 1, "#9": 3, "11": 5, "#11": 6, "13": 9, "b13": 8}


def note_pc(name: str) -> int:
    name = normalize_accidentals(name)
    pc = NOTE_BASE[name[0]]
    for ch in name[1:]:
        pc += 1 if ch == "#" else -1
    return pc % 12


def normalize_accidentals(s: str) -> str:
    return s.replace("♯", "#").replace("♭", "b")


class ChordParseError(ValueError):
    pass


@dataclass(frozen=True)
class Chord:
    root: int                      # pitch class 0-11
    triad: str                     # key of TRIADS
    seventh: str | None = None     # key of SEVENTHS or None
    ext: frozenset = field(default_factory=frozenset)  # extension names (EXT_INTERVAL keys)
    bass: int | None = None        # pitch class of slash bass, None if bass == root
    text: str = ""                 # original spelling (not used for comparison)

    @property
    def intervals(self) -> frozenset:
        iv = set(TRIADS[self.triad])
        if self.seventh:
            iv.add(SEVENTHS[self.seventh])
        iv.update(EXT_INTERVAL[e] for e in self.ext)
        return frozenset(iv)

    @property
    def bass_pc(self) -> int:
        return self.root if self.bass is None else self.bass

    @property
    def has_tension(self) -> bool:
        return self.seventh is not None or bool(self.ext)

    def tensions(self) -> set:
        """Tones beyond the basic triad/sus (7th, 6th, 9th, ...)."""
        t = set(self.ext)
        if self.seventh:
            t.add(self.seventh)
        return t

    def key(self):
        """Identity used for 'same chord' comparisons."""
        return (self.root, self.intervals, self.bass_pc)

    def __eq__(self, other):
        return isinstance(other, Chord) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __str__(self):
        return self.text or self.name()

    def quality_name(self) -> str:
        t, s, e = self.triad, self.seventh, set(self.ext)
        if t == "dim" and s == "b7":
            q = "m7b5"
        elif t == "dim" and s == "dim7":
            q = "dim7"
        else:
            base = {"maj": "", "min": "m", "dim": "dim", "aug": "aug",
                    "sus4": "", "sus2": "", "5": "5"}[t]
            sev = {None: "", "b7": "7", "maj7": "maj7", "dim7": "dim7", "6": "6"}[s]
            if t == "min" and s == "maj7":
                sev = "(maj7)"
            # fold the highest natural extension into the seventh number (C9, Cm11)
            if s in ("b7", "maj7") and "9" in e and not (t == "min" and s == "maj7"):
                top = "13" if "13" in e else "11" if "11" in e else "9"
                sev = sev.replace("7", top)
                e -= {"9", "11", "13"} if top != "9" else {"9"}
            if s == "6" and "9" in e:
                sev = "6/9"
                e -= {"9"}
            q = base + sev
            if t in ("sus4", "sus2"):
                q += t
            for name in ("9", "11", "13"):
                if name in e:
                    q += "add" + name
                    e.discard(name)
            for name in sorted(e):
                q += name
        return q

    def name(self, spell_root: str | None = None, spell_bass: str | None = None) -> str:
        root = spell_root or PC_NAMES_FLAT[self.root]
        s = root + self.quality_name()
        if self.bass is not None:
            s += "/" + (spell_bass or PC_NAMES_FLAT[self.bass])
        return s


PC_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
PC_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_CHORD_RE = re.compile(rf"^({NOTE_RE})(.*?)(?:/({NOTE_RE}))?$")


def _parse_quality(q: str):
    """Return (triad, seventh, ext set). Raises ChordParseError."""
    orig = q
    q = (q.replace("Δ", "maj").replace("°", "dim").replace("ø", "m7b5")
          .replace("−", "-").replace("(", "").replace(")", "").replace(",", "")
          .replace(" ", ""))
    triad = "maj"
    seventh = None
    ext: set = set()
    i = 0

    def take(*options):
        nonlocal i
        for opt in options:
            if q.startswith(opt, i):
                i += len(opt)
                return opt
        return None

    def take_number():
        nonlocal i
        for n in ("13", "11", "9", "7", "6", "5", "4", "2"):
            if q.startswith(n, i):
                i += len(n)
                return n
        return None

    def apply_number(n, seventh_kind):
        """'7'/'9'/'11'/'13' imply a seventh plus stacked extensions."""
        nonlocal seventh
        if n == "7":
            seventh = seventh_kind
        elif n == "9":
            seventh = seventh_kind
            ext.add("9")
        elif n == "11":
            seventh = seventh_kind
            ext.update({"9", "11"})
        elif n == "13":
            seventh = seventh_kind
            ext.update({"9", "13"})
        else:
            return False
        return True

    # --- triad / leading family ---
    if take("mmaj", "mMaj", "minmaj", "mM", "m(maj", "-maj", "-M"):
        triad = "min"
        n = take_number()
        if n is None or not apply_number(n, "maj7"):
            raise ChordParseError(f"알 수 없는 코드 성격: {orig!r}")
    elif take("maj", "Maj", "ma", "M"):
        n = take_number()
        if n is None:
            pass  # 'Cmaj' == C
        elif not apply_number(n, "maj7"):
            raise ChordParseError(f"알 수 없는 코드 성격: {orig!r}")
    elif take("min", "mi", "m", "-"):
        triad = "min"
    elif take("dim", "o"):
        triad = "dim"
        if take("7"):
            seventh = "dim7"
    elif take("aug", "+"):
        triad = "aug"

    # --- number after the family (C7, Cm7, C6, C69, C5) ---
    if seventh is None:
        if take("69", "6/9"):
            seventh = "6"
            ext.add("9")
        else:
            save = i
            n = take_number()
            if n in ("7", "9", "11", "13"):
                apply_number(n, "b7")
                # 'maj' written after a minor triad: Cm(maj7) handled above; Cm7maj not supported
            elif n == "6":
                seventh = "6"
            elif n == "5" and triad == "maj" and i == len(q):
                triad = "5"
            elif n is not None:
                i = save

    # --- modifiers: sus, add, alterations ---
    while i < len(q):
        if take("sus2"):
            triad = "sus2"
        elif take("sus4", "sus"):
            triad = "sus4"
        elif take("add9", "add2", "2"):
            ext.add("9")
        elif take("add11", "add4", "4"):
            ext.add("11")
        elif take("add13"):
            ext.add("13")
        elif take("maj7", "M7"):
            seventh = "maj7"
        elif take("b5", "-5"):
            if triad == "min":
                triad = "dim"
            elif triad == "maj":
                ext.add("#11")  # C7b5 -> treat flat-five as #11 color
            else:
                raise ChordParseError(f"알 수 없는 코드 성격: {orig!r}")
        elif take("#5", "+5"):
            triad = "aug"
        elif take("b9", "-9"):
            ext.add("b9")
        elif take("#9", "+9"):
            ext.add("#9")
        elif take("#11", "+11"):
            ext.add("#11")
        elif take("b13", "-13"):
            ext.add("b13")
        elif take("9"):
            ext.add("9")
        elif take("11"):
            ext.add("11")
        elif take("13"):
            ext.add("13")
        elif take("7"):
            seventh = seventh or "b7"
        elif take("alt"):
            ext.update({"b9", "#9"})
        else:
            raise ChordParseError(f"알 수 없는 코드 성격: {orig!r} (남은 부분 {q[i:]!r})")

    # sus2 + 9 is redundant
    if triad == "sus2":
        ext.discard("9")
    if triad == "sus4":
        ext.discard("11")
    return triad, seventh, ext


def parse_chord(text: str) -> Chord:
    """Parse a chord symbol like 'Em7', 'D/F#', 'Cmaj7', 'Bbm7b5', 'Dsus4'."""
    s = normalize_accidentals(text.strip()).replace("6/9", "69")
    m = _CHORD_RE.match(s)
    if not m:
        raise ChordParseError(f"코드로 읽을 수 없음: {text!r}")
    root_s, qual, bass_s = m.groups()
    triad, seventh, ext = _parse_quality(qual)
    root = note_pc(root_s)
    bass = note_pc(bass_s) if bass_s else None
    if bass == root:
        bass = None
    return Chord(root=root, triad=triad, seventh=seventh, ext=frozenset(ext),
                 bass=bass, text=text.strip())


# --- keys ---------------------------------------------------------------

_KEY_RE = re.compile(rf"^({NOTE_RE})\s*(m|min|minor|major|maj)?$", re.IGNORECASE)


@dataclass(frozen=True)
class Key:
    tonic: int
    minor: bool
    text: str = ""

    def __eq__(self, other):
        return isinstance(other, Key) and (self.tonic, self.minor) == (other.tonic, other.minor)

    def __hash__(self):
        return hash((self.tonic, self.minor))

    def __str__(self):
        return self.text or (PC_NAMES_FLAT[self.tonic] + ("m" if self.minor else ""))

    def transpose(self, semitones: int) -> "Key":
        t = (self.tonic + semitones) % 12
        return Key(t, self.minor, PC_NAMES_FLAT[t] + ("m" if self.minor else ""))


def parse_key(text: str) -> Key:
    s = normalize_accidentals(text.strip())
    m = _KEY_RE.match(s)
    if not m or not (m.group(1)[0].isupper()):
        raise ChordParseError(f"키로 읽을 수 없음: {text!r} (예: G, Em, Ab, F#m)")
    mode = (m.group(2) or "").lower()
    return Key(note_pc(m.group(1)), mode in ("m", "min", "minor"), text.strip())
