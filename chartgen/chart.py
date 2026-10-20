"""Chart model, text format parser/renderer, JSON I/O, and expansion.

Text format is documented in docs/CHART_FORMAT.md. Short version:

    Title: 곡 제목
    URL: https://youtu.be/...
    Key: G   BPM: 72   4/4

    [Intro] (free)
    | G | Cmaj7 | Em7 | Dsus4 D |

    [Chorus] x2
    | C | D | Em7 D/F# | G |
    | C . . >D |            # '.' = 한 박 유지, '>' = 8분음표 앞당김(푸시)

    [Key: Ab]               # 전조 (이후 전부)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .chords import Chord, ChordParseError, Key, parse_chord, parse_key

LOW_CONF = 0.3  # confidence stored for chords marked with '?'


class ChartError(ValueError):
    def __init__(self, msg, line=None):
        super().__init__(f"{line}번째 줄: {msg}" if line else msg)
        self.line = line


# ---------------------------------------------------------------- model

@dataclass
class Event:
    beat: float                 # nominal beat in the bar, 1-based (time-sig denominator units)
    chord: Chord | None         # None = N.C.
    push: bool = False          # arrives one 8th note before `beat`
    conf: float | None = None

    @property
    def text(self) -> str:
        return "N.C." if self.chord is None else str(self.chord)


@dataclass
class Bar:
    events: list[Event]
    time: tuple = (4, 4)
    flags: list = field(default_factory=list)   # 'stop', 'break', 'fermata', 'free', or free text
    start_sec: float | None = None              # filled by the recognizer (video sync)
    line: int = 0

    @property
    def free(self) -> bool:
        return "free" in self.flags


@dataclass
class Directive:
    kind: str           # 'key' | 'time' | 'bpm'
    value: str          # key text / '3/4' / '80'
    relative: int | None = None   # key: semitones relative to current key
    line: int = 0


@dataclass
class Section:
    name: str
    repeat: int = 1
    free: bool = False
    items: list = field(default_factory=list)   # Bar | Directive
    notes: list = field(default_factory=list)
    line: int = 0

    @property
    def type(self) -> str:
        return section_type(self.name)


@dataclass
class Chart:
    meta: dict = field(default_factory=dict)    # title, artist, url, key, bpm, time
    sections: list = field(default_factory=list)

    # ------------- convenience
    @property
    def key(self) -> Key | None:
        k = self.meta.get("key")
        return parse_key(k) if k else None

    @property
    def time(self) -> tuple:
        return tuple(self.meta.get("time", (4, 4)))

    def expand(self) -> list["ExpBar"]:
        return expand(self)

    def to_text(self) -> str:
        return render_text(self)

    def to_dict(self) -> dict:
        return chart_to_dict(self)


# ---------------------------------------------------------------- section names

_SECTION_PATTERNS = [
    ("prechorus", r"pre[\s-]?chorus|pc|pre|프리\s*코러스|프리\s*후렴|프리"),
    ("chorus", r"chorus|c|ch|refrain|후렴|코러스"),
    ("verse", r"verse|v|vs|절|벌스"),
    ("bridge", r"bridge|b|br|브릿지|브리지"),
    ("intro", r"intro|in|i|전주|인트로"),
    ("outro", r"outro|ending|end|o|coda|엔딩|후주|아웃트로"),
    ("interlude", r"interlude|inst|instrumental|간주|인터루드"),
    ("solo", r"solo|.*\s솔로|솔로|.*\ssolo"),
    ("tag", r"tag|태그"),
    ("break", r"break|브레이크"),
]


def section_type(name: str) -> str:
    core = re.sub(r"[\d.#]+", " ", name.lower())
    core = re.sub(r"\s+", " ", core).strip()
    for typ, pat in _SECTION_PATTERNS:
        if re.fullmatch(pat, core):
            return typ
    return "other"


# ---------------------------------------------------------------- text parser

_HEADER_KV = re.compile(r"^(title|artist|url|source|notes?|제목|가수|링크)\s*[:=]\s*(.*)$", re.I)
_KEY_IN_LINE = re.compile(r"\bkey\s*[:=]?\s*([A-G](?:#|b|♯|♭)?m?)(?![A-Za-z#])", re.I)
_BPM_IN_LINE = re.compile(r"\b(?:bpm|tempo)\s*[:=≈~]?\s*≈?\s*(\d+(?:\.\d+)?)", re.I)
_TIME_IN_LINE = re.compile(r"(?:\btime\s*[:=]?\s*)?\b(\d{1,2})\s*/\s*(\d{1,2})\b", re.I)
_SECTION_LINE = re.compile(r"^\[([^\]]+)\](.*)$")
_REPEAT = re.compile(r"[x×X*]\s*(\d+)|(\d+)\s*[x×X회번]")
_PAREN = re.compile(r"\(([^)]*)\)")
_KEY_DIRECTIVE = re.compile(r"^(?:key|전조|키)\b(.*)$", re.I)
_TIME_DIRECTIVE = re.compile(r"^(?:time|박자)\s*[:=]?\s*(\d{1,2})\s*/\s*(\d{1,2})\s*$", re.I)
_BPM_DIRECTIVE = re.compile(r"^(?:bpm|tempo|템포)\s*[:=≈~]?\s*(\d+(?:\.\d+)?)\s*$", re.I)
_BAR_PAREN = re.compile(r"(?:(?<=\s)|^)\(([^)]*)\)")
_CHORD_TOKEN = re.compile(r"^(>)?(.+?)(\?)?(?:@(\d+(?:\.\d+)?))?(\?)?$")
_NC = {"n.c.", "n.c", "nc", "n.c", "x"}
_FLAG_WORDS = {
    "stop": "stop", "스탑": "stop", "정지": "stop",
    "break": "break", "브레이크": "break",
    "fermata": "fermata", "늘임": "fermata", "페르마타": "fermata",
    "free": "free", "프리": "free", "rubato": "free", "루바토": "free", "자유": "free",
}
VALID_TIME_DEN = (2, 4, 8)


def _flag_from_note(note: str) -> str:
    return _FLAG_WORDS.get(note.strip().lower(), note.strip())


def _parse_time(n: str, d: str, line: int) -> tuple:
    n, d = int(n), int(d)
    if d not in VALID_TIME_DEN or n < 1:
        raise ChartError(f"박자표 {n}/{d} 는 지원하지 않습니다 (분모 2, 4, 8)", line)
    return (n, d)


def frames_per_beat(time: tuple) -> int:
    """Timeline resolution is one 8th note."""
    return 8 // time[1]


def bar_frames(time: tuple) -> int:
    return time[0] * frames_per_beat(time)


def parse_chart(text: str) -> Chart:
    chart = Chart()
    chart.meta = {"time": (4, 4)}
    current: Section | None = None
    cur_time = (4, 4)
    prev_bar: Bar | None = None
    seen_body = False
    pending: list = []   # directives waiting to see whether a new section starts next

    def add_directive(d):
        pending.append(d)

    def flush_pending(target):
        target.items.extend(pending)
        pending.clear()

    def ensure_section(lineno):
        nonlocal current
        if current is None:
            current = Section(name="", line=lineno)
            chart.sections.append(current)
        return current

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw).strip()
        if not line or line.startswith("//"):
            continue

        # --- bar line
        if line.startswith("|"):
            sec = ensure_section(lineno)
            flush_pending(sec)
            seen_body = True
            for seg in _split_bars(line):
                bar = _parse_bar(seg, cur_time, prev_bar, sec.free, lineno)
                sec.items.append(bar)
                prev_bar = bar
            continue

        # --- [Section] or [Key: Ab] / [Time 3/4]
        m = _SECTION_LINE.match(line)
        if m:
            inner, rest = m.group(1).strip(), m.group(2).strip()
            d = _parse_directive(inner, lineno)
            if d:
                if d.kind == "time":
                    cur_time = tuple(int(x) for x in d.value.split("/"))
                if not seen_body and d.kind != "key":
                    _apply_header_directive(chart, d)
                elif not seen_body and d.relative is None:
                    chart.meta["key"] = d.value
                else:
                    add_directive(d)
                continue
            seen_body = True
            sec = Section(name=inner, line=lineno)
            rm = _REPEAT.search(_PAREN.sub(" ", rest))
            if rm:
                sec.repeat = int(rm.group(1) or rm.group(2))
                if sec.repeat < 1:
                    raise ChartError("반복 횟수는 1 이상이어야 합니다", lineno)
            for note in _PAREN.findall(rest):
                if _flag_from_note(note) == "free":
                    sec.free = True
                else:
                    sec.notes.append(note.strip())
            if re.search(r"\bfree\b", _PAREN.sub(" ", rest), re.I):
                sec.free = True
            flush_pending(sec)
            chart.sections.append(sec)
            current = sec
            continue

        # --- header / mid-chart directive lines (Key: Ab, BPM: 80, 3/4, Title: ...)
        hm = _HEADER_KV.match(line)
        if hm:
            k = hm.group(1).lower()
            k = {"제목": "title", "가수": "artist", "링크": "url", "source": "url", "note": "notes"}.get(k, k)
            chart.meta[k] = hm.group(2).strip()
            continue
        found = False
        km = _KEY_IN_LINE.search(line)
        if km:
            found = True
            try:
                parse_key(km.group(1))
            except ChordParseError as e:
                raise ChartError(str(e), lineno)
            if seen_body:
                add_directive(Directive("key", km.group(1), line=lineno))
            else:
                chart.meta["key"] = km.group(1)
        bm = _BPM_IN_LINE.search(line)
        if bm:
            found = True
            if seen_body:
                add_directive(Directive("bpm", bm.group(1), line=lineno))
            else:
                chart.meta["bpm"] = float(bm.group(1))
                if re.search(r"가변|variable|rubato|흔들", line, re.I):
                    chart.meta["bpm_variable"] = True
        tm = _TIME_IN_LINE.search(_KEY_IN_LINE.sub(" ", line))
        if tm:
            found = True
            t = _parse_time(tm.group(1), tm.group(2), lineno)
            cur_time = t
            if seen_body:
                add_directive(Directive("time", f"{t[0]}/{t[1]}", line=lineno))
            else:
                chart.meta["time"] = t
        if not found:
            raise ChartError(f"이해할 수 없는 줄: {line!r} (마디는 '|'로 시작, 구간은 [이름], 메모는 '#' 뒤에)", lineno)

    if pending:
        raise ChartError("마지막 마디 뒤에 전조/박자 표기가 있습니다", pending[0].line)
    if not any(isinstance(it, Bar) for s in chart.sections for it in s.items):
        raise ChartError("마디가 하나도 없습니다")
    # drop an empty implicit leading section
    chart.sections = [s for s in chart.sections if s.items or s.name]
    return chart


def _strip_comment(line: str) -> str:
    """'#' starts a comment only at line start or after whitespace (F# stays a sharp)."""
    return re.sub(r"(^|\s)#.*$", "", line)


def _apply_header_directive(chart: Chart, d: Directive):
    if d.kind == "time":
        chart.meta["time"] = tuple(int(x) for x in d.value.split("/"))
    elif d.kind == "bpm":
        chart.meta["bpm"] = float(d.value)


def _parse_directive(inner: str, lineno: int) -> Directive | None:
    m = _KEY_DIRECTIVE.match(inner)
    if m:
        rest = m.group(1).strip()
        rel = re.search(r"([+-]\d+)\s*$", rest)
        if rel:
            return Directive("key", "", relative=int(rel.group(1)), line=lineno)
        km = re.search(r"([A-G](?:#|b|♯|♭)?m?)\s*$", rest)
        if not km:
            raise ChartError(
                f"전조 표기에 목표 키가 없습니다: [{inner}] → 예: [Key: Ab], [Key Up → Ab], [Key +1]", lineno)
        try:
            parse_key(km.group(1))
        except ChordParseError as e:
            raise ChartError(str(e), lineno)
        return Directive("key", km.group(1), line=lineno)
    m = _TIME_DIRECTIVE.match(inner)
    if m:
        t = _parse_time(m.group(1), m.group(2), lineno)
        return Directive("time", f"{t[0]}/{t[1]}", line=lineno)
    m = _BPM_DIRECTIVE.match(inner)
    if m:
        return Directive("bpm", m.group(1), line=lineno)
    return None


def _split_bars(line: str) -> list[str]:
    s = line.strip()
    s = s.replace("||", "|")
    parts = s.split("|")
    # leading '|' gives an empty first part; a trailing '|' an empty last part
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    for p in parts:
        if p.strip() == "":
            raise ChartError("빈 마디가 있습니다. 이전 코드를 유지하는 마디는 '%' 또는 '.' 로 적어주세요")
    return parts


def _parse_bar(seg: str, time: tuple, prev: Bar | None, section_free: bool, lineno: int) -> Bar:
    flags = []
    # inline time signature for a single bar: {3/4}
    tm = re.search(r"\{(\d{1,2})/(\d{1,2})\}", seg)
    if tm:
        time = _parse_time(tm.group(1), tm.group(2), lineno)
        seg = seg[:tm.start()] + seg[tm.end():]
    for note in _BAR_PAREN.findall(seg):
        flags.append(_flag_from_note(note))
    seg = _BAR_PAREN.sub(" ", seg)
    if section_free and "free" not in flags:
        flags.append("free")
    tokens = seg.split()
    if not tokens:
        raise ChartError("빈 마디가 있습니다", lineno)

    if tokens == ["%"]:
        if prev is None:
            raise ChartError("'%' 앞에 반복할 마디가 없습니다", lineno)
        events = [Event(e.beat, e.chord, e.push, e.conf) for e in prev.events]
        return Bar(events=events, time=time, flags=flags, line=lineno)

    n_beats = time[0]
    fpb = frames_per_beat(time)
    slots = []          # (chord|None, push, conf, explicit_beat|None, is_placeholder)
    for tok in tokens:
        if tok in (".", "/", "-", "·"):
            slots.append((None, False, None, None, True))
            continue
        if tok == ">":
            raise ChartError("'>'는 코드 바로 앞에 붙여 쓰세요 (예: >D)", lineno)
        m = _CHORD_TOKEN.match(tok)
        push, body, q1, beat, q2 = m.groups()
        conf = LOW_CONF if (q1 or q2) else None
        if body.lower() in _NC:
            chord = None
        else:
            try:
                chord = parse_chord(body)
            except ChordParseError as e:
                raise ChartError(str(e), lineno)
        slots.append((chord if body.lower() not in _NC else "NC", bool(push), conf,
                      float(beat) if beat else None, False))

    has_explicit = any(s[3] is not None for s in slots)
    events = []
    if has_explicit:
        if any(s[4] for s in slots):
            raise ChartError("'@박' 표기와 '.' 를 한 마디에 섞을 수 없습니다", lineno)
        for i, (c, push, conf, beat, _) in enumerate(slots):
            if beat is None:
                if i == 0:
                    beat = 1.0
                else:
                    raise ChartError("한 마디에서 '@박'을 쓰면 두 번째 코드부터 모두 '@박'을 붙여야 합니다", lineno)
            events.append(Event(beat, None if c == "NC" else c, push, conf))
    else:
        k = len(slots)
        if k == n_beats:
            step = 1.0
        elif n_beats % k == 0:
            step = n_beats / k
        else:
            raise ChartError(
                f"{time[0]}/{time[1]} 마디에 칸이 {k}개라 박 위치가 애매합니다. "
                f"'.'로 박을 채우거나(예: C . G A) '@박'을 쓰세요(예: C G@3 A@4)", lineno)
        for i, (c, push, conf, _, ph) in enumerate(slots):
            if ph:
                continue
            events.append(Event(1.0 + i * step, None if c == "NC" else c, push, conf))

    for e in events:
        off = (e.beat - 1) * fpb
        if e.beat < 1 or e.beat >= n_beats + 1 or abs(off - round(off)) > 1e-9:
            raise ChartError(f"박 위치 {e.beat:g} 가 {time[0]}/{time[1]} 마디에 맞지 않습니다", lineno)
    beats = [e.beat for e in events]
    if len(set(beats)) != len(beats):
        raise ChartError("같은 박에 코드가 두 개 있습니다", lineno)
    events.sort(key=lambda e: e.beat)
    return Bar(events=events, time=time, flags=flags, line=lineno)


# ---------------------------------------------------------------- expansion

@dataclass
class ExpBar:
    index: int
    section_instance: int
    section_name: str
    section_type: str
    key: Key | None
    time: tuple
    free: bool
    flags: list
    events: list
    start_sec: float | None = None


def expand(chart: Chart) -> list[ExpBar]:
    """Unroll repeats into the bar sequence actually played."""
    out = []
    key = chart.key
    inst = -1
    for sec in chart.sections:
        for _ in range(sec.repeat):
            inst += 1
            for it in sec.items:
                if isinstance(it, Directive):
                    if it.kind == "key":
                        if it.relative is not None:
                            if key is None:
                                raise ChartError("상대 전조([Key +1])를 쓰려면 처음에 Key: 가 있어야 합니다", it.line)
                            key = key.transpose(it.relative)
                        else:
                            key = parse_key(it.value)
                    continue
                out.append(ExpBar(
                    index=len(out), section_instance=inst, section_name=sec.name,
                    section_type=sec.type, key=key, time=it.time,
                    free=it.free or sec.free, flags=list(it.flags),
                    events=list(it.events), start_sec=it.start_sec))
    return out


# ---------------------------------------------------------------- timeline (8th-note frames)

@dataclass
class Timeline:
    bars: list
    bar_start: list         # first frame of each bar
    frames: list            # Chord | None per 8th note
    conf: list              # confidence per frame (None if unknown)

    @property
    def n_frames(self):
        return len(self.frames)

    def bar_of_frame(self, f: int) -> int:
        import bisect
        return bisect.bisect_right(self.bar_start, f) - 1

    def segments(self):
        """[(start, end, chord)] with consecutive identical chords merged."""
        segs = []
        for f, c in enumerate(self.frames):
            if segs and _same(segs[-1][2], c):
                segs[-1][1] = f + 1
            else:
                segs.append([f, f + 1, c])
        return [tuple(s) for s in segs]

    def changes(self) -> list[int]:
        return [s[0] for s in self.segments()[1:]]


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a == b


def build_timeline(bars: list[ExpBar]) -> Timeline:
    starts = []
    total = 0
    for b in bars:
        starts.append(total)
        total += bar_frames(b.time)
    placed = []   # (frame, order, chord, conf)
    for b, s in zip(bars, starts):
        fpb = frames_per_beat(b.time)
        for e in b.events:
            f = s + int(round((e.beat - 1) * fpb)) - (1 if e.push else 0)
            placed.append((max(f, 0), len(placed), e.chord, e.conf))
    placed.sort()
    frames = [None] * total
    conf = [None] * total
    for i, (f, _, c, cf) in enumerate(placed):
        end = placed[i + 1][0] if i + 1 < len(placed) else total
        for t in range(f, end):
            frames[t] = c
            conf[t] = cf
    return Timeline(bars=bars, bar_start=starts, frames=frames, conf=conf)


# ---------------------------------------------------------------- rendering

def _event_token(e: Event, beat: float | None = None) -> str:
    t = ("N.C." if e.chord is None else str(e.chord))
    if e.push:
        t = ">" + t
    if e.conf is not None and e.conf < 0.5:
        t += "?"
    if beat is not None:
        t += f"@{beat:g}"
    return t


def render_bar(bar: Bar) -> str:
    n = bar.time[0]
    ev = bar.events
    k = len(ev)
    notes = " ".join(f"({f})" for f in bar.flags if f != "free" or False)
    body = None
    if k == 0:
        body = "."
    elif all(abs(e.beat - (1 + i * n / k)) < 1e-9 for i, e in enumerate(ev)) and n % k == 0:
        body = " ".join(_event_token(e) for e in ev)
    elif all(float(e.beat).is_integer() for e in ev):
        slots = ["."] * n
        for e in ev:
            slots[int(e.beat) - 1] = _event_token(e)
        body = " ".join(slots)
    else:
        body = " ".join(_event_token(e, None if i == 0 and e.beat == 1 else e.beat)
                        for i, e in enumerate(ev))
    return (body + (" " + notes if notes else "")).strip()


def render_text(chart: Chart, bars_per_line: int = 4) -> str:
    m = chart.meta
    lines = []
    for k in ("title", "artist", "url"):
        if m.get(k):
            lines.append(f"{k.capitalize() if k != 'url' else 'URL'}: {m[k]}")
    head = []
    if m.get("key"):
        head.append(f"Key: {m['key']}")
    if m.get("bpm"):
        head.append(f"BPM: {m['bpm']:g}" + (" (가변)" if m.get("bpm_variable") else ""))
    t = tuple(m.get("time", (4, 4)))
    head.append(f"{t[0]}/{t[1]}")
    lines.append("   ".join(head))
    cur_time = t
    def directive_line(it):
        if it.kind == "key":
            return f"[Key {it.relative:+d}]" if it.relative is not None else f"[Key: {it.value}]"
        if it.kind == "time":
            return f"[Time {it.value}]"
        return f"[BPM {it.value}]"

    for sec in chart.sections:
        lines.append("")
        items = list(sec.items)
        while items and isinstance(items[0], Directive):
            d = items.pop(0)
            lines.append(directive_line(d))
            if d.kind == "time":
                cur_time = tuple(int(x) for x in d.value.split("/"))
        if sec.name:
            h = f"[{sec.name}]"
            if sec.repeat > 1:
                h += f" x{sec.repeat}"
            if sec.free:
                h += " (free)"
            for n in sec.notes:
                h += f" ({n})"
            lines.append(h)
        row = []
        for it in items:
            if isinstance(it, Directive):
                if row:
                    lines.append("| " + " | ".join(row) + " |")
                    row = []
                lines.append(directive_line(it))
                if it.kind == "time":
                    cur_time = tuple(int(x) for x in it.value.split("/"))
                continue
            tok = render_bar(Bar(it.events, it.time, [f for f in it.flags if not (sec.free and f == "free")]))
            if tuple(it.time) != tuple(cur_time):
                tok = f"{{{it.time[0]}/{it.time[1]}}} " + tok
            row.append(tok)
            if len(row) == bars_per_line:
                lines.append("| " + " | ".join(row) + " |")
                row = []
        if row:
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- JSON

def chart_to_dict(chart: Chart) -> dict:
    meta = dict(chart.meta)
    meta["time"] = list(meta.get("time", (4, 4)))
    secs = []
    for s in chart.sections:
        items = []
        for it in s.items:
            if isinstance(it, Directive):
                d = {"directive": it.kind, "value": it.value}
                if it.relative is not None:
                    d["relative"] = it.relative
                items.append(d)
            else:
                b = {"time": list(it.time),
                     "events": [{"beat": e.beat, "chord": e.text,
                                 **({"push": True} if e.push else {}),
                                 **({"conf": e.conf} if e.conf is not None else {})}
                                for e in it.events]}
                if it.flags:
                    b["flags"] = list(it.flags)
                if it.start_sec is not None:
                    b["start_sec"] = it.start_sec
                items.append(b)
        secs.append({"name": s.name, "repeat": s.repeat, "free": s.free,
                     "notes": s.notes, "items": items})
    return {"format": "chartgen/1", "meta": meta, "sections": secs}


def chart_from_dict(d: dict) -> Chart:
    meta = dict(d.get("meta", {}))
    meta["time"] = tuple(meta.get("time", (4, 4)))
    chart = Chart(meta=meta)
    for s in d["sections"]:
        sec = Section(name=s.get("name", ""), repeat=int(s.get("repeat", 1)),
                      free=bool(s.get("free", False)), notes=list(s.get("notes", [])))
        for it in s["items"]:
            if "directive" in it:
                sec.items.append(Directive(it["directive"], it.get("value", ""), it.get("relative")))
                continue
            events = []
            for e in it["events"]:
                txt = e["chord"]
                chord = None if txt.lower() in _NC else parse_chord(txt)
                events.append(Event(float(e["beat"]), chord, bool(e.get("push", False)), e.get("conf")))
            events.sort(key=lambda e: e.beat)
            flags = list(it.get("flags", []))
            sec.items.append(Bar(events=events, time=tuple(it.get("time", meta["time"])),
                                 flags=flags, start_sec=it.get("start_sec")))
        chart.sections.append(sec)
    return chart


def load_chart(path) -> Chart:
    path = str(path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        return chart_from_dict(json.loads(text))
    return parse_chart(text)
