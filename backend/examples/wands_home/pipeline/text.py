# SPDX-License-Identifier: Apache-2.0

"""WANDS text as a storefront shows it.

WANDS ships its names, descriptions and feature values lowercased and split into tokens
(``it 's``, ``32 '' wide``, ``3 - light``, ``qt . slow cooker``), and part of it was read in
the wrong encoding on the way (``dã©cor``, ``itâ€™s``). This module puts the spacing around
punctuation back, restores the misread characters, and gives names title case and
descriptions sentence case. Wording changes in one way: a name loses the statements of a
length it makes and a description the sentences that make one (``without_lengths``,
``without_length_sentences``), because a product's width is a column the backend reads
and no service returns. Otherwise no word is added, dropped or respelt, so a proper noun
inside a description stays lowercase. Standard library only.
"""

from __future__ import annotations

import html
import re
import unicodedata

# What a character outside ASCII becomes; one not listed stays as it is (the letters with
# diacritics of "décor" and "brás", the degree and plus-minus signs, the fractions). The
# quotation marks become the tokenizer's own spelling, which the spacing repairs then read.
_CHARACTERS: dict[str, str] = {
    "\u2019": "'",  # right single quote: the apostrophe of most descriptions
    "\u00b4": "'",
    "\uff92": "'",
    "\u201c": " `` ",
    "\u201d": " '' ",
    '"': " '' ",
    "\u2013": "-",
    "\u2014": " - ",
    "\u2026": "... ",
    "\u00d7": " x ",
    "\u2044": "/",
    "\u2022": " ",
    "\u00ae": "",  # registered, trademark and copyright signs
    "\u2122": "",
    "\u00a9": "",
    "\u00ad": "",  # soft hyphen and the zero-width characters
    "\u200b": "",
    "\u200e": "",
    "\ufeff": "",
    "\u00ba": "\u00b0",  # a masculine ordinal or a superscript zero written for a degree sign
    "\u2070": "\u00b0",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\uff08": " ( ",  # full-width punctuation
    "\uff09": " ) ",
    "\uff0c": " , ",
    "\uff1a": " : ",
    "\uff1b": " ; ",
    "\uff01": " ! ",
    "\uff1f": " ? ",
    "\u3002": " . ",
}
_SUPERSCRIPTS = {
    chr(point): str(digit)
    for digit, point in enumerate((0xB9, 0xB2, 0xB3, *range(0x2074, 0x207A)), 1)
}
_SUBSCRIPTS = {chr(0x2080 + digit): str(digit) for digit in range(10)}
_LEFT_QUOTE = "\u2018"
_UNKNOWN = "\ufffd"
# Characters the source holds misread besides the ones above: they are restored and kept.
_KEPT = "éèêëáâäãåîïóòôöõúùûüñçøÉ°±½¼¾·" + _LEFT_QUOTE + _UNKNOWN


def _misread(text: str) -> str:
    """``text`` as it reads when its UTF-8 bytes are taken for Windows-1252."""
    out = []
    for byte in text.encode("utf-8"):
        try:
            out.append(bytes([byte]).decode("cp1252"))
        except UnicodeDecodeError:  # one of the five bytes Windows-1252 leaves undefined
            out.append(chr(byte))
    return "".join(out)


def _misreadings() -> list[tuple[re.Pattern[str], str]]:
    """A pattern for each restorable character read wrongly once or twice and then
    lowercased, longest first. The tokenizer split some of them, so whitespace may stand
    between the characters of one."""
    restorable = [*_CHARACTERS, *_SUPERSCRIPTS, *_SUBSCRIPTS, *_KEPT]
    forms: dict[str, str] = {}
    for character in restorable:
        if ord(character) < 0x80:
            continue
        once = _misread(character)
        for form in (_misread(once).lower(), once.lower()):
            if not any(ch.isspace() or ch == "\u00ad" for ch in form):
                forms.setdefault(form, character.lower())
    ordered = sorted(forms.items(), key=lambda item: (-len(item[0]), item[0]))
    return [(re.compile(r"\s*".join(re.escape(ch) for ch in form)), to) for form, to in ordered]


_MISREADINGS = _misreadings()
_TRANSLATION = str.maketrans(_CHARACTERS | _SUPERSCRIPTS | _SUBSCRIPTS)
_C1 = re.compile(r"[\u0080-\u009f]")
_ENTITY = re.compile(
    r"&\s*(#\s*\d{2,5}|nbsp|quot|amp|apos|deg|reg|trade|copy|ndash|mdash|lsquo|rsquo|ldquo|rdquo"
    r"|bull|hellip|times|frac12|frac14|frac34|eacute)\s*;"
)
_FRACTION = re.compile(
    r"([\u00b9\u00b2\u00b3\u2074-\u2079]+)[^\s\d\u2080-\u2089]{0,3}([\u2080-\u2089]+)"
)
# "manufacturer '' ™s warranty": an apostrophe that lost its first two bytes to a quotation mark.
_DAMAGED_APOSTROPHE = re.compile(r"\s*''\s*\u2122")
# "36-42 ’ ’ diameter": an inch mark typed as two apostrophes.
_DOUBLED_QUOTE = re.compile(r"\u2019\s*\u2019")
_SPACE = re.compile(r"\s+")

# The tokenizer's spacing, undone in this order.
_REPAIRS: tuple[tuple[re.Pattern[str], str], ...] = (
    # `` opens a quotation; after a number it is an inch mark the tokenizer took for one.
    (re.compile(r"(?<=\d) ?`` ?"), '" '),
    (re.compile(r"`` ?"), '"'),
    # "the artist '' s work": a possessive whose apostrophe the source doubled.
    (re.compile(r"(?<=[a-z]) ?'' s\b"), "'s"),
    # '' closes a quotation, and is how the tokenizer writes the inch mark.
    (re.compile(r" ?''"), '"'),
    (re.compile(r" ?`"), "'"),
    # Contractions and possessives: "do n't", "it 's", "camper ' s", "kids ' room"; the
    # last also closes a single quotation and writes the foot mark of "4 ' x 4 '".
    (re.compile(r" n't\b"), "n't"),
    (re.compile(r" '(s|re|ve|ll|d|m|t)\b"), r"'\1"),
    (re.compile(r"(?<=[a-z]) ' (s|t|re|ve|ll|d|m)\b"), r"'\1"),
    (re.compile(r"(?<=[\w.,!?]) '(?= |$)"), "'"),
    (re.compile(r" ([,;:!?%])"), r"\1"),
    # A period after a word: " .you can" lost the space on the wrong side.
    (re.compile(r"(?<=[^\s.!?]) \.(?=[a-z]{2})"), ". "),
    (re.compile(r"(?<=[^\s.!?]) \.(?!\d)"), "."),
    (re.compile(r"(?<=[!?]) \.(?= |$)"), ""),  # "collection ! ." ends twice
    (re.compile(r"([(\[]) "), r"\1"),
    (re.compile(r" ([)\]])"), r"\1"),
    # "3 - light" and "32 -oz" are one word; a hyphen between other words is a dash.
    (
        re.compile(
            r"(?<=\d) ?- ?(?=(?:light|bulb|drawer|piece|blade|person|door|player|hook|oz|panel"
            r"|shelf|tier)s?\b)"
        ),
        "-",
    ),
    (re.compile(r"# (?=\d)"), "#"),
    (re.compile(r"\$ (?=\d)"), "$"),
    (re.compile(r"(?<=\d) \u00b0"), "\u00b0"),  # "350 ° f"
    (re.compile(r"\u00b0 ?f\b"), "\u00b0F"),
    (re.compile(r"\u00b0 ?c\b"), "\u00b0C"),
)


def _c1(match: re.Match[str]) -> str:
    """A C1 control is a Windows-1252 byte that was never decoded: the character it stands
    for, or nothing for a byte Windows-1252 leaves undefined."""
    try:
        return bytes([ord(match.group(0))]).decode("cp1252")
    except UnicodeDecodeError:
        return ""


def _entity(match: re.Match[str]) -> str:
    return html.unescape("&" + _SPACE.sub("", match.group(1)) + ";")


def _fraction(match: re.Match[str]) -> str:
    return f"{match.group(1)}/{match.group(2)}".translate(_TRANSLATION)


def repair(raw: str | None) -> str:
    """The source's text with its characters restored and its punctuation attached again,
    still lowercase."""
    if not raw:
        return ""
    text = unicodedata.normalize("NFC", raw)
    for pattern, character in _MISREADINGS:
        text = pattern.sub(character, text)
    text = _ENTITY.sub(_entity, _C1.sub(_c1, text))
    text = _FRACTION.sub(_fraction, text)
    text = _DAMAGED_APOSTROPHE.sub("'", text)
    text = _DOUBLED_QUOTE.sub(" '' ", text)
    text = re.sub(_LEFT_QUOTE + r"\s*(?=\w)", "'", text).replace(_LEFT_QUOTE, "'")
    text = re.sub(r"(?<=[a-z])\u00e6s\b", "'s", text)  # "delphineæs" for "delphine's"
    text = text.replace(f"d{_UNKNOWN}cor", "d\u00e9cor").replace(_UNKNOWN, "")
    text = re.sub(r"\u00e2(?=\s|$)", "", text)  # what is left of a misread no-break space
    text = _SPACE.sub(" ", text.translate(_TRANSLATION))
    for pattern, replacement in _REPAIRS:
        text = pattern.sub(replacement, text)
    return _SPACE.sub(" ", text).strip()


# Words a title keeps lowercase unless it starts with them.
_SMALL = frozenset({"a", "an", "and", "at", "but", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"})  # fmt: skip
_UNITS = frozenset({"in", "ft", "qt", "qts", "cu", "oz", "lb", "lbs", "fl", "pcs", "sq", "cm", "mm", "gal"})  # fmt: skip
_UPPER = {
    "led": "LED", "tv": "TV", "tvs": "TVs", "usb": "USB", "cfm": "CFM", "pvc": "PVC",
    "gpm": "GPM", "usa": "USA", "bbq": "BBQ", "xl": "XL", "pu": "PU", "pe": "PE", "diy": "DIY",
    "rgb": "RGB", "ii": "II", "iii": "III", "iv": "IV", "ge": "GE", "rv": "RV", "uv": "UV",
    "mdf": "MDF", "lcd": "LCD", "nsf": "NSF", "npt": "NPT", "smd": "SMD", "pc": "PC",
}  # fmt: skip
_PARTS = re.compile(r"[^-/]+|[-/]")
_LETTERS = re.compile(r"[^\W\d_]+")
_OPENERS = frozenset('("[')


def _title_part(part: str, opening: bool) -> str:
    """One hyphen- or slash-separated part of a word in title case. A part that starts with
    a digit stays as the source has it ("6pcs", "420d"), except "3d"."""
    run = _LETTERS.search(part)
    if run is None:
        return part
    if any(ch.isdigit() for ch in part[: run.start()]):
        return "3D" + part[2:] if part[:2] == "3d" and not part[2:3].isalpha() else part
    word = run.group(0)
    if word in _UPPER:
        cased = _UPPER[word]
    elif (word in _SMALL or word in _UNITS) and not opening:
        cased = word
    else:
        cased = word[0].upper() + word[1:]
    return part[: run.start()] + cased + part[run.end() :]


def _title_word(word: str, opening: bool) -> str:
    if word == "x" or re.fullmatch(r"x\d.*", word):  # "18 x 13 in.", "x1.5"
        return word
    if word.startswith("w/"):
        return "w/" + _title_word(word[2:], True)
    parts = _PARTS.findall(word)
    return "".join(
        # The first part of a compound is capitalized ("In-Wall", "A-Frame"), a later one
        # follows the rule for small words ("Built-in", "4-in-1").
        _title_part(part, index == 0 and (opening or len(parts) > 1))
        for index, part in enumerate(parts)
    )


def title_case(text: str) -> str:
    """Title case for a product name: small words and units stay lowercase, the known
    initialisms go uppercase, a word led by a digit is left alone. A word opens a phrase
    at the start, after a dash or a colon, and when a bracket or a quotation mark leads it."""
    words = []
    opening = True
    for word in text.split(" "):
        words.append(_title_word(word, opening or word[:1] in _OPENERS))
        opening = word in {"-", "|"} or word.endswith(":")
    return " ".join(words)


# After one of these a period does not end a sentence.
_ABBREVIATIONS = frozenset(
    {
        "in", "ft", "lb", "lbs", "oz", "qt", "qts", "cu", "sq", "fl", "pc", "pcs", "approx",
        "vs", "dia", "etc", "eg", "ie", "inc", "incl", "co", "st", "dr", "mr", "mrs", "ms", "jr",
        "sr", "no", "max", "min",
    }
)  # fmt: skip
_BOUNDARY = re.compile(r"(?P<word>[^\W_]*)(?P<end>[.!?]+[\"')\]]*\s+[\"'(\[]*)(?P<letter>[a-z])")
_FIRST_LETTER = re.compile(r"^([\"'(\[]*)([a-z])")
_PRONOUN = re.compile(r"(?<![\w'.-])i(?=$|[\s,;:!?)]|'(?:m|ve|ll|d)\b)")


def _capitalize_after(match: re.Match[str]) -> str:
    word, end = match.group("word"), match.group("end")
    # "12 in. w. x 30 in. h." abbreviates; the F of "350 F." is in upper case by now.
    abbreviated = word.lower() in _ABBREVIATIONS or (len(word) == 1 and word.islower())
    if end.startswith("...") or (end.startswith(".") and abbreviated):
        return match.group(0)
    return word + end + match.group("letter").upper()


def sentence_case(text: str) -> str:
    """Sentence case for a description: the first letter of the text and of each sentence,
    and the pronoun "I"; a period after an abbreviation, a unit or a single letter does not
    end a sentence, and neither does an ellipsis."""
    text = _BOUNDARY.sub(_capitalize_after, text)
    text = _FIRST_LETTER.sub(lambda m: m.group(1) + m.group(2).upper(), text)
    return _PRONOUN.sub("I", text)


def shorten(text: str, limit: int, ellipsis: str = "...") -> str:
    """``text`` cut at a word boundary to at most ``limit`` characters, ``ellipsis`` included."""
    if len(text) <= limit:
        return text
    room = limit - len(ellipsis)
    cut = text[: room + 1].rsplit(" ", 1)[0] if " " in text[: room + 1] else text[:room]
    return cut.rstrip(" ,;:-(") + ellipsis


_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s+[\"'(\[]*[A-Z])")


def first_sentence(text: str) -> str:
    """The text up to the end of its first sentence; ``text`` is in sentence case, so a
    period followed by a lowercase word (an abbreviation's) does not end it."""
    match = _SENTENCE_END.search(text)
    return text[: match.end()] if match else text


# A statement of a length, as a name or a description makes one: a number (with a fraction,
# "94 1/2") and an inch or foot mark or a unit of length, with the side it measures when named
# ("47.2" w", "18 inches wide", "4 ft. h", "32 in"), or a number and a side word alone
# ("40 wide"); two or three of these joined by "x" or "by" are one statement ("18" h x 40" w
# x 16" d", "24 x 36", "47.3x23.6inch"), with the "up to" that may lead it ("tvs up to 60"")
# and the bracketed order of sides that may follow it ("(lxwxh)"). A bare "in" counts as an
# inch only before the end, a punctuation mark, an "x" or a side word ("3 in 1" and "seats 4
# in comfort" state no length), and "w", "h", "d" and "l" alone count as sides only after a
# unit or inside a statement ("60 w" is a wattage).
_NUMBER = r"(?:\d+(?:\.\d+)?(?:\s+\d+/\d+)?|\d+/\d+)"
_SIDE_WORD = r"(?:wide|width|high|height|deep|depth|long|length|tall)"
_UNIT = (
    r"(?:\"|''|\u201d|\u2033|'(?![a-z])"
    r"|\s*-?\s*(?:inches|inch|in\.|ft\.|ft|feet|foot|cm|mm)(?![a-z])"
    rf"|\s*-?\s*in(?=\s*(?:$|[,;:)!?]|x(?![a-z])|{_SIDE_WORD})))"
)
_SIDE = rf"(?:\s*-?\s*(?:{_SIDE_WORD}|[whdl])(?![a-z-]))"
_MEASURE = rf"(?:{_NUMBER}\s*{_UNIT}{_SIDE}?|{_NUMBER}\s+{_SIDE_WORD}(?![a-z-]))"
_ELEMENT = rf"(?:{_MEASURE}|{_NUMBER}(?:\s*[whdl](?![a-z-]))?)"
_LENGTH = re.compile(
    rf"(?:\bup to\s+)?(?:{_ELEMENT}(?:\s*(?:x|by)\s*{_ELEMENT})+(?:\s*{_UNIT})?{_SIDE}?|{_MEASURE})"
    r"(?:\s*\(\s*[lwhd](?:\s*x\s*[lwhd])+\s*\))?",
    re.IGNORECASE,
)
_EMPTIED = re.compile(r"\(\s*\)|\[\s*\]")
_BEFORE_PUNCTUATION = re.compile(r"\s+(?=[,;:.!?)\]])")
_AFTER_OPENER = re.compile(r"(?<=[(\[])\s+")
_DOUBLED = re.compile(r"([,;:])(?:\s*[,;:])+")


def states_length(text: str) -> bool:
    return _LENGTH.search(text) is not None


def without_lengths(text: str) -> str:
    """A name without the statements of a length it makes: "arlo 47.2" w height adjustable
    standing desk" becomes "arlo height adjustable standing desk"."""
    text = _EMPTIED.sub("", _LENGTH.sub(" ", text))
    text = _DOUBLED.sub(r"\1", _BEFORE_PUNCTUATION.sub("", _AFTER_OPENER.sub("", text)))
    return _SPACE.sub(" ", text).strip(" ,;:-/|")


def without_length_sentences(text: str) -> str:
    """A description in sentence case without the sentences that state a length."""
    kept, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        sentence = text[start : match.end()].strip()
        if sentence and not states_length(sentence):
            kept.append(sentence)
        start = match.end()
    last = text[start:].strip()
    if last and not states_length(last):
        kept.append(last)
    return " ".join(kept)


def clean_name(raw: str | None, limit: int = 200) -> str:
    return shorten(title_case(without_lengths(repair(raw))), limit, ellipsis="")


def clean_description(raw: str | None) -> str:
    return without_length_sentences(sentence_case(repair(raw)))


def capitalized(text: str) -> str:
    return text[:1].upper() + text[1:]


def clean_value(raw: str | None, limit: int = 120) -> str:
    """A feature's value as a specification shows it: repaired, cut to ``limit`` and with
    its first letter in upper case. An attribute keeps the repaired value as it is."""
    return capitalized(shorten(repair(raw), limit))
