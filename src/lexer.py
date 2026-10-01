from __future__ import annotations

from dataclasses import dataclass

KEYWORDS = {
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
    "goto", "if", "in", "local", "nil", "not", "or", "repeat", "return",
    "then", "true", "until", "while",
}

SYMBOLS = [
    "...", "..", ".",
    "::",
    "<=", ">=", "==", "~=",
    "<<", ">>",
    "//",
    "+", "-", "*", "/", "%", "^", "#",
    "&", "~", "|",
    "<", ">", "=",
    "(", ")", "{", "}", "[", "]",
    ";", ":", ",",
]


class LuaLexError(ValueError):
    pass


@dataclass
class Token:
    kind: str
    value: str
    pos: int
    line: int


def _long_bracket_level(source: str, i: int) -> int | None:
    if source[i] != "[":
        return None
    j = i + 1
    level = 0
    while j < len(source) and source[j] == "=":
        level += 1
        j += 1
    if j < len(source) and source[j] == "[":
        return level
    return None


def _read_long_bracket(source: str, i: int, level: int) -> tuple[str, int]:
    start = i + 2 + level
    if start < len(source) and source[start] == "\r":
        start += 1
    if start < len(source) and source[start] == "\n":
        start += 1
    closer = "]" + ("=" * level) + "]"
    end = source.find(closer, start)
    if end == -1:
        raise LuaLexError("Unclosed long bracket")
    return source[start:end], end + len(closer)


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(source)
    line = 1

    def advance_line_count(text: str) -> None:
        nonlocal line
        line += text.count("\n")

    while i < n:
        ch = source[i]

        if ch in " \t\r\n":
            if ch == "\n":
                line += 1
            i += 1
            continue

        if source.startswith("--", i):
            level = _long_bracket_level(source, i + 2)
            if level is not None:
                content, after = _read_long_bracket(source, i + 2, level)
                advance_line_count(content)
                i = after
                continue
            newline = source.find("\n", i)
            i = n if newline == -1 else newline
            continue

        level = _long_bracket_level(source, i)
        if level is not None:
            content, after = _read_long_bracket(source, i, level)
            tokens.append(Token("string", content, i, line))
            advance_line_count(source[i:after])
            i = after
            continue

        if ch in ('"', "'"):
            quote = ch
            j = i + 1
            buf: list[str] = []
            while j < n and source[j] != quote:
                if source[j] == "\\" and j + 1 < n:
                    buf.append(source[j])
                    buf.append(source[j + 1])
                    j += 2
                    continue
                if source[j] == "\n":
                    raise LuaLexError("Unterminated string literal")
                buf.append(source[j])
                j += 1
            if j >= n:
                raise LuaLexError("Unterminated string literal")
            raw = "".join(buf)
            tokens.append(Token("string", _decode_lua_string(raw), i, line))
            i = j + 1
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            j = i
            is_hex = False
            if source.startswith(("0x", "0X"), i):
                is_hex = True
                j += 2
                while j < n and (source[j] in "0123456789abcdefABCDEF."):
                    j += 1
                if j < n and source[j] in "pP":
                    j += 1
                    if j < n and source[j] in "+-":
                        j += 1
                    while j < n and source[j].isdigit():
                        j += 1
            else:
                while j < n and (source[j].isdigit() or source[j] == "."):
                    j += 1
                if j < n and source[j] in "eE":
                    j += 1
                    if j < n and source[j] in "+-":
                        j += 1
                    while j < n and source[j].isdigit():
                        j += 1
            tokens.append(Token("number", source[i:j], i, line))
            i = j
            continue

        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            word = source[i:j]
            kind = "keyword" if word in KEYWORDS else "name"
            tokens.append(Token(kind, word, i, line))
            i = j
            continue

        matched = False
        for sym in SYMBOLS:
            if source.startswith(sym, i):
                tokens.append(Token("symbol", sym, i, line))
                i += len(sym)
                matched = True
                break
        if matched:
            continue

        raise LuaLexError(f"Unexpected character {ch!r} at offset {i} (line {line})")

    tokens.append(Token("eof", "", n, line))
    return tokens


def _decode_lua_string(raw: str) -> str:
    out: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= n:
            break
        esc = raw[i]
        simple = {
            "a": "\a", "b": "\b", "f": "\f", "n": "\n", "r": "\r",
            "t": "\t", "v": "\v", "\\": "\\", '"': '"', "'": "'", "\n": "\n",
        }
        if esc in simple:
            out.append(simple[esc])
            i += 1
            continue
        if esc == "z":
            i += 1
            while i < n and raw[i] in " \t\r\n":
                i += 1
            continue
        if esc == "x":
            hex_digits = raw[i + 1 : i + 3]
            out.append(chr(int(hex_digits, 16)))
            i += 3
            continue
        if esc.isdigit():
            j = i
            digits = ""
            while j < n and raw[j].isdigit() and len(digits) < 3:
                digits += raw[j]
                j += 1
            out.append(chr(int(digits) % 256))
            i = j
            continue
        out.append(esc)
        i += 1
    return "".join(out)
