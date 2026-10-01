import ast
import re


class WeAreDevsFormatError(ValueError):
    pass


def _constant_number(expression: str) -> int:
    expression = expression.strip().replace("^", "**")
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.Pow, ast.FloorDiv, ast.Div)):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            return int(left / right)
        raise WeAreDevsFormatError(f"Non-constant arithmetic: {expression!r}")

    return visit(tree.body)


def _balanced(text: str, start: int, opening: str = "{", closing: str = "}") -> tuple[str, int]:
    depth = 0
    quote = ""
    i = start
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ('"', "'"):
            quote = ch
        elif ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    raise WeAreDevsFormatError("Unclosed Lua table")


def _split_items(text: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth = 0
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in ('"', "'"):
            quote = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        elif ch in ",;" and depth == 0:
            if text[start:i].strip():
                items.append(text[start:i].strip())
            start = i + 1
        i += 1
    if text[start:].strip():
        items.append(text[start:].strip())
    return items


def _lua_string(value: str) -> str:
    quote = value[0]
    body = value[1:-1]
    body = body.replace("\\" + quote, quote).replace("\\'", "'").replace("\\\\", "\\")
    body = re.sub(r"\\(\d{1,3})", lambda match: chr(int(match.group(1))), body)
    return body


def _parse_string_table(text: str, variable: str) -> tuple[list[str], int]:
    match = re.search(rf"local\s+{re.escape(variable)}\s*=\s*\{{", text)
    if not match:
        raise WeAreDevsFormatError("Encoded string table was not found")
    body, end = _balanced(text, match.end() - 1)
    values = []
    for item in _split_items(body):
        if not (item.startswith(('"', "'")) and item.endswith(("\"", "'"))):
            raise WeAreDevsFormatError("String table contains a non-string value")
        values.append(_lua_string(item))
    return values, end


def _apply_permutations(text: str, values: list[str], start: int) -> None:
    match = re.search(r"ipairs\s*\(\s*\{", text[start:])
    if not match:
        return
    absolute = start + match.end() - 1
    body, _ = _balanced(text, absolute)
    for pair in re.findall(r"\{([^{}]+)\}", body):
        numbers = _split_items(pair)
        if len(numbers) != 2:
            continue
        left = _constant_number(numbers[0])
        right = _constant_number(numbers[1])
        while left < right:
            values[left - 1], values[right - 1] = values[right - 1], values[left - 1]
            left += 1
            right -= 1


def _parse_map(text: str, name: str, start: int) -> dict[str, int]:
    match = re.search(rf"local\s+{re.escape(name)}\s*=\s*\{{", text[start:])
    if not match:
        return {}
    absolute = start + match.end() - 1
    body, _ = _balanced(text, absolute)
    result: dict[str, int] = {}
    for item in _split_items(body):
        separator = _map_separator(item)
        if separator == -1:
            continue
        key, value = item[:separator], item[separator + 1 :]
        key = key.strip()
        if key.startswith("[") and key.endswith("]"):
            key = key[1:-1].strip()
            if key.startswith(('"', "'")):
                key = _lua_string(key)
        result[key] = _constant_number(value)
    return result


def _map_separator(item: str) -> int:
    depth = 0
    quote = ""
    i = 0
    while i < len(item):
        ch = item[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in ('"', "'"):
            quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0:
            return i
        i += 1
    return -1


def _decode_base85(value: str, mapping: dict[str, int]) -> str:
    if len(value) < 1:
        return value
    payload = value[1:]
    output: list[str] = []
    for offset in range(0, len(payload), 5):
        chunk = payload[offset : offset + 5]
        count = len(chunk)
        if count <= 1:
            break
        try:
            packed = 0
            for index in range(5):
                code = mapping[chunk[index]] if index < count else 84
                packed = packed * 85 + code
            byte_count = count - 1
            shifts = (24, 16, 8, 0)
            output.extend(chr((packed >> shift) & 255) for shift in shifts[:byte_count])
        except (KeyError, ValueError):
            return value
    return "".join(output)


def _lua_literal(value: str) -> str:
    escaped: list[str] = []
    for char in value:
        code = ord(char)
        if 32 <= code <= 126 and char not in ('"', "\\"):
            escaped.append(char)
        elif char == '"':
            escaped.append('\\"')
        elif char == "\\":
            escaped.append("\\\\")
        else:
            escaped.append(f"\\{code:03d}")
    return '"' + "".join(escaped) + '"'


def _find_resolver(source: str, table_name: str) -> tuple[str, int] | None:
    match = re.search(
        rf"local\s+function\s+([A-Za-z_]\w*)\s*\(\s*[A-Za-z_]\w*\s*\)\s*return\s+{re.escape(table_name)}\s*\[[A-Za-z_]\w*\s*([+-][^\]]+)\]\s*end",
        source,
    )
    if not match:
        return None
    try:
        return match.group(1), _constant_number(match.group(2))
    except (SyntaxError, ValueError, WeAreDevsFormatError):
        return None


def _replace_resolver_calls(source: str, resolver: str, values: list[str], offset: int) -> str:
    output: list[str] = []
    i = 0
    while i < len(source):
        if source[i] in ('"', "'"):
            quote = source[i]
            j = i + 1
            while j < len(source):
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == quote:
                    j += 1
                    break
                j += 1
            output.append(source[i:j])
            i = j
            continue

        match = re.match(rf"{re.escape(resolver)}\s*\(", source[i:])
        if not match:
            output.append(source[i])
            i += 1
            continue

        start = i + match.end()
        depth = 1
        j = start
        while j < len(source) and depth:
            if source[j] == "(":
                depth += 1
            elif source[j] == ")":
                depth -= 1
            j += 1
        if depth:
            output.append(source[i])
            i += 1
            continue

        expression = source[start : j - 1].strip()
        try:
            index = _constant_number(expression) + offset
            replacement = _lua_literal(values[index - 1]) if 1 <= index <= len(values) else None
        except (SyntaxError, ValueError, WeAreDevsFormatError):
            replacement = None
        if replacement is None:
            output.append(source[i:j])
        else:
            output.append(replacement)
        i = j
    return "".join(output)


_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_BINARY_CONSTANT = re.compile(
    rf"(?<![\w.])(?P<left>-?{_NUMBER})\s*(?P<op>[+\-*/%])\s*(?P<right>-?{_NUMBER})(?![\w.])"
)
_PAREN_CONSTANT = re.compile(rf"\((?P<expr>-?{_NUMBER}(?:\s*[+\-*/%]\s*-?{_NUMBER})+)\)")


def _fold_numeric_segment(segment: str) -> str:
    for _ in range(32):
        previous = segment

        def replace_parenthesized(match: re.Match[str]) -> str:
            try:
                return str(_constant_number(match.group("expr")))
            except (SyntaxError, ValueError, WeAreDevsFormatError, ZeroDivisionError):
                return match.group(0)

        segment = _PAREN_CONSTANT.sub(replace_parenthesized, segment)

        def replace_binary(match: re.Match[str]) -> str:
            try:
                expression = f"{match.group('left')}{match.group('op')}{match.group('right')}"
                return str(_constant_number(expression))
            except (SyntaxError, ValueError, WeAreDevsFormatError, ZeroDivisionError):
                return match.group(0)

        segment = _BINARY_CONSTANT.sub(replace_binary, segment)
        if segment == previous:
            break
    return segment


def _fold_numeric_constants(source: str) -> str:
    output: list[str] = []
    start = 0
    i = 0
    while i < len(source):
        if source.startswith("--", i):
            output.append(_fold_numeric_segment(source[start:i]))
            newline = source.find("\n", i)
            if newline == -1:
                output.append(source[i:])
                return "".join(output)
            output.append(source[i:newline])
            i = newline
            start = i
            continue
        if source[i] in ('"', "'"):
            output.append(_fold_numeric_segment(source[start:i]))
            quote = source[i]
            j = i + 1
            while j < len(source):
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == quote:
                    j += 1
                    break
                j += 1
            output.append(source[i:j])
            i = j
            start = j
            continue
        i += 1
    output.append(_fold_numeric_segment(source[start:]))
    return "".join(output)


def deobfuscate_wearedevs(source: str) -> str:
    match = re.search(r"local\s+([A-Za-z_]\w*)\s*=\s*\{", source)
    if not match:
        raise WeAreDevsFormatError("wearedevs string table was not found")
    variable = match.group(1)
    values, end = _parse_string_table(source, variable)
    _apply_permutations(source, values, end)
    prefix = values[0][:1] if values else ""
    mapping_name = "o" if prefix == "?" else "T"
    mapping = _parse_map(source, mapping_name, end)
    decoded = [_decode_base85(value, mapping) for value in values]
    table = "-- decoded_strings (static layer)\n" + "\n".join(
        f"-- [{index}] = {_lua_literal(value)}" for index, value in enumerate(decoded, 1)
    )
    expanded = source
    resolver_info = _find_resolver(source, variable)
    if resolver_info:
        resolver, offset = resolver_info
        expanded = _replace_resolver_calls(expanded, resolver, decoded, offset)
    expanded = _fold_numeric_constants(expanded)
    return table + "\n\n" + expanded
