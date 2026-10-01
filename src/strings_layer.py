from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from .ast_nodes import (
    BinOp, Block, Call, Do, Expr, FunctionDecl, FunctionExpr, GenericFor,
    If, IfClause, Index, LocalAssign, Name, Number, NumericFor, Return, Stat,
    String, Table, TableField, UnOp,
)
from .fold import _parse_number


@dataclass
class StringLayerReport:
    found: bool = False
    table_size: int = 0
    decoded_entries: int = 0
    resolved_calls: int = 0
    unresolved_calls: int = 0
    prologue_removed: bool = False
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class _Layer:
    block: Block
    start: int
    table_name: str
    resolver_name: str
    offset: int
    strings: list[Expr]
    reversals: list[tuple[int, int]]
    alphabets: dict[str, dict[str, int]]
    tags: dict[str, str]


def _number(expr: Expr) -> int | float | None:
    if isinstance(expr, Number):
        return _parse_number(expr.text)
    if isinstance(expr, UnOp) and expr.op == "-":
        inner = _number(expr.operand)
        return -inner if inner is not None else None
    return None


def _children(node: object):
    if not dataclasses.is_dataclass(node):
        return
    for item in dataclasses.fields(node):
        yield from _flatten(getattr(node, item.name))


def _flatten(value: object):
    if isinstance(value, list):
        for element in value:
            yield from _flatten(element)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        yield value


def _walk(node: object):
    if isinstance(node, list):
        for element in node:
            yield from _walk(element)
        return
    yield node
    for child in _children(node):
        yield from _walk(child)


def _all_blocks(root: Block):
    for node in _walk(root):
        if isinstance(node, Block):
            yield node


def _match_table(stat: Stat) -> tuple[str, list[Expr]] | None:
    if not isinstance(stat, LocalAssign):
        return None
    if len(stat.names) != 1 or len(stat.exprs) != 1:
        return None
    table = stat.exprs[0]
    if not isinstance(table, Table) or not table.fields:
        return None
    if any(f.key is not None for f in table.fields):
        return None
    values = [f.value for f in table.fields]
    if not any(isinstance(v, String) for v in values):
        return None
    return stat.names[0], values


def _match_resolver(stat: Stat, table_name: str) -> tuple[str, int] | None:
    if not isinstance(stat, FunctionDecl) or not stat.is_local:
        return None
    if not isinstance(stat.target, Name) or len(stat.params) != 1:
        return None
    if len(stat.body.stats) != 1 or not isinstance(stat.body.stats[0], Return):
        return None
    exprs = stat.body.stats[0].exprs
    if len(exprs) != 1 or not isinstance(exprs[0], Index):
        return None
    index = exprs[0]
    if not isinstance(index.obj, Name) or index.obj.name != table_name:
        return None
    key = index.key
    if not isinstance(key, BinOp) or key.op not in ("-", "+"):
        return None
    if not isinstance(key.left, Name) or key.left.name != stat.params[0]:
        return None
    amount = _number(key.right)
    if not isinstance(amount, int):
        return None
    offset = amount if key.op == "-" else -amount
    return stat.target.name, offset


def _match_reversals(stat: Stat) -> list[tuple[int, int]] | None:
    if not isinstance(stat, GenericFor) or len(stat.exprs) != 1:
        return None
    call = stat.exprs[0]
    if not isinstance(call, Call) or not isinstance(call.func, Name):
        return None
    if call.func.name != "ipairs" or len(call.args) != 1:
        return None
    table = call.args[0]
    if not isinstance(table, Table) or not table.fields:
        return None
    pairs: list[tuple[int, int]] = []
    for item in table.fields:
        inner = item.value
        if item.key is not None or not isinstance(inner, Table):
            return None
        if len(inner.fields) != 2:
            return None
        low = _number(inner.fields[0].value)
        high = _number(inner.fields[1].value)
        if not isinstance(low, int) or not isinstance(high, int):
            return None
        pairs.append((low, high))
    return pairs


def _key_text(key: Expr | None) -> str | None:
    if isinstance(key, String):
        return key.value
    return None


def _alphabet_from(expr: Expr) -> dict[str, int] | None:
    if not isinstance(expr, Table) or not expr.fields:
        return None
    result: dict[str, int] = {}
    for item in expr.fields:
        key = _key_text(item.key)
        value = _number(item.value)
        if key is None or not isinstance(value, int):
            return None
        result[key] = value
    return result


def _clause_kind(clause: IfClause) -> tuple[str, str] | None:
    cond = clause.cond
    if not isinstance(cond, BinOp) or cond.op != "==":
        return None
    literal = cond.right if isinstance(cond.right, String) else cond.left
    if not isinstance(literal, String) or len(literal.value) != 1:
        return None
    numbers = {_number(item) for item in _walk(clause.body) if isinstance(item, Number)}
    if 85 in numbers:
        return literal.value, "b85"
    if 64 in numbers or 65536 in numbers:
        return literal.value, "b64"
    return None


def _match_decoder(stat: Stat) -> tuple[dict[str, dict[str, int]], dict[str, str]] | None:
    if not isinstance(stat, Do):
        return None
    alphabets: dict[str, dict[str, int]] = {}
    for inner in stat.body.stats:
        if isinstance(inner, LocalAssign) and len(inner.names) == 1 and len(inner.exprs) == 1:
            table = _alphabet_from(inner.exprs[0])
            if table is not None and len(table) in (64, 85):
                alphabets["b64" if len(table) == 64 else "b85"] = table
    if len(alphabets) != 2:
        return None
    tags: dict[str, str] = {}

    def scan(node: object) -> None:
        if isinstance(node, If):
            for clause in node.clauses:
                kind = _clause_kind(clause)
                if kind is not None:
                    tags[kind[0]] = kind[1]
                else:
                    scan(clause.body)
            if node.orelse is not None:
                scan(node.orelse)
            return
        for child in _children(node):
            scan(child)

    scan(stat.body)
    if set(tags.values()) != {"b64", "b85"}:
        return None
    return alphabets, tags


def _find_layer(root: Block) -> _Layer | None:
    for block in _all_blocks(root):
        stats = block.stats
        for i in range(len(stats) - 3):
            table = _match_table(stats[i])
            if table is None:
                continue
            resolver = _match_resolver(stats[i + 1], table[0])
            if resolver is None:
                continue
            reversals = _match_reversals(stats[i + 2])
            if reversals is None:
                continue
            decoder = _match_decoder(stats[i + 3])
            if decoder is None:
                continue
            return _Layer(
                block, i, table[0], resolver[0], resolver[1], table[1],
                reversals, decoder[0], decoder[1],
            )
    return None


def _decode_b64(payload: str, alphabet: dict[str, int]) -> str:
    out: list[str] = []
    length = len(payload)
    acc = 0
    count = 0
    position = 1
    while position <= length:
        char = payload[position - 1]
        value = alphabet.get(char)
        if value is not None:
            acc += value * 64 ** (3 - count)
            count += 1
            if count == 4:
                count = 0
                out.append(chr(acc // 65536))
                out.append(chr((acc % 65536) // 256))
                out.append(chr(acc % 256))
                acc = 0
        elif char == "=":
            out.append(chr(acc // 65536))
            if position >= length or payload[position] != "=":
                out.append(chr((acc % 65536) // 256))
            break
        position += 1
    return "".join(out)


def _decode_b85(payload: str, alphabet: dict[str, int]) -> str:
    out: list[str] = []
    length = len(payload)
    position = 1
    while position <= length:
        remaining = length - position + 1
        take = 5 if remaining >= 5 else remaining
        acc = 0
        valid = take > 1
        for step in range(5):
            if step < take:
                value = alphabet.get(payload[position + step - 1])
                if value is None:
                    valid = False
                    break
            else:
                value = 84
            acc = acc * 85 + value
        if valid:
            chunk = [
                chr((acc // 16777216) % 256),
                chr((acc // 65536) % 256),
                chr((acc // 256) % 256),
                chr(acc % 256),
            ]
            if take >= 2:
                out.extend(chunk[: take - 1])
        position += take
    return "".join(out)


def _decode_table(layer: _Layer) -> tuple[list[Expr], int]:
    values = list(layer.strings)
    for low, high in layer.reversals:
        if 1 <= low <= high <= len(values):
            values[low - 1:high] = reversed(values[low - 1:high])
    decoded = 0
    result: list[Expr] = []
    for item in values:
        if isinstance(item, String) and item.value:
            kind = layer.tags.get(item.value[0])
            if kind is not None:
                payload = item.value[1:]
                text = (
                    _decode_b64(payload, layer.alphabets["b64"])
                    if kind == "b64"
                    else _decode_b85(payload, layer.alphabets["b85"])
                )
                result.append(String(text))
                decoded += 1
                continue
        result.append(item)
    return result, decoded


class _Rewriter:
    def __init__(self, resolver: str, table: str, offset: int, values: list[Expr]) -> None:
        self.resolver = resolver
        self.table = table
        self.offset = offset
        self.values = values
        self.resolved = 0
        self.unresolved = 0
        self.references = 0

    def block(self, block: Block, shadow: frozenset[str]) -> Block:
        return Block(self.stats(block.stats, shadow))

    def stats(self, stats: list[Stat], shadow: frozenset[str]) -> list[Stat]:
        current = shadow
        result: list[Stat] = []
        for stat in stats:
            result.append(self.stat(stat, current))
            if isinstance(stat, LocalAssign):
                current = current | frozenset(stat.names)
            elif isinstance(stat, FunctionDecl) and stat.is_local and isinstance(stat.target, Name):
                current = current | {stat.target.name}
        return result

    def stat(self, stat: Stat, shadow: frozenset[str]) -> Stat:
        if isinstance(stat, FunctionDecl):
            inner = shadow | frozenset(stat.params)
            if stat.is_local and isinstance(stat.target, Name):
                inner = inner | {stat.target.name}
            target = stat.target if stat.is_local else self.value(stat.target, shadow)
            return FunctionDecl(
                target, stat.is_local, stat.params, stat.is_vararg,
                self.block(stat.body, inner),
            )
        if isinstance(stat, NumericFor):
            return NumericFor(
                stat.var,
                self.value(stat.start, shadow),
                self.value(stat.stop, shadow),
                self.value(stat.step, shadow) if stat.step is not None else None,
                self.block(stat.body, shadow | {stat.var}),
            )
        if isinstance(stat, GenericFor):
            return GenericFor(
                stat.names,
                [self.value(e, shadow) for e in stat.exprs],
                self.block(stat.body, shadow | frozenset(stat.names)),
            )
        return self.generic(stat, shadow)

    def generic(self, node: object, shadow: frozenset[str]):
        values = {
            item.name: self.value(getattr(node, item.name), shadow)
            for item in dataclasses.fields(node)
        }
        return type(node)(**values)

    def value(self, value: object, shadow: frozenset[str]):
        if isinstance(value, list):
            return [self.value(element, shadow) for element in value]
        if isinstance(value, Block):
            return self.block(value, shadow)
        if isinstance(value, FunctionExpr):
            return FunctionExpr(
                value.params, value.is_vararg,
                self.block(value.body, shadow | frozenset(value.params)),
            )
        if isinstance(value, Call):
            replaced = self.call(value, shadow)
            if replaced is not None:
                return replaced
            return self.generic(value, shadow)
        if isinstance(value, Name):
            if value.name in (self.resolver, self.table) and value.name not in shadow:
                self.references += 1
            return value
        if isinstance(value, (Expr, TableField, IfClause)):
            return self.generic(value, shadow)
        return value

    def call(self, node: Call, shadow: frozenset[str]) -> Expr | None:
        func = node.func
        if not isinstance(func, Name) or func.name != self.resolver:
            return None
        if func.name in shadow:
            return None
        if len(node.args) != 1:
            self.unresolved += 1
            return None
        number = _number(node.args[0])
        if not isinstance(number, int):
            self.unresolved += 1
            return None
        position = number - self.offset
        if not 1 <= position <= len(self.values):
            self.unresolved += 1
            return None
        target = self.values[position - 1]
        if not isinstance(target, String):
            self.unresolved += 1
            return None
        self.resolved += 1
        return String(target.value)


def decode_string_layer(root: Block) -> StringLayerReport:
    report = StringLayerReport()
    layer = _find_layer(root)
    if layer is None:
        return report
    report.found = True
    report.table_size = len(layer.strings)
    report.tags = dict(layer.tags)
    values, decoded = _decode_table(layer)
    report.decoded_entries = decoded
    rewriter = _Rewriter(layer.resolver_name, layer.table_name, layer.offset, values)
    tail = layer.block.stats[layer.start + 4:]
    new_tail = rewriter.stats(tail, frozenset())
    head = layer.block.stats[:layer.start]
    prologue = layer.block.stats[layer.start:layer.start + 4]
    report.resolved_calls = rewriter.resolved
    report.unresolved_calls = rewriter.unresolved
    if rewriter.references == 0:
        layer.block.stats = head + new_tail
        report.prologue_removed = True
    else:
        layer.block.stats = head + prologue + new_tail
    return report
