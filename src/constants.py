from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from .ast_nodes import (
    Assign, BinOp, Block, Call, Field, Index, LocalAssign, MethodCall, Name, Number,
    Paren, String,
)
from .fold import _parse_number
from .strings_layer import _walk

MODULUS_45 = 35184372088832
_KEYWORDS = {
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "if", "in",
    "local", "nil", "not", "or", "repeat", "return", "then", "true", "until", "while", "continue",
}


@dataclass
class CipherParams:
    mul_45: int
    add_45: int
    mul_8: int
    initial: int


@dataclass
class DecryptReport:
    params: CipherParams | None = None
    decrypted: int = 0
    rejected: int = 0
    fields: int = 0


def _strip(expr):
    while isinstance(expr, Paren):
        expr = expr.inner
    return expr


def _int(expr) -> int | None:
    expr = _strip(expr)
    if isinstance(expr, Number):
        value = _parse_number(expr.text)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, int):
            return value
    return None


def _find_keystream(functions) -> tuple[int, int, int] | None:
    mul_45 = add_45 = mul_8 = None
    for node in _walk(functions):
        if not isinstance(node, BinOp) or node.op != "%":
            continue
        modulus = _int(node.right)
        left = _strip(node.left)
        if modulus == MODULUS_45 and isinstance(left, BinOp) and left.op == "+":
            product = _strip(left.left)
            if isinstance(product, BinOp) and product.op == "*":
                a = _int(product.right)
                b = _int(left.right)
                if a is not None and b is not None:
                    mul_45, add_45 = a, b
        elif modulus == 257 and isinstance(left, BinOp) and left.op == "*":
            c = _int(left.right)
            if c is not None:
                mul_8 = c
    if mul_45 is None or add_45 is None or mul_8 is None:
        return None
    return mul_45, add_45, mul_8


def _find_initial(functions) -> int | None:
    for stats in functions:
        for node in _walk(stats):
            if not (isinstance(node, Assign) and len(node.targets) == 1 and len(node.exprs) == 1):
                continue
            target, expr = node.targets[0], node.exprs[0]
            if not isinstance(target, Name) or not (isinstance(expr, BinOp) and expr.op == "%"):
                continue
            if _int(expr.right) != 256:
                continue
            inner = _strip(expr.left)
            if not (isinstance(inner, BinOp) and inner.op == "+" and isinstance(_strip(inner.right), Name)):
                continue
            if _strip(inner.right).name != target.name:
                continue
            calls = [
                item for item in _walk(inner)
                if isinstance(item, Call)
                and ((isinstance(item.func, Index) and isinstance(item.func.key, String) and item.func.key.value == "byte")
                     or (isinstance(item.func, Field) and item.func.name == "byte"))
            ]
            if not calls:
                continue
            values = [
                _int(item.exprs[0]) for item in _walk(stats)
                if isinstance(item, Assign) and len(item.targets) == 1
                and isinstance(item.targets[0], Name) and item.targets[0].name == target.name
                and len(item.exprs) == 1 and _int(item.exprs[0]) is not None
            ]
            if len(values) == 1:
                return values[0]
    return None


def extract_params(function_stats: list) -> CipherParams | None:
    keystream = _find_keystream(function_stats)
    initial = _find_initial(function_stats)
    if keystream is None or initial is None:
        return None
    return CipherParams(keystream[0], keystream[1], keystream[2], initial)


def decrypt(payload: str, seed: int, params: CipherParams) -> str:
    state_45 = seed % MODULUS_45
    state_8 = seed % 255 + 2
    cache: list[int] = []

    def next_byte() -> int:
        nonlocal state_45, state_8, cache
        if not cache:
            state_45 = (state_45 * params.mul_45 + params.add_45) % MODULUS_45
            while True:
                state_8 = state_8 * params.mul_8 % 257
                if state_8 != 1:
                    break
            shift_low = state_8 % 32
            exponent = 13 - (state_8 - shift_low) // 32
            n = (math.floor(state_45 / 2.0 ** exponent) % 4294967296) / 2.0 ** shift_low
            rnd = math.floor((n % 1) * 4294967296) + math.floor(n)
            low = rnd % 65536
            high = (rnd - low) // 65536
            b1 = low % 256
            b3 = high % 256
            cache = [b1, (low - b1) // 256, b3, (high - b3) // 256]
        return cache.pop()

    previous = params.initial
    out: list[str] = []
    for char in payload:
        previous = (ord(char) + next_byte() + previous) % 256
        out.append(chr(previous))
    return "".join(out)


def _printable(text: str) -> bool:
    return all(32 <= ord(c) <= 126 or c in "\n\r\t" for c in text)


def _looks_like_decryptor(func) -> bool:
    return isinstance(func, (Name, Index))


class _Decryptor:
    def __init__(self, params: CipherParams) -> None:
        self.params = params
        self.report = DecryptReport(params=params)
        self.table: dict[int, str] = {}

    def visit(self, node):
        if isinstance(node, list):
            return [self.visit(item) for item in node]
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        rebuilt = type(node)(**{
            f.name: self.visit(getattr(node, f.name)) for f in dataclasses.fields(node)
        })
        if isinstance(rebuilt, Call):
            args = rebuilt.args
            if (
                len(args) == 2 and isinstance(args[0], String) and _int(args[1]) is not None
                and _looks_like_decryptor(rebuilt.func) and _int(args[1]) >= 2 ** 30
            ):
                seed = _int(args[1])
                text = decrypt(args[0].value, seed, self.params)
                if _printable(text):
                    self.table[seed] = text
                    return Number(str(seed))
                self.report.rejected += 1
            return rebuilt
        if isinstance(rebuilt, Index):
            key = rebuilt.key
            seed = _int(key)
            if seed is not None and seed in self.table:
                self.report.decrypted += 1
                return String(self.table[seed])
        return rebuilt

    def resolve_registers(self, stats: list) -> list:
        assigned: dict[str, list] = {}
        for item in _walk(stats):
            if isinstance(item, Assign):
                for position, target in enumerate(item.targets):
                    if isinstance(target, Name):
                        expr = item.exprs[position] if position < len(item.exprs) and len(item.targets) == len(item.exprs) else None
                        assigned.setdefault(target.name, []).append(expr)
            elif isinstance(item, LocalAssign):
                for name in item.names:
                    assigned.setdefault(name, []).append(None)
        keyed = {}
        for name, values in assigned.items():
            if len(values) == 1 and values[0] is not None:
                seed = _int(values[0])
                if seed is not None and seed in self.table:
                    keyed[name] = seed
        if not keyed:
            return stats
        return self._replace(stats, keyed)

    def _replace(self, node, keyed: dict[str, int]):
        if isinstance(node, list):
            out = []
            for item in node:
                if (
                    isinstance(item, Assign) and len(item.targets) == 1 and len(item.exprs) == 1
                    and isinstance(item.targets[0], Name) and item.targets[0].name in keyed
                ):
                    continue
                out.append(self._replace(item, keyed))
            return out
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        rebuilt = type(node)(**{
            f.name: self._replace(getattr(node, f.name), keyed) for f in dataclasses.fields(node)
        })
        if isinstance(rebuilt, Index) and isinstance(rebuilt.key, Name) and rebuilt.key.name in keyed:
            self.report.decrypted += 1
            return String(self.table[keyed[rebuilt.key.name]])
        return rebuilt

    def local_pass(self, stats: list) -> list:
        return self._dead_keys(self._block(stats, {}), True)

    def _reads(self, stat, name: str) -> bool:
        total = sum(1 for item in _walk(stat) if isinstance(item, Name) and item.name == name)
        writes = sum(
            1 for item in _walk(stat) if isinstance(item, Assign)
            for target in item.targets if isinstance(target, Name) and target.name == name
        )
        return total - writes > 0

    def _dead_keys(self, stats: list, top: bool) -> list:
        stats = [self._dead_in(stat) for stat in stats]
        keep = [True] * len(stats)
        for i, stat in enumerate(stats):
            if not (
                isinstance(stat, Assign) and len(stat.targets) == 1 and len(stat.exprs) == 1
                and isinstance(stat.targets[0], Name) and _int(stat.exprs[0]) in self.table
            ):
                continue
            name = stat.targets[0].name
            verdict = top
            for later in stats[i + 1:]:
                if self._reads(later, name):
                    verdict = False
                    break
                if isinstance(later, Assign) and any(
                    isinstance(t, Name) and t.name == name for t in later.targets
                ):
                    verdict = True
                    break
            if verdict:
                keep[i] = False
        return [stat for stat, flag in zip(stats, keep) if flag]

    def _dead_in(self, stat):
        if not dataclasses.is_dataclass(stat) or isinstance(stat, type):
            return stat
        values = {}
        for f in dataclasses.fields(stat):
            value = getattr(stat, f.name)
            values[f.name] = self._dead_value(value)
        return type(stat)(**values)

    def _dead_value(self, value):
        if isinstance(value, Block):
            return Block(self._dead_keys(value.stats, False))
        if isinstance(value, list):
            return [self._dead_value(item) for item in value]
        if dataclasses.is_dataclass(value) and not isinstance(value, type) and hasattr(value, "body"):
            return self._dead_in(value)
        return value

    def _defs(self, node) -> set[str]:
        names: set[str] = set()
        for item in _walk(node):
            if isinstance(item, Assign):
                names.update(t.name for t in item.targets if isinstance(t, Name))
            elif isinstance(item, LocalAssign):
                names.update(item.names)
        return names

    def _block(self, stats: list, env: dict[str, int]) -> list:
        env = dict(env)
        out = []
        for stat in stats:
            out.append(self._local_stat(stat, env))
            if (
                isinstance(stat, Assign) and len(stat.targets) == 1 and len(stat.exprs) == 1
                and isinstance(stat.targets[0], Name) and _int(stat.exprs[0]) in self.table
            ):
                env[stat.targets[0].name] = _int(stat.exprs[0])
            else:
                for name in self._defs(stat):
                    env.pop(name, None)
        return out

    def _local_stat(self, stat, env: dict[str, int]):
        from .ast_nodes import While as WhileStat

        if isinstance(stat, WhileStat):
            killed = self._defs(stat)
            inner = {k: v for k, v in env.items() if k not in killed}
            return WhileStat(self._local_expr(stat.cond, inner), Block(self._block(stat.body.stats, inner)))
        values = {}
        for f in dataclasses.fields(stat):
            values[f.name] = self._local_value(getattr(stat, f.name), env)
        return type(stat)(**values)

    def _local_value(self, value, env: dict[str, int]):
        if isinstance(value, Block):
            return Block(self._block(value.stats, env))
        if isinstance(value, list):
            return [self._local_value(item, env) for item in value]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            if hasattr(value, "cond") and hasattr(value, "body") and not isinstance(value, Block):
                return type(value)(**{
                    f.name: self._local_value(getattr(value, f.name), env)
                    for f in dataclasses.fields(value)
                })
            return self._local_expr(value, env)
        return value

    def _local_expr(self, node, env: dict[str, int]):
        if isinstance(node, list):
            return [self._local_expr(item, env) for item in node]
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        if isinstance(node, Block):
            return Block(self._block(node.stats, env))
        rebuilt = type(node)(**{
            f.name: self._local_expr(getattr(node, f.name), env) for f in dataclasses.fields(node)
        })
        if isinstance(rebuilt, Index) and isinstance(rebuilt.key, Name) and rebuilt.key.name in env:
            self.report.decrypted += 1
            return String(self.table[env[rebuilt.key.name]])
        return rebuilt

    def polish(self, node):
        if isinstance(node, list):
            return [self.polish(item) for item in node]
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        rebuilt = type(node)(**{
            f.name: self.polish(getattr(node, f.name)) for f in dataclasses.fields(node)
        })
        if isinstance(rebuilt, Index) and isinstance(rebuilt.key, String):
            key = rebuilt.key.value
            if key.isidentifier() and key not in _KEYWORDS:
                obj = rebuilt.obj
                if isinstance(obj, Name) and obj.name == "ENV":
                    self.report.fields += 1
                    return Name(key)
                if isinstance(obj, (Name, Field, Index, Call, MethodCall, Paren)):
                    self.report.fields += 1
                    return Field(obj, key)
        return rebuilt


def decrypt_constants(function_stats: list) -> tuple[list, DecryptReport]:
    params = extract_params(function_stats)
    if params is None:
        return function_stats, DecryptReport()
    worker = _Decryptor(params)
    visited = [worker.visit(stats) for stats in function_stats]
    resolved = [worker.local_pass(worker.resolve_registers(stats)) for stats in visited]
    return [worker.polish(stats) for stats in resolved], worker.report
