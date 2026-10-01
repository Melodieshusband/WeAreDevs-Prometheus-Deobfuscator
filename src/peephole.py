from __future__ import annotations

import dataclasses
from collections import Counter

from .ast_nodes import (
    Assign, Block, Call, CallStat, Expr, FunctionExpr, GenericFor, If, IfClause,
    Do, LocalAssign, MethodCall, Name, NumericFor, Repeat, Return, Stat, Table,
    TableField, While,
)
from .strings_layer import _walk

_LOOPS = (While, NumericFor, GenericFor, Repeat)


def _declared(stats) -> set[str]:
    names: set[str] = set()
    stack = [stats]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
            continue
        if isinstance(node, FunctionExpr):
            continue
        if isinstance(node, LocalAssign):
            names.update(node.names)
        if isinstance(node, NumericFor):
            names.add(node.var)
        if isinstance(node, GenericFor):
            names.update(node.names)
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            for f in dataclasses.fields(node):
                stack.append(getattr(node, f.name))
    return names


def _reads(node) -> set[str]:
    if isinstance(node, list):
        out: set[str] = set()
        for item in node:
            out |= _reads(item)
        return out
    if isinstance(node, Name):
        return {node.name}
    if isinstance(node, FunctionExpr):
        return _reads(node.body) - set(node.params) - _declared(node.body.stats)
    if isinstance(node, NumericFor):
        inner = _reads(node.body) - {node.var}
        return _reads([node.start, node.stop, node.step]) | inner
    if isinstance(node, GenericFor):
        return _reads(node.exprs) | (_reads(node.body) - set(node.names))
    if isinstance(node, Assign):
        out = _reads(node.exprs)
        for target in node.targets:
            if not isinstance(target, Name):
                out |= _reads(target)
        return out
    if not dataclasses.is_dataclass(node) or isinstance(node, type):
        return set()
    out = set()
    for f in dataclasses.fields(node):
        out |= _reads(getattr(node, f.name))
    return out


def _must_kill_list(stats) -> set[str]:
    kills: set[str] = set()
    for stat in stats:
        kills |= _must_kill(stat)
    return kills


def _must_kill(stat) -> set[str]:
    if isinstance(stat, Assign):
        return {t.name for t in stat.targets if isinstance(t, Name)}
    if isinstance(stat, LocalAssign):
        return set(stat.names)
    if isinstance(stat, If):
        if stat.orelse is None:
            return set()
        sets = [_must_kill_list(c.body.stats) for c in stat.clauses]
        sets.append(_must_kill_list(stat.orelse.stats))
        return set.intersection(*sets) if sets else set()
    if isinstance(stat, Repeat):
        return _must_kill_list(stat.body.stats)
    if isinstance(stat, Do):
        return _must_kill_list(stat.body.stats)
    return set()


def _ue_list(stats) -> set[str]:
    live: set[str] = set()
    for stat in reversed(stats):
        if isinstance(stat, Return):
            live = _ue(stat)
            continue
        live = (live - _must_kill(stat)) | _ue(stat)
    return live


def _ue(stat) -> set[str]:
    if isinstance(stat, If):
        out = _reads(stat.clauses[0].cond)
        for position, clause in enumerate(stat.clauses):
            if position > 0:
                out |= _reads(clause.cond)
            out |= _ue_list(clause.body.stats)
        if stat.orelse is not None:
            out |= _ue_list(stat.orelse.stats)
        return out
    if isinstance(stat, While):
        return _reads(stat.cond) | _ue_list(stat.body.stats)
    if isinstance(stat, Repeat):
        return _ue_list(stat.body.stats) | (_reads(stat.cond) - _must_kill_list(stat.body.stats))
    if isinstance(stat, NumericFor):
        return _reads([stat.start, stat.stop, stat.step]) | (_ue_list(stat.body.stats) - {stat.var})
    if isinstance(stat, GenericFor):
        return _reads(stat.exprs) | (_ue_list(stat.body.stats) - set(stat.names))
    if isinstance(stat, Do):
        return _ue_list(stat.body.stats)
    return _reads(stat)


_kills = _must_kill


def _pure(node) -> bool:
    if isinstance(node, FunctionExpr):
        return True
    if isinstance(node, (Call, MethodCall)):
        return False
    if isinstance(node, list):
        return all(_pure(item) for item in node)
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return all(_pure(getattr(node, f.name)) for f in dataclasses.fields(node))
    return True


def _is_unpack_of(expr, name: str) -> bool:
    return (
        isinstance(expr, Call) and isinstance(expr.func, Name) and expr.func.name == "unpack"
        and len(expr.args) == 1 and isinstance(expr.args[0], Name) and expr.args[0].name == name
    )


def _plain_table(expr) -> list[Expr] | None:
    if isinstance(expr, Table) and all(f.key is None for f in expr.fields):
        return [f.value for f in expr.fields]
    return None


def _splice_last(stat, name: str, values: list[Expr]):
    count = sum(1 for item in _walk(stat) if isinstance(item, Name) and item.name == name)
    if count != 1:
        return None
    changed = [False]

    def rewrite(node):
        if isinstance(node, list):
            return [rewrite(item) for item in node]
        if isinstance(node, FunctionExpr):
            return node
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        if isinstance(node, (Call, MethodCall)) and node.args and _is_unpack_of(node.args[-1], name):
            earlier = node.args[:-1]
            head = node.func if isinstance(node, Call) else node.obj
            if _pure(earlier) and _pure(head):
                changed[0] = True
                args = [rewrite(a) for a in earlier] + list(values)
                if isinstance(node, Call):
                    return Call(rewrite(node.func), args)
                return MethodCall(rewrite(node.obj), node.method, args)
        if isinstance(node, Return) and node.exprs and _is_unpack_of(node.exprs[-1], name):
            if _pure(node.exprs[:-1]):
                changed[0] = True
                return Return([rewrite(e) for e in node.exprs[:-1]] + list(values))
        if isinstance(node, Table) and node.fields:
            last = node.fields[-1]
            if last.key is None and _is_unpack_of(last.value, name) and _pure(node.fields[:-1]):
                changed[0] = True
                return Table(
                    [rewrite(f) for f in node.fields[:-1]] + [TableField(None, v) for v in values]
                )
        return type(node)(**{f.name: rewrite(getattr(node, f.name)) for f in dataclasses.fields(node)})

    result = rewrite(stat)
    return result if changed[0] else None


def _expr(node):
    if isinstance(node, list):
        return [_expr(item) for item in node]
    if isinstance(node, FunctionExpr):
        return FunctionExpr(node.params, node.is_vararg, Block(peephole(node.body.stats, set())))
    if not dataclasses.is_dataclass(node) or isinstance(node, type):
        return node
    return type(node)(**{f.name: _expr(getattr(node, f.name)) for f in dataclasses.fields(node)})


def _nested(stat, live_after: set[str]):
    if isinstance(stat, If):
        clauses = [
            IfClause(_expr(c.cond), Block(peephole(c.body.stats, live_after))) for c in stat.clauses
        ]
        orelse = Block(peephole(stat.orelse.stats, live_after)) if stat.orelse is not None else None
        return If(clauses, orelse)
    if isinstance(stat, _LOOPS):
        inner_live = live_after | _ue(stat)
        values = {}
        for f in dataclasses.fields(stat):
            value = getattr(stat, f.name)
            if isinstance(value, Block):
                values[f.name] = Block(peephole(value.stats, inner_live))
            else:
                values[f.name] = _expr(value)
        return type(stat)(**values)
    return _expr(stat)


def peephole(stats: list[Stat], live_out: set[str]) -> list[Stat]:
    count = len(stats)
    live_after: list[set[str]] = [set() for _ in range(count)]
    live = set(live_out)
    for i in range(count - 1, -1, -1):
        live_after[i] = set(live)
        live = _ue(stats[i]) if isinstance(stats[i], Return) else (live - _kills(stats[i])) | _ue(stats[i])

    work = [_nested(stat, live_after[i]) for i, stat in enumerate(stats)]
    index = 0
    while index < len(work):
        stat = work[index]
        if (
            isinstance(stat, Assign) and len(stat.targets) == 1 and len(stat.exprs) == 1
            and isinstance(stat.targets[0], Name)
        ):
            name = stat.targets[0].name
            values = _plain_table(stat.exprs[0])
            if values is not None and len(values) == 1 and _is_unpack_of(values[0], name):
                del work[index]
                continue
            if values is not None and index + 1 < len(work):
                following = work[index + 1]
                after = _live_after_at(work, index + 1, live_out)
                if name not in after:
                    if isinstance(following, Return) and len(following.exprs) == 1 and _is_unpack_of(following.exprs[0], name):
                        work[index:index + 2] = [Return(list(values))]
                        continue
                    replaced = _splice_last(following, name, values)
                    if replaced is not None:
                        work[index:index + 2] = [replaced]
                        continue
        index += 1
    return work


def _live_after_at(work: list[Stat], position: int, live_out: set[str]) -> set[str]:
    live = set(live_out)
    for stat in reversed(work[position + 1:]):
        live = _ue(stat) if isinstance(stat, Return) else (live - _kills(stat)) | _ue(stat)
    return live
