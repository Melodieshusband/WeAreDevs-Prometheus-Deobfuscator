from __future__ import annotations

import dataclasses

from .ast_nodes import (
    Assign, BinOp, Block, Break, Call, Expr, If, IfClause, MethodCall, Name,
    Number, NumericFor, Paren, Stat, TrueLit, UnOp, While,
)
from .simplify import negate
from .strings_layer import _walk


def _strip(expr):
    while isinstance(expr, Paren):
        expr = expr.inner
    return expr


def _name(expr) -> str | None:
    expr = _strip(expr)
    return expr.name if isinstance(expr, Name) else None


def _is_const(expr) -> bool:
    expr = _strip(expr)
    return isinstance(expr, Number) or (
        isinstance(expr, UnOp) and expr.op == "-" and isinstance(expr.operand, Number)
    )


def _compare_sides(expr, counter: str) -> tuple[str, Expr] | None:
    expr = _strip(expr)
    if not isinstance(expr, BinOp) or expr.op not in (">=", "<="):
        return None
    left, right = _strip(expr.left), _strip(expr.right)
    if expr.op == ">=":
        greater, lesser = left, right
    else:
        greater, lesser = right, left
    if _name(lesser) == counter:
        return "counter_le", greater
    if _name(greater) == counter:
        return "counter_ge", lesser
    return None


def _match_condition(cond, counter: str):
    cond = _strip(cond)
    if not (isinstance(cond, BinOp) and cond.op == "or"):
        return None
    first, second = _strip(cond.left), _strip(cond.right)
    if not (isinstance(first, BinOp) and first.op == "and" and isinstance(second, BinOp) and second.op == "and"):
        return None
    flag = _name(first.left)
    negated = _strip(second.left)
    if flag is None or not (isinstance(negated, UnOp) and negated.op == "not" and _name(negated.operand) == flag):
        return None
    descending = _compare_sides(first.right, counter)
    ascending = _compare_sides(second.right, counter)
    if descending is None or ascending is None:
        return None
    if descending[0] != "counter_ge" or ascending[0] != "counter_le":
        return None
    return flag, ascending[1]


def _match_loop(loop: While):
    body = loop.body.stats
    if len(body) < 2 or not isinstance(loop.cond, TrueLit):
        return None
    inc = body[0]
    if not (isinstance(inc, Assign) and len(inc.targets) == 1 and len(inc.exprs) == 1):
        return None
    counter = _name(inc.targets[0])
    expr = _strip(inc.exprs[0])
    if counter is None or not (isinstance(expr, BinOp) and expr.op == "+"):
        return None
    left, right = _name(expr.left), _name(expr.right)
    if left == counter and right is not None:
        step = right
    elif right == counter and left is not None:
        step = left
    else:
        return None
    guard = body[1]
    if not isinstance(guard, If) or len(guard.clauses) != 1:
        return None
    clause = guard.clauses[0]
    if guard.orelse is not None and len(guard.orelse.stats) == 1 and isinstance(guard.orelse.stats[0], Break):
        matched = _match_condition(clause.cond, counter)
        if matched is None or body[2:]:
            return None
        return counter, step, matched[0], matched[1], list(clause.body.stats)
    if guard.orelse is None and len(clause.body.stats) == 1 and isinstance(clause.body.stats[0], Break):
        cond = _strip(clause.cond)
        if not (isinstance(cond, UnOp) and cond.op == "not"):
            return None
        matched = _match_condition(cond.operand, counter)
        if matched is None:
            return None
        return counter, step, matched[0], matched[1], list(body[2:])
    return None


def _last_def(stats: list, upto: int, name: str) -> int | None:
    for j in range(upto - 1, -1, -1):
        stat = stats[j]
        if isinstance(stat, Assign) and any(_name(t) == name for t in stat.targets):
            return j
        for item in _walk(stat):
            if isinstance(item, Assign) and any(isinstance(t, Name) and t.name == name for t in item.targets):
                return None
    return None


def _reads(stat, name: str) -> bool:
    total = sum(1 for item in _walk(stat) if isinstance(item, Name) and item.name == name)
    writes = sum(
        1 for item in _walk(stat) if isinstance(item, Assign)
        for target in item.targets if isinstance(target, Name) and target.name == name
    )
    return total - writes > 0


def _try_for(stats: list, index: int):
    loop = stats[index]
    matched = _match_loop(loop)
    if matched is None:
        return None
    counter, step_name, flag, limit_expr, body = matched
    j_counter = _last_def(stats, index, counter)
    j_flag = _last_def(stats, index, flag)
    if j_counter is None or j_flag is None:
        return None
    counter_stat, flag_stat = stats[j_counter], stats[j_flag]
    if not (len(counter_stat.targets) == 1 and len(counter_stat.exprs) == 1):
        return None
    if not (len(flag_stat.targets) == 1 and len(flag_stat.exprs) == 1):
        return None
    init = _strip(counter_stat.exprs[0])
    if not (isinstance(init, BinOp) and init.op == "-" and _name(init.right) == step_name):
        return None
    flag_expr = _strip(flag_stat.exprs[0])
    if not (
        isinstance(flag_expr, BinOp) and (
            (flag_expr.op == ">" and _is_const(flag_expr.left) and _name(flag_expr.right) == step_name)
            or (flag_expr.op == "<" and _name(flag_expr.left) == step_name and _is_const(flag_expr.right))
        )
    ):
        return None
    for j in range(min(j_counter, j_flag) + 1, index):
        if j in (j_counter, j_flag):
            continue
        if any(_reads(stats[j], n) or _defines(stats[j], n) for n in (counter, flag)):
            return None
    start = init.left
    step: Expr = Name(step_name)
    step_def = _last_def(stats, index, step_name)
    if step_def is not None:
        candidate = stats[step_def]
        if len(candidate.targets) == 1 and len(candidate.exprs) == 1 and _is_const(candidate.exprs[0]):
            step = candidate.exprs[0]
    limit: Expr = limit_expr
    limit_name = _name(limit_expr)
    if limit_name is not None:
        limit_def = _last_def(stats, index, limit_name)
        if limit_def is not None:
            candidate = stats[limit_def]
            if len(candidate.targets) == 1 and len(candidate.exprs) == 1 and _is_const(candidate.exprs[0]):
                limit = candidate.exprs[0]
    step_text = _strip(step)
    step_arg = None if (isinstance(step_text, Number) and step_text.text == "1") else step
    replacement = NumericFor(counter, start, limit, step_arg, Block(body))
    return j_counter, j_flag, replacement


def _defines(stat, name: str) -> bool:
    return any(
        isinstance(item, Assign) and any(isinstance(t, Name) and t.name == name for t in item.targets)
        for item in _walk(stat)
    )


def convert_for_loops(stats: list) -> list:
    stats = [_recurse(stat, convert_for_loops) for stat in stats]
    index = 0
    while index < len(stats):
        if isinstance(stats[index], While):
            found = _try_for(stats, index)
            if found is not None:
                j_counter, j_flag, replacement = found
                replacement = NumericFor(
                    replacement.var, replacement.start, replacement.stop, replacement.step,
                    Block(convert_for_loops(replacement.body.stats)),
                )
                stats = _splice(stats, j_counter, j_flag, index, replacement)
                index = 0
                continue
        index += 1
    return stats


def _splice(stats: list, j_counter: int, j_flag: int, index: int, replacement) -> list:
    result = []
    for k, stat in enumerate(stats):
        if k in (j_counter, j_flag):
            continue
        if k == index:
            result.append(replacement)
        else:
            result.append(stat)
    return result


def _recurse(stat, fn):
    if not dataclasses.is_dataclass(stat) or isinstance(stat, type):
        return stat
    values = {}
    for f in dataclasses.fields(stat):
        values[f.name] = _recurse_value(getattr(stat, f.name), fn)
    return type(stat)(**values)


def _recurse_value(value, fn):
    if isinstance(value, Block):
        return Block(fn(value.stats))
    if isinstance(value, list):
        return [_recurse_value(item, fn) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, (type, Expr)) and hasattr(value, "body"):
        return _recurse(value, fn)
    return value


def _pure(expr) -> bool:
    return not any(isinstance(item, (Call, MethodCall)) for item in _walk(expr))


def dead_stores(stats: list, top: bool = True) -> list:
    stats = [_recurse(stat, lambda s: dead_stores(s, False)) for stat in stats]
    keep = [True] * len(stats)
    for i, stat in enumerate(stats):
        if not (
            isinstance(stat, Assign) and len(stat.targets) == 1 and len(stat.exprs) == 1
            and isinstance(stat.targets[0], Name) and _pure(stat.exprs[0])
        ):
            continue
        name = stat.targets[0].name
        verdict = top
        for later in stats[i + 1:]:
            if _reads(later, name):
                verdict = False
                break
            if isinstance(later, Assign) and any(isinstance(t, Name) and t.name == name for t in later.targets):
                verdict = True
                break
        if verdict:
            keep[i] = False
    return [stat for stat, flag in zip(stats, keep) if flag]


def cleanup_function(stats: list) -> list:
    previous = None
    current = convert_for_loops(stats)
    for _ in range(8):
        current = dead_stores(current, True)
        if previous is not None and repr(previous) == repr(current):
            break
        previous = current
    return current
