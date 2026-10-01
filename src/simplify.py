from __future__ import annotations

import dataclasses

from .ast_nodes import (
    Assign, BinOp, Call, CallStat, Do, GenericFor, NumericFor, Repeat, While, Expr, FalseLit, Field, FunctionExpr, If,
    Index, LocalAssign, MethodCall, Name, Nil, Number, Paren, Return, Stat,
    String, Table, TableField, TrueLit, UnOp, Vararg,
)
from .strings_layer import _walk
from .transitions import _assigned_names

_PRECEDENCE = {
    "or": 1, "and": 2,
    "<": 3, ">": 3, "<=": 3, ">=": 3, "~=": 3, "==": 3,
    "..": 5, "+": 6, "-": 6, "*": 7, "/": 7, "%": 7, "//": 7, "^": 10,
}
_RIGHT_ASSOC = {"..", "^"}
_UNARY_PRECEDENCE = 8
_CONSTANT_TYPES = (Number, String, TrueLit, FalseLit, Nil)
PSEUDO_COND = "__cond"
_COMPOUND = (If, While, NumericFor, GenericFor, Repeat, Do)


def names_in(node) -> list[str]:
    return [item.name for item in _walk(node) if isinstance(item, Name)]


def count_name(node, name: str) -> int:
    return sum(1 for item in _walk(node) if isinstance(item, Name) and item.name == name)


def has_call(node) -> bool:
    return any(isinstance(item, (Call, MethodCall)) for item in _walk(node))


def has_function(node) -> bool:
    return any(isinstance(item, FunctionExpr) for item in _walk(node))


def has_memory_read(node) -> bool:
    return any(isinstance(item, (Index, Field)) for item in _walk(node))


def uses_of(stat: Stat) -> set[str]:
    if isinstance(stat, Assign):
        used = set(names_in(stat.exprs))
        for target in stat.targets:
            if not isinstance(target, Name):
                used.update(names_in(target))
        return used
    if isinstance(stat, LocalAssign):
        return set(names_in(stat.exprs))
    return set(names_in(stat))


def defs_of(stat: Stat) -> set[str]:
    if isinstance(stat, Assign):
        return {t.name for t in stat.targets if isinstance(t, Name)}
    if isinstance(stat, LocalAssign):
        return set(stat.names)
    return _assigned_names(stat)


def has_side_effects(stat: Stat) -> bool:
    if isinstance(stat, Assign):
        if any(not isinstance(t, Name) for t in stat.targets):
            return True
        return has_call(stat.exprs)
    if isinstance(stat, LocalAssign):
        return has_call(stat.exprs)
    return True


def _precedence(expr: Expr) -> int:
    if isinstance(expr, BinOp):
        return _PRECEDENCE.get(expr.op, 0)
    if isinstance(expr, UnOp):
        return _UNARY_PRECEDENCE
    return 100


def needs_paren(child: Expr, parent: Expr | None, side: str, prefix: bool) -> bool:
    if prefix:
        return not isinstance(child, (Name, Index, Field, Call, MethodCall, Paren))
    if not isinstance(child, (BinOp, UnOp)):
        return False
    if isinstance(parent, UnOp):
        return isinstance(child, BinOp) and _precedence(child) < _UNARY_PRECEDENCE
    if not isinstance(parent, BinOp):
        return False
    child_level = _precedence(child)
    parent_level = _precedence(parent)
    if child_level < parent_level:
        return True
    if child_level > parent_level:
        return isinstance(child, UnOp) and parent.op == "^" and side == "left"
    if parent.op in _RIGHT_ASSOC:
        return side == "left"
    return side == "right"


def substitute(node, name: str, replacement: Expr, parent: Expr | None = None, side: str = "top", prefix: bool = False):
    if isinstance(node, list):
        return [substitute(item, name, replacement, None, "top", False) for item in node]
    if isinstance(node, Name):
        if node.name != name:
            return node
        if needs_paren(replacement, parent, side, prefix):
            return Paren(replacement)
        return replacement
    if isinstance(node, FunctionExpr):
        return node
    if isinstance(node, NumericFor):
        body = node.body if node.var == name else substitute(node.body, name, replacement)
        return NumericFor(
            node.var,
            substitute(node.start, name, replacement),
            substitute(node.stop, name, replacement),
            substitute(node.step, name, replacement) if node.step is not None else None,
            body,
        )
    if isinstance(node, GenericFor):
        body = node.body if name in node.names else substitute(node.body, name, replacement)
        return GenericFor(node.names, substitute(node.exprs, name, replacement), body)
    if isinstance(node, TableField):
        key = substitute(node.key, name, replacement) if node.key is not None else None
        return TableField(key, substitute(node.value, name, replacement))
    if isinstance(node, BinOp):
        return BinOp(
            node.op,
            substitute(node.left, name, replacement, node, "left", False),
            substitute(node.right, name, replacement, node, "right", False),
        )
    if isinstance(node, UnOp):
        return UnOp(node.op, substitute(node.operand, name, replacement, node, "operand", False))
    if isinstance(node, Index):
        return Index(
            substitute(node.obj, name, replacement, node, "obj", True),
            substitute(node.key, name, replacement),
        )
    if isinstance(node, Field):
        return Field(substitute(node.obj, name, replacement, node, "obj", True), node.name)
    if isinstance(node, Call):
        return Call(
            substitute(node.func, name, replacement, node, "func", True),
            [substitute(a, name, replacement) for a in node.args],
        )
    if isinstance(node, MethodCall):
        return MethodCall(
            substitute(node.obj, name, replacement, node, "obj", True),
            node.method,
            [substitute(a, name, replacement) for a in node.args],
        )
    if isinstance(node, Paren):
        return Paren(substitute(node.inner, name, replacement))
    if isinstance(node, Assign):
        targets = []
        for target in node.targets:
            if isinstance(target, Name):
                targets.append(target)
            else:
                targets.append(substitute(target, name, replacement))
        return Assign(targets, [substitute(e, name, replacement) for e in node.exprs])
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        values = {
            item.name: substitute(getattr(node, item.name), name, replacement)
            for item in dataclasses.fields(node)
        }
        return type(node)(**values)
    return node


def split_assign(stat: Stat) -> list[Stat]:
    if not isinstance(stat, Assign):
        return [stat]
    if len(stat.targets) < 2 or len(stat.targets) != len(stat.exprs):
        return [stat]
    if not all(isinstance(t, Name) for t in stat.targets):
        return [stat]
    for i, target in enumerate(stat.targets):
        for j in range(i + 1, len(stat.exprs)):
            if target.name in names_in(stat.exprs[j]):
                return [stat]
    return [Assign([t], [e]) for t, e in zip(stat.targets, stat.exprs)]


def _is_simple(expr: Expr) -> bool:
    return isinstance(expr, _CONSTANT_TYPES) or (
        isinstance(expr, UnOp) and expr.op == "-" and isinstance(expr.operand, Number)
    )


def _expr_movable(expr: Expr, between: list[Stat], names: set[str]) -> bool:
    for stat in between:
        if defs_of(stat) & names:
            return False
    impure = has_call(expr)
    reads = has_memory_read(expr)
    if impure:
        for stat in between:
            if has_side_effects(stat) or has_call(stat) or has_memory_read(stat):
                return False
    elif reads:
        for stat in between:
            if has_side_effects(stat):
                return False
    return True


def _use_is_first_evaluated(stat: Stat, name: str, expr: Expr) -> bool:
    if not has_call(expr):
        return True
    calls = sum(1 for item in _walk(stat) if isinstance(item, (Call, MethodCall)))
    if isinstance(stat, CallStat):
        return calls <= 1
    if isinstance(stat, (Assign, LocalAssign)):
        return calls <= 1
    return False


def simplify_stats(stats: list[Stat], live_out: set[str]) -> list[Stat]:
    work: list[Stat] = []
    for stat in stats:
        work.extend(split_assign(stat))

    changed = True
    while changed:
        changed = False
        for i, stat in enumerate(work):
            if not isinstance(stat, Assign) or len(stat.targets) != 1 or len(stat.exprs) != 1:
                continue
            target = stat.targets[0]
            if not isinstance(target, Name):
                continue
            register = target.name
            if register == PSEUDO_COND:
                continue
            expr = stat.exprs[0]
            if has_function(expr):
                continue
            redefined = None
            conditional = False
            for j in range(i + 1, len(work)):
                if register in defs_of(work[j]):
                    redefined = j
                    if not isinstance(work[j], (Assign, LocalAssign)):
                        conditional = True
                    break
            if conditional:
                continue
            last = redefined if redefined is not None else len(work) - 1
            use_positions = [j for j in range(i + 1, last + 1) if register in uses_of(work[j])]
            escapes = redefined is None and register in live_out
            if escapes:
                continue
            if not use_positions:
                if has_call(expr):
                    if isinstance(expr, (Call, MethodCall)):
                        work[i] = CallStat(expr)
                        changed = True
                        break
                    continue
                del work[i]
                changed = True
                break
            expr_names = set(names_in(expr))
            if _is_simple(expr) or isinstance(expr, Name):
                if isinstance(expr, Name) and expr.name == register:
                    continue
                span_end = use_positions[-1]
                if isinstance(expr, Name) and any(
                    expr.name in defs_of(work[j]) for j in range(i + 1, span_end)
                ):
                    continue
                if any(has_function(work[j]) for j in use_positions):
                    continue
                if isinstance(expr, Name) and any(
                    isinstance(work[j], _COMPOUND) and expr.name in defs_of(work[j]) for j in use_positions
                ):
                    continue
                for j in use_positions:
                    work[j] = substitute(work[j], register, expr)
                del work[i]
                changed = True
                break
            if len(use_positions) == 1:
                j = use_positions[0]
                if count_name(work[j], register) != 1:
                    continue
                if isinstance(work[j], _COMPOUND) or has_function(work[j]):
                    continue
                if not _expr_movable(expr, work[i + 1:j], expr_names):
                    continue
                if not _use_is_first_evaluated(work[j], register, expr):
                    continue
                if expr_names & (defs_of(work[j]) - {register}) and not isinstance(work[j], Assign):
                    continue
                work[j] = substitute(work[j], register, expr)
                del work[i]
                changed = True
                break
    return work


def negate(cond: Expr) -> Expr:
    if isinstance(cond, UnOp) and cond.op == "not":
        inner = cond.operand
        return inner.inner if isinstance(inner, Paren) and not isinstance(inner.inner, BinOp) else inner
    if isinstance(cond, Paren):
        return negate(cond.inner)
    if isinstance(cond, BinOp) and cond.op == "==":
        return BinOp("~=", cond.left, cond.right)
    if isinstance(cond, BinOp) and cond.op == "~=":
        return BinOp("==", cond.left, cond.right)
    if isinstance(cond, TrueLit):
        return FalseLit()
    if isinstance(cond, FalseLit):
        return TrueLit()
    if isinstance(cond, (BinOp,)):
        return UnOp("not", Paren(cond))
    return UnOp("not", cond)


def merge_return(stats: list[Stat], unpack_name: str, table_name: str) -> list[Stat]:
    if not stats or not isinstance(stats[-1], Return):
        return stats
    ret = stats[-1]
    if len(ret.exprs) != 1 or not isinstance(ret.exprs[0], Call):
        return stats
    call = ret.exprs[0]
    if not (isinstance(call.func, Name) and call.func.name == unpack_name) or len(call.args) != 1:
        return stats
    arg = call.args[0]
    if isinstance(arg, Table) and all(f.key is None for f in arg.fields):
        return stats[:-1] + [Return([f.value for f in arg.fields])]
    return stats
