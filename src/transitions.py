from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable

from .strings_layer import _walk
from .ast_nodes import Assign, BinOp, Expr, FunctionExpr, LocalAssign, Name, Number, Stat, UnOp
from .fold import ConstFoldError, try_eval_const


EXIT = -1


@dataclass
class Jump:
    kind: str
    target: int | None = None
    cond: Expr | None = None
    true_target: int | None = None
    false_target: int | None = None
    true_expr: Expr | None = None
    false_expr: Expr | None = None
    raw: Expr | None = None


def _as_int(expr: Expr) -> int | None:
    try:
        value = try_eval_const(expr)
    except ConstFoldError:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    return None


def _target(expr: Expr, exit_check: Callable[[Expr], bool] | None) -> int | None:
    const = _as_int(expr)
    if const is not None:
        return const
    if exit_check is not None and exit_check(expr):
        return EXIT
    return None


def classify_jump(expr: Expr, exit_check: Callable[[Expr], bool] | None = None) -> Jump:
    if exit_check is not None and exit_check(expr):
        return Jump(kind="exit", target=EXIT, raw=expr)

    const = _as_int(expr)
    if const is not None:
        return Jump(kind="const", target=const)

    if isinstance(expr, BinOp) and expr.op == "or":
        left = expr.left
        if isinstance(left, BinOp) and left.op == "and":
            cond, true_branch = left.left, left.right
            false_branch = expr.right
            return Jump(
                kind="cond",
                cond=cond,
                true_target=_target(true_branch, exit_check),
                false_target=_target(false_branch, exit_check),
                true_expr=true_branch,
                false_expr=false_branch,
            )

    return Jump(kind="dynamic", raw=expr)


def find_state_assignment_index(stats: list[Stat], state_var: str) -> int | None:
    last = None
    for i, stat in enumerate(stats):
        if isinstance(stat, Assign):
            for target in stat.targets:
                if isinstance(target, Name) and target.name == state_var:
                    last = i
    return last


def find_all_state_assignment_indices(stats: list[Stat], state_var: str) -> list[int]:
    indices = []
    for i, stat in enumerate(stats):
        if isinstance(stat, Assign):
            for target in stat.targets:
                if isinstance(target, Name) and target.name == state_var:
                    indices.append(i)
                    break
    return indices


def _is_const_expr(expr: Expr) -> bool:
    if isinstance(expr, Number):
        return True
    return isinstance(expr, UnOp) and expr.op == "-" and isinstance(expr.operand, Number)


def _assigned_names(stat: Stat) -> set[str]:
    names: set[str] = set()
    stack: list[object] = [stat]
    while stack:
        node = stack.pop()
        if isinstance(node, Assign):
            names.update(t.name for t in node.targets if isinstance(t, Name))
        elif isinstance(node, LocalAssign):
            names.update(node.names)
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            for item in dataclasses.fields(node):
                value = getattr(node, item.name)
                if isinstance(value, list):
                    stack.extend(value)
                else:
                    stack.append(value)
    return names


def _const_environments(stats: list[Stat], state_var: str) -> list[dict[str, Expr]]:
    env: dict[str, Expr] = {}
    result: list[dict[str, Expr]] = []
    for stat in stats:
        result.append(dict(env))
        if isinstance(stat, (Assign, LocalAssign)):
            targets = stat.targets if isinstance(stat, Assign) else [Name(n) for n in stat.names]
            for position, target in enumerate(targets):
                if not isinstance(target, Name):
                    continue
                aligned = stat.exprs[position] if position < len(stat.exprs) else None
                if aligned is not None and _is_const_expr(aligned) and target.name != state_var:
                    env[target.name] = aligned
                else:
                    env.pop(target.name, None)
        else:
            for name in _assigned_names(stat):
                env.pop(name, None)
    return result


def _substitute(node, env: dict[str, Expr]):
    if isinstance(node, Name):
        return env.get(node.name, node)
    if isinstance(node, FunctionExpr):
        return node
    if isinstance(node, list):
        return [_substitute(item, env) for item in node]
    if isinstance(node, Expr) and dataclasses.is_dataclass(node):
        values = {
            item.name: _substitute(getattr(node, item.name), env)
            for item in dataclasses.fields(node)
        }
        return type(node)(**values)
    return node


def _expr_for_target(stat: Assign, state_var: str) -> Expr | None:
    for pos, target in enumerate(stat.targets):
        if isinstance(target, Name) and target.name == state_var:
            return stat.exprs[pos] if pos < len(stat.exprs) else None
    return None


def _is_self_or(expr: Expr, state_var: str) -> Expr | None:
    if isinstance(expr, BinOp) and expr.op == "or" and isinstance(expr.left, Name) and expr.left.name == state_var:
        return expr.right
    return None


SNAPSHOT = "__c"


def _snapshot_cond(jump: Jump, body: list[Stat], position: int, later: list[Stat]) -> None:
    if jump.kind != "cond" or jump.cond is None:
        return
    cond_names = {item.name for item in _walk(jump.cond) if isinstance(item, Name)}
    clobbered = False
    for stat in later:
        if _assigned_names(stat) & cond_names:
            clobbered = True
            break
    if clobbered:
        body.insert(position, Assign([Name(SNAPSHOT)], [jump.cond]))
        jump.cond = Name(SNAPSHOT)


def extract_transition(
    stats: list[Stat],
    state_var: str,
    exit_check: Callable[[Expr], bool] | None = None,
) -> tuple[list[Stat], Jump | None]:
    indices = find_all_state_assignment_indices(stats, state_var)
    if not indices:
        return stats, None
    envs = _const_environments(stats, state_var)

    last_index = indices[-1]
    last_stat = stats[last_index]
    assert isinstance(last_stat, Assign)
    last_expr = _expr_for_target(last_stat, state_var)
    if last_expr is not None:
        last_expr = _substitute(last_expr, envs[last_index])

    fused_expr = None
    consumed_prev_index = None
    if last_expr is not None:
        tail = _is_self_or(last_expr, state_var)
        if tail is not None and len(indices) >= 2:
            prev_index = indices[-2]
            prev_stat = stats[prev_index]
            assert isinstance(prev_stat, Assign)
            prev_expr = _expr_for_target(prev_stat, state_var)
            if prev_expr is not None:
                prev_expr = _substitute(prev_expr, envs[prev_index])
            if isinstance(prev_expr, BinOp) and prev_expr.op == "and":
                fused_expr = BinOp("or", prev_expr, tail)
                consumed_prev_index = prev_index

    if fused_expr is not None:
        jump = classify_jump(fused_expr, exit_check)
        prev_stat = stats[consumed_prev_index]
        other_targets = [t for i, t in enumerate(prev_stat.targets) if not (isinstance(t, Name) and t.name == state_var)]
        other_exprs = [e for i, t in enumerate(prev_stat.targets) if not (isinstance(t, Name) and t.name == state_var) for e in [prev_stat.exprs[i]] if i < len(prev_stat.exprs)]
        residual_prev: list[Stat] = [Assign(other_targets, other_exprs)] if other_targets else []

        body = stats[:consumed_prev_index] + residual_prev + stats[consumed_prev_index + 1 : last_index]
        other_targets2 = [t for t in last_stat.targets if not (isinstance(t, Name) and t.name == state_var)]
        other_exprs2 = [e for t, e in zip(last_stat.targets, last_stat.exprs) if not (isinstance(t, Name) and t.name == state_var)]
        if other_targets2:
            body = body + [Assign(other_targets2, other_exprs2)]
        body = body + stats[last_index + 1 :]
        _snapshot_cond(jump, body, consumed_prev_index, body[consumed_prev_index + len(residual_prev):])
        return body, jump

    state_pos = None
    for pos, target in enumerate(last_stat.targets):
        if isinstance(target, Name) and target.name == state_var:
            state_pos = pos
    assert state_pos is not None

    jump_expr = last_stat.exprs[state_pos] if state_pos < len(last_stat.exprs) else None
    if jump_expr is not None:
        jump_expr = _substitute(jump_expr, envs[last_index])
    jump = classify_jump(jump_expr, exit_check) if jump_expr is not None else None

    other_targets = [t for i, t in enumerate(last_stat.targets) if i != state_pos]
    other_exprs = [e for i, e in enumerate(last_stat.exprs) if i != state_pos]
    residual: list[Stat] = []
    if other_targets:
        residual.append(Assign(other_targets, other_exprs))

    tail = stats[last_index + 1 :]
    body = stats[:last_index] + residual + tail
    if jump is not None:
        _snapshot_cond(jump, body, last_index, residual + tail)
    return body, jump
