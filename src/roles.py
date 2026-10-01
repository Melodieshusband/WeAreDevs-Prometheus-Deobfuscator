from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .ast_nodes import (
    Assign, BinOp, Call, FunctionExpr, Index, Name, Nil, Number, NumericFor,
    Paren, Return, Table, While,
)
from .strings_layer import _walk

ARGS = "_args"
SLOTS = "_slots"
ENV = "_env"
UNPACK = "_unpack"
SELECT = "_select"
CELLS = "_cells"
ALLOC = "_alloc"
RELEASE = "_release"
RET = "_ret"


@dataclass
class Roles:
    state: str
    args: str
    slots: str
    handle: str
    env: str | None = None
    unpack: str | None = None
    select: str | None = None
    cells: str | None = None
    refcount: str | None = None
    alloc: str | None = None
    release: str | None = None
    retain: str | None = None
    ret_reg: str | None = None

    def rename_map(self, registers: set[str]) -> dict[str, str]:
        pairs = {
            self.args: ARGS, self.slots: SLOTS, self.env: ENV, self.unpack: UNPACK,
            self.select: SELECT, self.cells: CELLS, self.alloc: ALLOC,
            self.release: RELEASE, self.ret_reg: RET,
        }
        result: dict[str, str] = {}
        for old, new in pairs.items():
            if old is None:
                continue
            if old in registers and new != RET:
                continue
            result[old] = new
        return result


def _strip(expr):
    while isinstance(expr, Paren):
        expr = expr.inner
    return expr


def _names(expr) -> set[str]:
    return {item.name for item in _walk(expr) if isinstance(item, Name)}


def _is_alloc(fn: FunctionExpr) -> tuple[str, str] | None:
    if fn.params or len(fn.body.stats) < 3:
        return None
    last = fn.body.stats[-1]
    if not (isinstance(last, Return) and len(last.exprs) == 1 and isinstance(last.exprs[0], Name)):
        return None
    counter = last.exprs[0].name
    for stat in fn.body.stats:
        if (
            isinstance(stat, Assign) and len(stat.targets) == 1 and isinstance(stat.targets[0], Index)
            and isinstance(stat.targets[0].obj, Name) and isinstance(stat.targets[0].key, Name)
            and stat.targets[0].key.name == counter and len(stat.exprs) == 1
            and isinstance(stat.exprs[0], Number) and stat.exprs[0].text == "1"
        ):
            return counter, stat.targets[0].obj.name
    return None


def _is_release(fn: FunctionExpr) -> tuple[str, str] | None:
    if len(fn.params) != 1:
        return None
    if any(isinstance(item, (While, NumericFor)) for item in _walk(fn.body)):
        return None
    param = fn.params[0]
    for item in _walk(fn.body):
        if (
            isinstance(item, Assign) and len(item.targets) == 2 and len(item.exprs) == 2
            and all(isinstance(e, Nil) for e in item.exprs)
            and all(
                isinstance(t, Index) and isinstance(t.obj, Name)
                and isinstance(t.key, Name) and t.key.name == param
                for t in item.targets
            )
        ):
            return item.targets[0].obj.name, item.targets[1].obj.name
    return None


def _is_retain(fn: FunctionExpr) -> bool:
    if len(fn.params) != 1:
        return False
    return any(isinstance(item, NumericFor) for item in _walk(fn.body))


def _classify_argument(expr) -> str | None:
    expr = _strip(expr)
    names = _names(expr)
    if isinstance(expr, Name):
        if expr.name in ("select", "setmetatable", "getmetatable", "newproxy"):
            return expr.name
    if isinstance(expr, BinOp) and "unpack" in names:
        return "unpack"
    if "getfenv" in names or "_ENV" in names:
        return "env"
    return None


def find_roles(root, dispatcher) -> Roles | None:
    host = None
    disp_fn = None
    for node in _walk(root):
        if not isinstance(node, Assign):
            continue
        for expr in node.exprs:
            if isinstance(expr, FunctionExpr) and any(item is dispatcher.while_stat for item in _walk(expr)):
                host, disp_fn = node, expr
                break
        if host is not None:
            break
    if disp_fn is None or len(disp_fn.params) < 4:
        return None
    roles = Roles(disp_fn.params[0], disp_fn.params[1], disp_fn.params[2], disp_fn.params[3])

    for stat in reversed(disp_fn.body.stats):
        if isinstance(stat, Return) and len(stat.exprs) == 1 and isinstance(stat.exprs[0], Call):
            call = stat.exprs[0]
            if isinstance(call.func, Name) and len(call.args) == 1 and isinstance(call.args[0], Name):
                roles.unpack = call.func.name
                roles.ret_reg = call.args[0].name
                break

    best = None
    for node in _walk(root):
        if not isinstance(node, Call):
            continue
        func = _strip(node.func)
        if isinstance(func, FunctionExpr) and any(item is host for item in _walk(func.body)):
            if best is None or len(list(_walk(func.body))) < len(list(_walk(best[0].body))):
                best = (func, node)
    if best is not None:
        func, call = best
        for param, arg in zip(func.params, call.args):
            kind = _classify_argument(arg)
            if kind == "select":
                roles.select = param
            elif kind == "unpack":
                roles.unpack = roles.unpack or param
            elif kind == "env":
                roles.env = param

    for target, expr in zip(host.targets, host.exprs):
        if not isinstance(target, Name) or not isinstance(expr, FunctionExpr):
            continue
        alloc = _is_alloc(expr)
        if alloc is not None:
            roles.alloc = target.name
            roles.refcount = alloc[1]
            continue
        release = _is_release(expr)
        if release is not None:
            roles.release = target.name
            roles.refcount = roles.refcount or release[0]
            roles.cells = release[1]
            continue
        if _is_retain(expr):
            roles.retain = target.name
    return roles


def rename_names(node, mapping: dict[str, str]):
    if isinstance(node, list):
        return [rename_names(item, mapping) for item in node]
    if isinstance(node, Name):
        return Name(mapping.get(node.name, node.name))
    if not dataclasses.is_dataclass(node) or isinstance(node, type):
        return node
    return type(node)(**{f.name: rename_names(getattr(node, f.name), mapping) for f in dataclasses.fields(node)})
