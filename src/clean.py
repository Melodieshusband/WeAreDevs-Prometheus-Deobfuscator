from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .ast_nodes import (
    Assign, Block, Break, Call, CallStat, Continue, Do, Expr, Field, FunctionExpr,
    GenericFor, If, IfClause, Index, LocalAssign, MethodCall, Name, NumericFor,
    Repeat, Return, Stat, String, Table, TrueLit, While,
)
from .peephole import _declared, _kills, _reads, _ue, peephole
from .simplify import defs_of, simplify_stats
from .strings_layer import _children, _walk

_PURE_PREFIXES = ("math.", "string.", "bit32.", "utf8.")
_PURE_NAMES = {
    "tostring", "tonumber", "type", "select", "unpack", "table.unpack", "table.concat",
    "ipairs", "pairs", "next", "rawget", "rawequal", "rawlen", "getmetatable",
    "os.time", "os.clock", "os.date", "table.pack", "table.find",
}
_MUTATORS = {"table.insert", "table.remove", "table.sort", "table.move", "setmetatable", "rawset", "table.clear"}
_LOOPS = (While, NumericFor, GenericFor, Repeat)


def _strip(expr):
    return expr.inner if hasattr(expr, "inner") else expr


def path_of(expr) -> str | None:
    expr = _strip(expr)
    if isinstance(expr, Name):
        return expr.name
    if isinstance(expr, Field):
        base = path_of(expr.obj)
        return f"{base}.{expr.name}" if base else None
    if isinstance(expr, Index) and isinstance(expr.key, String):
        base = path_of(expr.obj)
        return f"{base}.{expr.key.value}" if base else None
    return None


def base_name(expr) -> str | None:
    expr = _strip(expr)
    if isinstance(expr, Name):
        return expr.name
    if isinstance(expr, (Index, Field)):
        return base_name(expr.obj)
    return None


def _classify(path: str | None) -> str:
    if path is None:
        return "impure"
    if path in _MUTATORS:
        return "mutator"
    if path in _PURE_NAMES or path.startswith(_PURE_PREFIXES):
        return "pure"
    return "impure"


def _scope_nodes(stats):
    stack = list(reversed(stats)) if isinstance(stats, list) else [stats]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(reversed(node))
            continue
        yield node
        if isinstance(node, FunctionExpr):
            continue
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            for item in reversed(list(_children(node))):
                stack.append(item)


def collect_scope(stats: list, params: list[str]):
    locals_: set[str] = set(params)
    paths: dict[str, set] = {}
    for node in _scope_nodes(stats):
        if isinstance(node, LocalAssign):
            locals_.update(node.names)
        elif isinstance(node, NumericFor):
            locals_.add(node.var)
        elif isinstance(node, GenericFor):
            locals_.update(node.names)
        if isinstance(node, Assign):
            if len(node.targets) == len(node.exprs):
                for target, expr in zip(node.targets, node.exprs):
                    if isinstance(target, Name):
                        paths.setdefault(target.name, set()).add(path_of(expr))
            else:
                for target in node.targets:
                    if isinstance(target, Name):
                        paths.setdefault(target.name, set()).add(None)
        elif isinstance(node, LocalAssign) and node.exprs and len(node.exprs) == len(node.names):
            for name, expr in zip(node.names, node.exprs):
                paths.setdefault(name, set()).add(path_of(expr))
        elif isinstance(node, LocalAssign):
            for name in node.names:
                if node.exprs:
                    paths.setdefault(name, set()).add(None)
    aliases = {
        name: next(iter(values))
        for name, values in paths.items()
        if len(values) == 1 and None not in values
    }
    return locals_, aliases


class _Dce:
    def __init__(self, locals_: set[str], aliases: dict[str, str]) -> None:
        self.locals = locals_
        self.aliases = aliases
        self.flow: dict[str, str] = {}

    def callee_path(self, func) -> str | None:
        func = _strip(func)
        if isinstance(func, Name):
            if func.name in self.flow:
                return self.flow[func.name]
            if func.name in self.aliases:
                return self.aliases[func.name]
            if func.name in self.locals:
                return None
            return func.name
        path = path_of(func)
        if path is None:
            return None
        head = path.split(".")[0]
        if head in self.locals and head not in self.aliases:
            return None
        if head in self.aliases:
            return self.aliases[head] + path[len(head):]
        return path

    def effects(self, node) -> tuple[bool, set[str]]:
        ok = [True]
        mutated: set[str] = set()

        def visit(item) -> None:
            if isinstance(item, list):
                for sub in item:
                    visit(sub)
                return
            if isinstance(item, FunctionExpr):
                return
            if isinstance(item, Call):
                kind = _classify(self.callee_path(item.func))
                if kind == "impure":
                    ok[0] = False
                elif kind == "mutator":
                    first = _strip(item.args[0]) if item.args else None
                    if isinstance(first, Table):
                        pass
                    else:
                        base = base_name(first) if first is not None else None
                        if base is None:
                            ok[0] = False
                        else:
                            mutated.add(base)
            elif isinstance(item, MethodCall):
                ok[0] = False
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                for child in _children(item):
                    visit(child)

        visit(node)
        return ok[0], mutated

    def dead_local(self, name: str | None, live: set[str]) -> bool:
        return name is not None and name in self.locals and name not in live

    def removable(self, stat: Stat, live: set[str], in_loop: bool = False) -> bool:
        if isinstance(stat, Assign):
            for target in stat.targets:
                if isinstance(target, Name):
                    if not self.dead_local(target.name, live):
                        return False
                else:
                    if not self.dead_local(base_name(target), live):
                        return False
            ok, mutated = self.effects([stat.exprs, [t for t in stat.targets if not isinstance(t, Name)]])
            return ok and all(self.dead_local(m, live) for m in mutated)
        if isinstance(stat, LocalAssign):
            if not stat.exprs:
                return False
            if any(name in live for name in stat.names):
                return False
            ok, mutated = self.effects(stat.exprs)
            return ok and all(self.dead_local(m, live) for m in mutated)
        if isinstance(stat, CallStat):
            ok, mutated = self.effects(stat.call)
            return ok and all(self.dead_local(m, live) for m in mutated)
        if isinstance(stat, If):
            conds = [c.cond for c in stat.clauses]
            ok, mutated = self.effects(conds)
            if not ok or not all(self.dead_local(m, live) for m in mutated):
                return False
            bodies = [c.body.stats for c in stat.clauses]
            if stat.orelse is not None:
                bodies.append(stat.orelse.stats)
            return all(self.removable(s, live, in_loop) for body in bodies for s in body)
        if isinstance(stat, _LOOPS):
            values = [getattr(stat, f.name) for f in dataclasses.fields(stat) if not isinstance(getattr(stat, f.name), Block)]
            ok, mutated = self.effects(values)
            if not ok or not all(self.dead_local(m, live) for m in mutated):
                return False
            return all(self.removable(s, live, True) for s in stat.body.stats)
        if isinstance(stat, Do):
            return all(self.removable(s, live, in_loop) for s in stat.body.stats)
        if isinstance(stat, (Break, Continue)):
            return in_loop
        return False

    def advance(self, env: dict[str, str], stat: Stat) -> dict[str, str]:
        env = dict(env)
        if (
            isinstance(stat, Assign) and len(stat.targets) == len(stat.exprs)
            and all(isinstance(t, Name) for t in stat.targets)
        ):
            resolved: list[str | None] = []
            for expr in stat.exprs:
                inner = _strip(expr)
                if isinstance(inner, Name) and inner.name in env:
                    resolved.append(env[inner.name])
                else:
                    resolved.append(path_of(inner))
            for target, path in zip(stat.targets, resolved):
                if path is None:
                    env.pop(target.name, None)
                else:
                    env[target.name] = path
            return env
        for name in defs_of(stat):
            env.pop(name, None)
        return env

    def block(self, stats: list[Stat], live_out: set[str], env: dict[str, str] | None = None) -> tuple[list[Stat], set[str]]:
        envs: list[dict[str, str]] = []
        current = dict(env or {})
        for stat in stats:
            envs.append(current)
            current = self.advance(current, stat)
        live = set(live_out)
        kept: list[Stat] = []
        for index in range(len(stats) - 1, -1, -1):
            stat = stats[index]
            self.flow = envs[index]
            if self.removable(stat, live):
                continue
            if (
                isinstance(stat, Assign) and len(stat.targets) == 1 and len(stat.exprs) == 1
                and isinstance(stat.targets[0], Name) and self.dead_local(stat.targets[0].name, live)
                and isinstance(stat.exprs[0], Call)
            ):
                stat = CallStat(stat.exprs[0])
            new, live = self.stat(stat, live, envs[index])
            kept.append(new)
        kept.reverse()
        return kept, live

    def stat(self, stat: Stat, live: set[str], env: dict[str, str]) -> tuple[Stat, set[str]]:
        if isinstance(stat, If):
            clauses = []
            union: set[str] = set()
            for clause in stat.clauses:
                body, body_live = self.block(clause.body.stats, live, env)
                clauses.append(IfClause(self.expr(clause.cond), Block(body)))
                union |= body_live | _reads(clause.cond)
            orelse = None
            if stat.orelse is not None:
                body, body_live = self.block(stat.orelse.stats, live, env)
                orelse = Block(body)
                union |= body_live
            else:
                union |= live
            self.flow = env
            return If(clauses, orelse), union
        if isinstance(stat, _LOOPS):
            inner = live | _ue(stat)
            killed = defs_of(stat)
            loop_env = {k: v for k, v in env.items() if k not in killed}
            values = {}
            for f in dataclasses.fields(stat):
                value = getattr(stat, f.name)
                if isinstance(value, Block):
                    body, _ = self.block(value.stats, inner, loop_env)
                    values[f.name] = Block(body)
                else:
                    values[f.name] = self.expr(value)
            self.flow = env
            return type(stat)(**values), inner
        new = self.expr(stat)
        if isinstance(new, Return):
            return new, _ue(new)
        return new, (live - _kills(new)) | _ue(new)

    def expr(self, node):
        if isinstance(node, list):
            return [self.expr(item) for item in node]
        if isinstance(node, FunctionExpr):
            return dce_function(node)
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        if isinstance(node, Block):
            return Block(self.block(node.stats, set())[0])
        return type(node)(**{f.name: self.expr(getattr(node, f.name)) for f in dataclasses.fields(node)})


def _prune_declarations(stats: list[Stat]) -> list[Stat]:
    counts: dict[str, int] = {}
    for node in _walk(stats):
        if isinstance(node, Name):
            counts[node.name] = counts.get(node.name, 0) + 1
        elif isinstance(node, Assign):
            for target in node.targets:
                pass
    result: list[Stat] = []
    for stat in stats:
        if isinstance(stat, LocalAssign) and not stat.exprs:
            names = [n for n in stat.names if counts.get(n, 0) > 0]
            if names:
                result.append(LocalAssign(names, []))
            continue
        result.append(stat)
    return result


def dce_function(fn: FunctionExpr) -> FunctionExpr:
    locals_, aliases = collect_scope(fn.body.stats, fn.params)
    engine = _Dce(locals_, aliases)
    body, _ = engine.block(fn.body.stats, set())
    return FunctionExpr(fn.params, fn.is_vararg, Block(_prune_declarations(body)))


def dce_program(stats: list[Stat]) -> list[Stat]:
    locals_, aliases = collect_scope(stats, [])
    engine = _Dce(locals_, aliases)
    body, _ = engine.block(stats, set())
    return _prune_declarations(body)


def simplify_tree(stats: list[Stat], live_out: set[str]) -> list[Stat]:
    count = len(stats)
    live_after: list[set[str]] = [set() for _ in range(count)]
    live = set(live_out)
    for i in range(count - 1, -1, -1):
        live_after[i] = set(live)
        live = _ue(stats[i]) if isinstance(stats[i], Return) else (live - _kills(stats[i])) | _ue(stats[i])
    rebuilt = [_simplify_stat(stat, live_after[i]) for i, stat in enumerate(stats)]
    return simplify_stats(rebuilt, live_out)


def _simplify_expr(node):
    if isinstance(node, list):
        return [_simplify_expr(item) for item in node]
    if isinstance(node, FunctionExpr):
        return FunctionExpr(node.params, node.is_vararg, Block(simplify_tree(node.body.stats, set())))
    if not dataclasses.is_dataclass(node) or isinstance(node, type):
        return node
    return type(node)(**{f.name: _simplify_expr(getattr(node, f.name)) for f in dataclasses.fields(node)})


def _simplify_stat(stat: Stat, live_after: set[str]) -> Stat:
    if isinstance(stat, If):
        clauses = [
            IfClause(_simplify_expr(c.cond), Block(simplify_tree(c.body.stats, live_after)))
            for c in stat.clauses
        ]
        orelse = Block(simplify_tree(stat.orelse.stats, live_after)) if stat.orelse is not None else None
        return If(clauses, orelse)
    if isinstance(stat, _LOOPS):
        inner = live_after | _ue(stat)
        values = {}
        for f in dataclasses.fields(stat):
            value = getattr(stat, f.name)
            if isinstance(value, Block):
                values[f.name] = Block(simplify_tree(value.stats, inner))
            else:
                values[f.name] = _simplify_expr(value)
        return type(stat)(**values)
    return _simplify_expr(stat)


def _trap_like(block: Block) -> bool:
    for node in _walk(block):
        if isinstance(node, While) and isinstance(node.cond, TrueLit):
            if not any(isinstance(item, Break) for item in _walk(node.body)):
                return True
    return False


def _find_tamper(stats: list[Stat]):
    for index in range(len(stats) - 1, -1, -1):
        stat = stats[index]
        if not (isinstance(stat, If) and len(stat.clauses) == 1 and stat.orelse is not None):
            continue
        if not _trap_like(stat.orelse):
            continue
        head = 0
        while head < index and isinstance(stats[head], LocalAssign) and not stats[head].exprs:
            head += 1
        return stats[:head], stats[head:index], list(stat.clauses[0].body.stats)
    return None


def _uninitialized(stats: list[Stat]) -> set[str]:
    declared: set[str] = set()
    for stat in stats:
        if isinstance(stat, LocalAssign) and not stat.exprs:
            declared.update(stat.names)
    assigned: set[str] = set()
    used: set[str] = set()
    for node in _walk(stats):
        if isinstance(node, Name):
            used.add(node.name)
        elif isinstance(node, Assign):
            assigned.update(t.name for t in node.targets if isinstance(t, Name))
        elif isinstance(node, LocalAssign) and node.exprs:
            assigned.update(node.names)
        elif isinstance(node, NumericFor):
            assigned.add(node.var)
        elif isinstance(node, GenericFor):
            assigned.update(node.names)
        elif isinstance(node, FunctionExpr):
            assigned.update(node.params)
    return {name for name in declared if name in used and name not in assigned}


def _slice(preamble: list[Stat], needed: set[str]) -> list[Stat]:
    needed = set(needed)
    kept: list[Stat] = []
    for item in reversed(preamble):
        if defs_of(item) & needed:
            kept.append(item)
            needed |= _reads(item)
    kept.reverse()
    return kept


def _run_passes(current: list[Stat]) -> list[Stat]:
    for _ in range(12):
        before = repr(current)
        current = simplify_tree(current, set())
        current = dce_program(current)
        current = peephole(current, set())
        if repr(current) == before:
            break
    return current


def clean_program(stats: list[Stat]) -> tuple[list[Stat], bool]:
    parts = _find_tamper(stats)
    if parts is None:
        return _strip_tail(_run_passes(stats)), False
    declarations, preamble, branch = parts
    result = _run_passes(declarations + branch)
    missing = _uninitialized(result)
    attempts = 0
    while missing and attempts < 6:
        attempts += 1
        kept = _slice(preamble, missing)
        result = _run_passes(declarations + kept + branch)
        again = _uninitialized(result)
        if again == missing:
            break
        missing = again
    return _strip_tail(result), True


def _strip_tail(current: list[Stat]) -> list[Stat]:
    while current and isinstance(current[-1], Return) and not current[-1].exprs:
        current = current[:-1]
    return current
