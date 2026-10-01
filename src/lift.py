from __future__ import annotations

import dataclasses
import re
from collections import Counter
from dataclasses import dataclass

from .ast_nodes import (
    Assign, Block, Call, CallStat, Expr, FunctionExpr, GenericFor, If, IfClause,
    Index, LocalAssign, MethodCall, Name, Number, NumericFor, Repeat, Stat,
    Table, While,
)
from .peephole import peephole
from .roles import ALLOC, CELLS, RELEASE, SLOTS
from .strings_layer import _walk

_UPVALUE = re.compile(r"^upvalue(\d+)$")


@dataclass
class LiftReport:
    lifted: int = 0
    unlifted: int = 0
    cells: int = 0
    removed: int = 0
    unresolved_slots: int = 0


class _Ctx:
    def __init__(self, upmap: dict[int, Expr], registers: set[str]) -> None:
        self.upmap = upmap
        self.registers = registers
        self.cells: list[str] = []


def _assigned(node) -> set[str]:
    names: set[str] = set()
    for item in _walk(node):
        if isinstance(item, Assign):
            names.update(t.name for t in item.targets if isinstance(t, Name))
        elif isinstance(item, LocalAssign):
            names.update(item.names)
        elif isinstance(item, NumericFor):
            names.add(item.var)
        elif isinstance(item, GenericFor):
            names.update(item.names)
    return names


class _Lifter:
    def __init__(self, cfg, results) -> None:
        self.cfg = cfg
        self.by_state = {r.function.entry_state: r for r in results}
        self.factories = set(cfg.factories)
        self.counter = 0
        self.stack: list[int] = []
        self.report = LiftReport()

    def new_cell(self, ctx: _Ctx) -> Name:
        self.counter += 1
        name = f"cell{self.counter}"
        ctx.cells.append(name)
        self.report.cells += 1
        return Name(name)

    def instance(self, state: int, upmap: dict[int, Expr]) -> FunctionExpr | None:
        result = self.by_state.get(state)
        if result is None or state in self.stack:
            return None
        self.stack.append(state)
        try:
            ctx = _Ctx(upmap, set(result.locals) | set(self.cfg.registers))
            body = self.block(result.stats, {}, ctx)
            declared = list(result.locals) + ctx.cells
            stats: list[Stat] = []
            if declared:
                stats.append(LocalAssign(declared, []))
            stats.extend(result.prologue)
            stats.extend(body)
            params, vararg = self.cfg.factory_params.get(result.function.factory or "", ([], True))
            names = [f"arg{i + 1}" for i in range(len(params))]
            return FunctionExpr(names, vararg, Block(stats))
        finally:
            self.stack.pop()

    def slot_value(self, expr: Expr, env: dict[str, Expr], ctx: _Ctx) -> Expr | None:
        if isinstance(expr, Name) and expr.name in env:
            return env[expr.name]
        if (
            isinstance(expr, Index) and isinstance(expr.obj, Name) and expr.obj.name == SLOTS
            and isinstance(expr.key, Number) and expr.key.text.isdigit()
            and int(expr.key.text) in ctx.upmap
        ):
            return ctx.upmap[int(expr.key.text)]
        if self.is_alloc(expr, ctx):
            return self.new_cell(ctx)
        return None

    def is_alloc(self, expr: Expr, ctx: _Ctx) -> bool:
        return (
            isinstance(expr, Call) and isinstance(expr.func, Name) and expr.func.name == ALLOC
            and not expr.args
        )

    def block(self, stats: list[Stat], env: dict[str, Expr], ctx: _Ctx) -> list[Stat]:
        out: list[Stat] = []
        for stat in stats:
            out.extend(self.stat(stat, env, ctx))
        return out

    def stat(self, stat: Stat, env: dict[str, Expr], ctx: _Ctx) -> list[Stat]:
        if isinstance(stat, Assign):
            return self.assign(stat, env, ctx)
        if isinstance(stat, CallStat):
            call = stat.call
            if (
                isinstance(call, Call) and isinstance(call.func, Name) and call.func.name == RELEASE
                and len(call.args) == 1
                and isinstance(call.args[0], Name) and call.args[0].name in env
            ):
                return []
            return [CallStat(self.rx(call, env, ctx))]
        if isinstance(stat, If):
            clauses = []
            envs = []
            for clause in stat.clauses:
                cond = self.rx(clause.cond, env, ctx)
                inner = dict(env)
                body = self.block(clause.body.stats, inner, ctx)
                clauses.append(IfClause(cond, Block(body)))
                envs.append(inner)
            orelse = None
            if stat.orelse is not None:
                inner = dict(env)
                orelse = Block(self.block(stat.orelse.stats, inner, ctx))
                envs.append(inner)
            else:
                envs.append(dict(env))
            merged = {k: v for k, v in env.items() if all(e.get(k) == v for e in envs)}
            env.clear()
            env.update(merged)
            return [If(clauses, orelse)]
        if isinstance(stat, (While, NumericFor, GenericFor, Repeat)):
            killed = _assigned(stat)
            inner = {k: v for k, v in env.items() if k not in killed}
            values = {}
            for f in dataclasses.fields(stat):
                value = getattr(stat, f.name)
                if isinstance(value, Block):
                    before = len(ctx.cells)
                    body = self.block(value.stats, dict(inner), ctx)
                    created = ctx.cells[before:]
                    del ctx.cells[before:]
                    if created:
                        body = [LocalAssign(created, [])] + body
                    values[f.name] = Block(body)
                else:
                    values[f.name] = self.rx(value, inner, ctx)
            for name in killed:
                env.pop(name, None)
            return [type(stat)(**values)]
        values = {}
        for f in dataclasses.fields(stat):
            value = getattr(stat, f.name)
            if isinstance(value, Block):
                values[f.name] = Block(self.block(value.stats, dict(env), ctx))
            else:
                values[f.name] = self.rx(value, env, ctx)
        for name in _assigned(stat):
            env.pop(name, None)
        return [type(stat)(**values)]

    def assign(self, stat: Assign, env: dict[str, Expr], ctx: _Ctx) -> list[Stat]:
        if len(stat.targets) == 1 and len(stat.exprs) == 1 and isinstance(stat.targets[0], Name):
            register = stat.targets[0].name
            expr = stat.exprs[0]
            if register not in ctx.registers or not _UPVALUE.match(register):
                if self.is_alloc(expr, ctx):
                    env[register] = self.new_cell(ctx)
                    return []
                if isinstance(expr, Name) and expr.name in env:
                    env[register] = env[expr.name]
                    return []
                if (
                    isinstance(expr, Index) and isinstance(expr.obj, Name) and expr.obj.name == SLOTS
                    and isinstance(expr.key, Number) and expr.key.text.isdigit()
                    and int(expr.key.text) in ctx.upmap
                ):
                    env[register] = ctx.upmap[int(expr.key.text)]
                    return []
        targets = [self.rx(t, env, ctx) for t in stat.targets]
        exprs = [self.rx(e, env, ctx) for e in stat.exprs]
        for original in stat.targets:
            if isinstance(original, Name):
                env.pop(original.name, None)
        return [Assign(targets, exprs)]

    def rx(self, node, env: dict[str, Expr], ctx: _Ctx):
        if isinstance(node, list):
            return [self.rx(item, env, ctx) for item in node]
        if isinstance(node, FunctionExpr):
            return node
        if isinstance(node, Name):
            match = _UPVALUE.match(node.name)
            if match and int(match.group(1)) in ctx.upmap and node.name not in ctx.registers:
                return ctx.upmap[int(match.group(1))]
            return node
        if isinstance(node, Index) and isinstance(node.obj, Name) and node.obj.name == CELLS:
            key = node.key
            if isinstance(key, Name) and key.name in env:
                return env[key.name]
            if self.is_alloc(key, ctx):
                return self.new_cell(ctx)
        if isinstance(node, Call):
            lifted = self.lift_call(node, env, ctx)
            if lifted is not None:
                return lifted
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        return type(node)(**{f.name: self.rx(getattr(node, f.name), env, ctx) for f in dataclasses.fields(node)})

    def lift_call(self, node: Call, env: dict[str, Expr], ctx: _Ctx) -> Expr | None:
        func = node.func
        if not (isinstance(func, Name) and func.name in self.factories and func.name not in ctx.registers):
            return None
        if len(node.args) != 2 or not isinstance(node.args[0], Number) or not isinstance(node.args[1], Table):
            return None
        if not node.args[0].text.isdigit():
            return None
        table = node.args[1]
        ups: dict[int, Expr] = {}
        for position, item in enumerate(table.fields, 1):
            if item.key is not None:
                self.report.unlifted += 1
                return None
            value = self.slot_value(item.value, env, ctx)
            if value is None:
                self.report.unlifted += 1
                return None
            ups[position] = value
        built = self.instance(int(node.args[0].text), ups)
        if built is None:
            self.report.unlifted += 1
            return None
        self.report.lifted += 1
        return built


def _pure(expr) -> bool:
    if isinstance(expr, FunctionExpr):
        return True
    if isinstance(expr, (Call, MethodCall)):
        return False
    if isinstance(expr, list):
        return all(_pure(item) for item in expr)
    if dataclasses.is_dataclass(expr) and not isinstance(expr, type):
        return all(_pure(getattr(expr, f.name)) for f in dataclasses.fields(expr))
    return True


def _drop(node, dead: set[str]):
    if isinstance(node, list):
        out = []
        for item in node:
            if (
                isinstance(item, Assign) and len(item.targets) == 1 and len(item.exprs) == 1
                and isinstance(item.targets[0], Name) and item.targets[0].name in dead
            ):
                if _pure(item.exprs[0]):
                    continue
                expr = item.exprs[0]
                if isinstance(expr, (Call, MethodCall)):
                    out.append(CallStat(_drop(expr, dead)))
                    continue
            out.append(_drop(item, dead))
        return out
    if not dataclasses.is_dataclass(node) or isinstance(node, type):
        return node
    return type(node)(**{f.name: _drop(getattr(node, f.name), dead) for f in dataclasses.fields(node)})


def _prune_declarations(node, live: set[str]):
    if isinstance(node, list):
        out = []
        for item in node:
            if isinstance(item, LocalAssign) and not item.exprs:
                names = [n for n in item.names if not n.startswith("cell") or n in live]
                if not names:
                    continue
                out.append(LocalAssign(names, []))
                continue
            out.append(_prune_declarations(item, live))
        return out
    if not dataclasses.is_dataclass(node) or isinstance(node, type):
        return node
    return type(node)(**{f.name: _prune_declarations(getattr(node, f.name), live) for f in dataclasses.fields(node)})


def remove_dead_cells(stats: list, report: LiftReport) -> list:
    for _ in range(64):
        reads: Counter[str] = Counter()
        writes: Counter[str] = Counter()
        for node in _walk(stats):
            if isinstance(node, Name) and node.name.startswith("cell"):
                reads[node.name] += 1
            elif isinstance(node, Assign):
                for target in node.targets:
                    if isinstance(target, Name) and target.name.startswith("cell"):
                        writes[target.name] += 1
        dead = {name for name in writes if reads[name] == writes[name]}
        if not dead:
            break
        before = repr(stats)
        stats = _drop(stats, dead)
        if repr(stats) == before:
            break
        report.removed += len(dead)
    live = {name for name in reads if reads[name] > writes[name] or name in reads}
    return _prune_declarations(stats, {n for n in live})


def lift_program(cfg, results) -> tuple[list[Stat] | None, LiftReport]:
    lifter = _Lifter(cfg, results)
    referenced: set[int] = set()
    for result in results:
        for node in _walk(result.stats):
            if (
                isinstance(node, Call) and isinstance(node.func, Name) and node.func.name in lifter.factories
                and node.args and isinstance(node.args[0], Number) and node.args[0].text.isdigit()
            ):
                referenced.add(int(node.args[0].text))
    main = None
    for result in results:
        vararg = cfg.factory_params.get(result.function.factory or "", ([], False))[1]
        if vararg and result.function.entry_state not in referenced:
            main = result
            break
    if main is None:
        return None, lifter.report
    built = lifter.instance(main.function.entry_state, {})
    if built is None:
        return None, lifter.report
    stats = remove_dead_cells(built.body.stats, lifter.report)
    stats = peephole(stats, set())
    lifter.report.unresolved_slots = sum(
        1 for node in _walk(stats)
        if isinstance(node, Index) and isinstance(node.obj, Name) and node.obj.name == CELLS
    )
    return stats, lifter.report
