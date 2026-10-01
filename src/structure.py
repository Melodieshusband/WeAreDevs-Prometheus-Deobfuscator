from __future__ import annotations

import dataclasses
import keyword
from dataclasses import dataclass

from .ast_nodes import (
    FalseLit, Assign, BinOp, Block, Break, Call, CallStat, Continue, Expr, Field, If,
    IfClause, Index, LocalAssign, Name, Number, Return, Stat, String,
    TrueLit, While,
)
from .graph import EXIT_NODE, Cfg, CfgNode, VmFunction
from .roles import ARGS, CELLS, ENV, SELECT, SLOTS, UNPACK, RET
from .simplify import (
    PSEUDO_COND, defs_of, has_call, merge_return, negate, simplify_stats, uses_of,
)
from .strings_layer import _walk


class Unstructured(Exception):
    pass


@dataclass
class _Loop:
    header: int
    body: frozenset[int]
    exit: int | None
    secondary: dict[int, int] = dataclasses.field(default_factory=dict)
    flag: str = ""


@dataclass
class PreparedNode:
    stats: list[Stat]
    cond: Expr | None


def _return_stat() -> Stat:
    return Return([Call(Name(UNPACK), [Name(RET)])])


def _raw_node_stats(node: CfgNode) -> list[Stat]:
    stats = list(node.body)
    if node.term_kind == "cond":
        stats.append(Assign([Name(PSEUDO_COND)], [node.cond_expr]))
    elif node.term_kind == "exit":
        stats.append(_return_stat())
    return stats


def _liveness(cfg: Cfg, function: VmFunction) -> dict[int, set[str]]:
    members = set(function.nodes)
    raw = {i: _raw_node_stats(cfg.nodes[i]) for i in members}
    extra: dict[int, set[str]] = {}
    for i in members:
        node = cfg.nodes[i]
        extra[i] = {RET} if node.term_kind == "cond" and EXIT_NODE in (node.true_node, node.false_node) else set()
    live_in: dict[int, set[str]] = {i: set() for i in members}
    live_out: dict[int, set[str]] = {i: set() for i in members}
    changed = True
    while changed:
        changed = False
        for i in members:
            node = cfg.nodes[i]
            out = set(extra[i])
            for successor in node.successors:
                if successor in members:
                    out |= live_in[successor]
            live = set(out)
            for stat in reversed(raw[i]):
                live = (live - defs_of(stat)) | uses_of(stat)
            if out != live_out[i] or live != live_in[i]:
                live_out[i] = out
                live_in[i] = live
                changed = True
    return live_out


def prepare_nodes(cfg: Cfg, function: VmFunction) -> dict[int, PreparedNode]:
    live_out = _liveness(cfg, function)
    prepared: dict[int, PreparedNode] = {}
    for i in function.nodes:
        node = cfg.nodes[i]
        stats = simplify_stats(_raw_node_stats(node), live_out[i] | {PSEUDO_COND})
        cond = None
        if node.term_kind == "cond":
            last = stats.pop()
            cond = last.exprs[0]
        if node.term_kind == "exit":
            stats = merge_return(stats, UNPACK, RET)
        prepared[i] = PreparedNode(stats, cond)
    return prepared


def _successors(cfg: Cfg, index: int) -> list[int]:
    node = cfg.nodes[index]
    if node.term_kind == "goto":
        return [node.goto_node]
    if node.term_kind == "cond":
        return [node.true_node, node.false_node]
    return []


class _Structurer:
    def __init__(self, cfg: Cfg, function: VmFunction, prepared: dict[int, PreparedNode]) -> None:
        self.cfg = cfg
        self.prepared = prepared
        self.entry = function.entry_node
        self.members = set(function.nodes)
        self.succ = {
            i: [s for s in _successors(cfg, i) if s != EXIT_NODE and s in self.members]
            for i in self.members
        }
        self.pred: dict[int, list[int]] = {i: [] for i in self.members}
        for i, targets in self.succ.items():
            for target in targets:
                self.pred[target].append(i)
        self.dom = self._dominators()
        self.pdom = self._postdominators()
        self.loops = self._loops()

    def _dominators(self) -> dict[int, set[int]]:
        nodes = self.members
        dom = {n: set(nodes) for n in nodes}
        dom[self.entry] = {self.entry}
        changed = True
        while changed:
            changed = False
            for n in nodes:
                if n == self.entry:
                    continue
                incoming = [dom[p] for p in self.pred[n]]
                new = {n} | (set.intersection(*incoming) if incoming else set())
                if new != dom[n]:
                    dom[n] = new
                    changed = True
        return dom

    def _postdominators(self) -> dict[int, set[int]]:
        nodes = self.members | {EXIT_NODE}
        successors: dict[int, list[int]] = {}
        for n in self.members:
            targets = list(self.succ[n])
            node = self.cfg.nodes[n]
            if node.term_kind == "exit" or EXIT_NODE in _successors(self.cfg, n):
                targets.append(EXIT_NODE)
            successors[n] = targets
        pdom = {n: set(nodes) for n in nodes}
        pdom[EXIT_NODE] = {EXIT_NODE}
        changed = True
        while changed:
            changed = False
            for n in self.members:
                outgoing = [pdom[s] for s in successors[n]]
                new = {n} | (set.intersection(*outgoing) if outgoing else set())
                if new != pdom[n]:
                    pdom[n] = new
                    changed = True
        return pdom

    def _ipdom(self, n: int) -> int | None:
        candidates = self.pdom[n] - {n}
        for c in candidates:
            if self.pdom[c] == candidates:
                return c
        return None

    def _loops(self) -> dict[int, frozenset[int]]:
        loops: dict[int, set[int]] = {}
        for u in self.members:
            for h in self.succ[u]:
                if h in self.dom[u]:
                    body = loops.setdefault(h, {h})
                    stack = [u]
                    while stack:
                        m = stack.pop()
                        if m in body:
                            continue
                        body.add(m)
                        stack.extend(self.pred[m])
        return {h: frozenset(b) for h, b in loops.items()}

    def _pick_exit(self, header: int, body: frozenset[int]) -> tuple[int | None, list[int]]:
        targets = {s for m in body for s in self.succ[m] if s not in body}
        if not targets:
            return None, []
        if len(targets) == 1:
            return next(iter(targets)), []
        ipdom = self._ipdom(header)
        primary = ipdom if ipdom in targets else min(targets)
        return primary, sorted(targets - {primary})

    def build(self) -> list[Stat]:
        return self.region(self.entry, frozenset(), [], False)

    def region(self, n: int, stop: frozenset[int], loops: list[_Loop], entering: bool) -> list[Stat]:
        out: list[Stat] = []
        while True:
            if n == EXIT_NODE:
                out.append(_return_stat())
                return out
            if n in stop:
                return out
            if loops:
                current = loops[-1]
                if n == current.header and not entering:
                    out.append(Continue())
                    return out
                if n not in current.body:
                    if n == current.exit:
                        out.append(Break())
                        return out
                    if n in current.secondary:
                        out.append(Assign([Name(current.flag)], [Number(str(current.secondary[n]))]))
                        out.append(Break())
                        return out
                    raise Unstructured()
            if n in self.loops and not (loops and loops[-1].header == n):
                body = self.loops[n]
                primary, others = self._pick_exit(n, body)
                loop = _Loop(n, body, primary)
                if others:
                    loop.flag = f"__brk{n}"
                    loop.secondary = {t: k + 1 for k, t in enumerate(others)}
                    out.append(Assign([Name(loop.flag)], [Number("0")]))
                inner = self.region(n, stop, loops + [loop], True)
                out.append(While(TrueLit(), Block(_strip_tail_continue(inner))))
                if loop.secondary:
                    join = self._ipdom(n)
                    if join == EXIT_NODE:
                        join = None
                    if join is not None and loops and join not in loops[-1].body:
                        join = None
                    inner_stop = stop | {join} if join is not None else stop
                    clauses = []
                    for target, code in loop.secondary.items():
                        arm = self.region(target, inner_stop, loops, False)
                        clauses.append(IfClause(BinOp("==", Name(loop.flag), Number(str(code))), Block(arm)))
                    orelse_body = self.region(primary, inner_stop, loops, False) if primary is not None else []
                    out.append(If(clauses, Block(orelse_body) if orelse_body else None))
                    if join is None:
                        return out
                    n = join
                    entering = False
                    continue
                if loop.exit is None:
                    return out
                n = loop.exit
                entering = False
                continue
            entering = False
            node = self.cfg.nodes[n]
            prepared = self.prepared[n]
            out.extend(prepared.stats)
            if node.term_kind == "exit":
                return out
            if node.term_kind == "goto":
                n = node.goto_node
                continue
            join = self._ipdom(n)
            if join == EXIT_NODE:
                join = None
            if join is not None and loops and join not in loops[-1].body:
                join = None
            inner_stop = stop | {join} if join is not None else stop
            then_body = self.region(node.true_node, inner_stop, loops, False)
            else_body = self.region(node.false_node, inner_stop, loops, False)
            cond = prepared.cond
            if not then_body and else_body:
                cond = negate(cond)
                then_body, else_body = else_body, []
            out.append(If([IfClause(cond, Block(then_body))], Block(else_body) if else_body else None))
            if join is None:
                return out
            n = join


def _strip_tail_continue(stats: list[Stat]) -> list[Stat]:
    if not stats:
        return stats
    last = stats[-1]
    if isinstance(last, Continue):
        return _strip_tail_continue(stats[:-1])
    if isinstance(last, If):
        clauses = [IfClause(c.cond, Block(_strip_tail_continue(c.body.stats))) for c in last.clauses]
        orelse = Block(_strip_tail_continue(last.orelse.stats)) if last.orelse is not None else None
        if orelse is not None and not orelse.stats:
            orelse = None
        return stats[:-1] + [If(clauses, orelse)]
    return stats


def _is_break_only(block: Block | None) -> bool:
    return block is not None and len(block.stats) == 1 and isinstance(block.stats[0], Break)


def simplify_loops(stats: list[Stat]) -> list[Stat]:
    result: list[Stat] = []
    for stat in stats:
        if isinstance(stat, If):
            clauses = [IfClause(c.cond, Block(simplify_loops(c.body.stats))) for c in stat.clauses]
            orelse = Block(simplify_loops(stat.orelse.stats)) if stat.orelse is not None else None
            result.append(If(clauses, orelse))
        elif isinstance(stat, While):
            body = simplify_loops(stat.body.stats)
            if isinstance(stat.cond, TrueLit) and body and isinstance(body[0], If):
                first = body[0]
                if len(first.clauses) == 1:
                    clause = first.clauses[0]
                    if _is_break_only(first.orelse):
                        result.append(While(clause.cond, Block(clause.body.stats + body[1:])))
                        continue
                    if first.orelse is None and _is_break_only(clause.body):
                        result.append(While(negate(clause.cond), Block(body[1:])))
                        continue
            result.append(While(stat.cond, Block(body)))
        else:
            result.append(stat)
    return result


def cleanup(stats: list[Stat]) -> list[Stat]:
    result: list[Stat] = []
    for stat in stats:
        if isinstance(stat, If):
            clauses = [IfClause(c.cond, Block(cleanup(c.body.stats))) for c in stat.clauses]
            orelse = Block(cleanup(stat.orelse.stats)) if stat.orelse is not None else None
            if orelse is not None and not orelse.stats:
                orelse = None
            if len(clauses) == 1:
                clause = clauses[0]
                if isinstance(clause.cond, TrueLit):
                    result.extend(clause.body.stats)
                    continue
                if isinstance(clause.cond, FalseLit):
                    if orelse is not None:
                        result.extend(orelse.stats)
                    continue
                if not clause.body.stats and orelse is None and not has_call(clause.cond):
                    continue
                if not clause.body.stats and orelse is not None:
                    clauses = [IfClause(negate(clause.cond), orelse)]
                    orelse = None
            result.append(If(clauses, orelse))
        elif isinstance(stat, While):
            result.append(While(stat.cond, Block(cleanup(stat.body.stats))))
        else:
            result.append(stat)
    return result


def state_machine(cfg: Cfg, function: VmFunction, prepared: dict[int, PreparedNode]) -> list[Stat]:
    clauses: list[IfClause] = []
    for i in function.nodes:
        node = cfg.nodes[i]
        stats = list(prepared[i].stats)
        if node.term_kind == "goto":
            stats.append(_set_pc(node.goto_node))
        elif node.term_kind == "cond":
            stats.append(
                If(
                    [IfClause(prepared[i].cond, Block([_set_pc(node.true_node)]))],
                    Block([_set_pc(node.false_node)]),
                )
            )
        clauses.append(IfClause(BinOp("==", Name("pc"), Number(str(i))), Block(stats)))
    dispatch = If(clauses, None)
    return [
        LocalAssign(["pc"], [Number(str(function.entry_node))]),
        While(TrueLit(), Block([dispatch])),
    ]


def _set_pc(target: int) -> Stat:
    if target == EXIT_NODE:
        return _return_stat()
    return Assign([Name("pc")], [Number(str(target))])


_KEYWORDS = set(keyword.kwlist) | {
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "if", "in",
    "local", "nil", "not", "or", "repeat", "return", "then", "true", "until", "while", "continue",
}


def _rewrite(node, registers: set[str], params: list[str], vararg: bool):
    if isinstance(node, list):
        return [_rewrite(item, registers, params, vararg) for item in node]
    if not dataclasses.is_dataclass(node) or isinstance(node, type):
        return node
    rebuilt = type(node)(**{
        f.name: _rewrite(getattr(node, f.name), registers, params, vararg)
        for f in dataclasses.fields(node)
    })
    if isinstance(rebuilt, Index):
        obj, key = rebuilt.obj, rebuilt.key
        if isinstance(obj, Name) and obj.name == ENV:
            if isinstance(key, String) and key.value.isidentifier() and key.value not in _KEYWORDS:
                if key.value not in registers and key.value not in {"T", "ENV", "unpack", "select"} and not key.value.startswith(("arg", "upvalue", "cell", "_")):
                    return Name(key.value)
            return Index(Name("ENV"), key)
        if (
            isinstance(obj, Name) and obj.name == CELLS
            and isinstance(key, Index) and isinstance(key.obj, Name) and key.obj.name == SLOTS
            and isinstance(key.key, Number)
        ):
            return Name(f"upvalue{key.key.text}")
        if (
            isinstance(obj, Name) and obj.name == ARGS and not vararg
            and isinstance(key, Number) and key.text.isdigit() and 1 <= int(key.text) <= len(params)
        ):
            return Name(f"arg{key.text}")
    if isinstance(rebuilt, Name):
        if rebuilt.name == SELECT:
            return Name("select")
        if rebuilt.name == UNPACK:
            return Name("unpack")
        if rebuilt.name == ARGS:
            return Name("T")
    return rebuilt


def _used_names(stats: list[Stat]) -> set[str]:
    return {item.name for item in _walk(stats) if isinstance(item, Name)}


def _assigned(stats: list[Stat]) -> set[str]:
    names: set[str] = set()
    for item in _walk(stats):
        if isinstance(item, Assign):
            names.update(t.name for t in item.targets if isinstance(t, Name))
        elif isinstance(item, LocalAssign):
            names.update(item.names)
    return names


@dataclass
class StructuredFunction:
    function: VmFunction
    header: str
    stats: list[Stat]
    structured: bool
    locals: list[str]
    prologue: list[Stat]


def structure_function(cfg: Cfg, function: VmFunction) -> StructuredFunction | None:
    if function.entry_node is None or not function.nodes:
        return None
    prepared = prepare_nodes(cfg, function)
    structured = True
    try:
        stats = _Structurer(cfg, function, prepared).build()
        stats = cleanup(simplify_loops(stats))
    except (Unstructured, RecursionError):
        structured = False
        stats = state_machine(cfg, function, prepared)

    registers = set(cfg.registers) | {cfg.dispatcher.state_var}
    params, vararg = cfg.factory_params.get(function.factory or "", ([], True))
    arg_names = [f"arg{i + 1}" for i in range(len(params))]
    stats = _rewrite(stats, registers, arg_names, vararg)

    used = _used_names(stats)
    prologue: list[Stat] = []
    if vararg:
        prologue.append(LocalAssign(["T"], [_table_of([_vararg()])]))
    elif "T" in used:
        prologue.append(LocalAssign(["T"], [_table_of([Name(n) for n in arg_names])]))
    flags = {name for name in used if name.startswith("__brk")}
    declared = sorted(
        ((registers | flags) & (used | _assigned(stats))) - {"T"} - set(arg_names),
        key=lambda name: (len(name), name),
    )
    header = "..." if vararg else ", ".join(arg_names)
    return StructuredFunction(function, header, stats, structured, declared, prologue)


def _table_of(values: list[Expr]):
    from .ast_nodes import Table, TableField

    return Table([TableField(None, v) for v in values])


def _vararg():
    from .ast_nodes import Vararg

    return Vararg()
