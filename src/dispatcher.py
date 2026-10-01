from __future__ import annotations

from dataclasses import dataclass, field

from .ast_nodes import Block, Expr, If, Name, Stat, While
from .visitor import find_while_loops


@dataclass
class LeafBranch:
    path: list[tuple[Expr, bool]]
    stats: list[Stat]
    index: int = 0


@dataclass
class Dispatcher:
    state_var: str
    while_stat: While
    leaves: list[LeafBranch] = field(default_factory=list)


def _tree_weight(stats: list[Stat]) -> int:
    weight = 0
    for stat in stats:
        if isinstance(stat, If):
            weight += 1
            for clause in stat.clauses:
                weight += _tree_weight(clause.body.stats)
            if stat.orelse:
                weight += _tree_weight(stat.orelse.stats)
    return weight


def _collect_leaves(stats: list[Stat], path: list[tuple[Expr, bool]] | None = None) -> list[LeafBranch]:
    path = path or []
    prologue: list[Stat] = []
    for position, stat in enumerate(stats):
        if isinstance(stat, If) and position == len(stats) - 1:
            leaves: list[LeafBranch] = []
            failed: list[tuple[Expr, bool]] = []
            for clause in stat.clauses:
                branch_path = path + failed + [(clause.cond, True)]
                for leaf in _collect_leaves(clause.body.stats, branch_path):
                    leaf.stats = prologue + leaf.stats
                    leaves.append(leaf)
                failed.append((clause.cond, False))
            if stat.orelse is not None:
                for leaf in _collect_leaves(stat.orelse.stats, path + failed):
                    leaf.stats = prologue + leaf.stats
                    leaves.append(leaf)
            return leaves
        prologue.append(stat)
    return [LeafBranch(path=path, stats=prologue)]


def find_dispatcher(block: Block) -> Dispatcher:
    candidates = []
    for loop in find_while_loops(block):
        if isinstance(loop.cond, Name):
            candidates.append((_tree_weight(loop.body.stats), loop.cond.name, loop))
    if not candidates:
        raise ValueError("no dispatcher-shaped while loop found")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, state_name, loop = candidates[0]
    dispatcher = Dispatcher(state_var=state_name, while_stat=loop)
    dispatcher.leaves = _collect_leaves(loop.body.stats)
    for index, leaf in enumerate(dispatcher.leaves):
        leaf.index = index
    return dispatcher
