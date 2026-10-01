from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass, field

from .ast_nodes import Assign, Call, Expr, FunctionExpr, Index, LocalAssign, Name, Stat, String
from .dispatcher import Dispatcher, find_dispatcher
from .roles import ALLOC, ARGS, CELLS, ENV, RELEASE, RET, SELECT, SLOTS, UNPACK, Roles, find_roles, rename_names
from .strings_layer import _walk
from .transitions import EXIT, Jump, _as_int, extract_transition

EXIT_NODE = -1

_COMPARE = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "~=": lambda a, b: a != b,
}


@dataclass
class CfgNode:
    index: int
    body: list[Stat]
    jump: Jump | None
    successors: list[int] = field(default_factory=list)
    exits: bool = False
    unresolved: bool = False
    entry_states: set[int] = field(default_factory=set)
    term_kind: str = "exit"
    goto_node: int = EXIT_NODE
    cond_expr: Expr | None = None
    true_node: int = EXIT_NODE
    false_node: int = EXIT_NODE


@dataclass
class VmFunction:
    entry_state: int
    entry_node: int | None
    nodes: list[int] = field(default_factory=list)
    factory: str | None = None


@dataclass
class Cfg:
    dispatcher: Dispatcher
    nodes: list[CfgNode]
    functions: list[VmFunction]
    factories: list[str]
    dispatcher_name: str | None
    unmatched_states: set[int]
    factory_params: dict[str, tuple[list[str], bool]] = field(default_factory=dict)
    registers: list[str] = field(default_factory=list)
    roles: Roles | None = None


def _holds(cond: Expr, state_var: str, value: int) -> bool | None:
    from .ast_nodes import BinOp

    if not isinstance(cond, BinOp) or cond.op not in _COMPARE:
        return None
    left, right = cond.left, cond.right
    if isinstance(left, Name) and left.name == state_var:
        other = _as_int(right)
        return None if other is None else _COMPARE[cond.op](value, other)
    if isinstance(right, Name) and right.name == state_var:
        other = _as_int(left)
        return None if other is None else _COMPARE[cond.op](other, value)
    return None


def _leaf_for_state(dispatcher: Dispatcher, value: int) -> int | None:
    for leaf in dispatcher.leaves:
        if all(_holds(cond, dispatcher.state_var, value) is sense for cond, sense in leaf.path):
            return leaf.index
    return None


def _make_exit_check(root):
    counts: Counter[str] = Counter()
    for node in _walk(root):
        if isinstance(node, String):
            counts[node.value] += 1

    def check(expr: Expr) -> bool:
        return (
            isinstance(expr, Index)
            and isinstance(expr.obj, Name)
            and isinstance(expr.key, String)
            and counts[expr.key.value] == 1
        )

    return check


def _contains_call_to(node: object, name: str) -> bool:
    for item in _walk(node):
        if isinstance(item, Call) and isinstance(item.func, Name) and item.func.name == name:
            return True
    return False


def _inner_closure_params(body) -> tuple[list[str], bool] | None:
    for stat in body.stats:
        if isinstance(stat, LocalAssign) and len(stat.exprs) == 1 and isinstance(stat.exprs[0], FunctionExpr):
            return list(stat.exprs[0].params), stat.exprs[0].is_vararg
    return None


def _dispatcher_registers(function_expr: FunctionExpr) -> list[str]:
    for stat in function_expr.body.stats:
        if isinstance(stat, LocalAssign) and not stat.exprs:
            return list(stat.names)
    return []


def _find_dispatcher_info(root, dispatcher: Dispatcher):
    for node in _walk(root):
        if not isinstance(node, Assign):
            continue
        for position, expr in enumerate(node.exprs):
            if not isinstance(expr, FunctionExpr):
                continue
            if not any(item is dispatcher.while_stat for item in _walk(expr)):
                continue
            if position >= len(node.targets) or not isinstance(node.targets[position], Name):
                continue
            name = node.targets[position].name
            factories: dict[str, tuple[list[str], bool]] = {}
            for other_position, other in enumerate(node.exprs):
                if other_position == position or not isinstance(other, FunctionExpr):
                    continue
                if other_position >= len(node.targets) or not isinstance(node.targets[other_position], Name):
                    continue
                if len(other.params) == 2 and _contains_call_to(other.body, name):
                    params = _inner_closure_params(other.body)
                    factories[node.targets[other_position].name] = params if params is not None else ([], True)
            return name, factories, _dispatcher_registers(expr)
    return None, {}, []


def build_cfg(root) -> Cfg:
    dispatcher = find_dispatcher(root)
    exit_check = _make_exit_check(root)
    dispatcher_name, factory_params, registers = _find_dispatcher_info(root, dispatcher)
    roles = find_roles(root, dispatcher)
    if roles is not None and roles.alloc and roles.cells and roles.unpack and roles.ret_reg:
        mapping = roles.rename_map(set(registers))
    else:
        mapping = {
            "T": ARGS, "y": SLOTS, "x": ENV, "h": UNPACK, "E": SELECT,
            "a": CELLS, "I": ALLOC, "A": RELEASE, "j": RET,
        }
    registers = [mapping.get(name, name) for name in registers]
    nodes: list[CfgNode] = []
    for leaf in dispatcher.leaves:
        renamed = rename_names(leaf.stats, mapping)
        body, jump = extract_transition(renamed, dispatcher.state_var, exit_check)
        nodes.append(CfgNode(index=leaf.index, body=body, jump=jump))

    unmatched: set[int] = set()

    def resolve(state: int | None) -> int | None:
        if state is None or state == EXIT:
            return None
        found = _leaf_for_state(dispatcher, state)
        if found is None:
            unmatched.add(state)
        return found

    def node_of(state: int | None) -> int:
        found = resolve(state)
        return EXIT_NODE if found is None else found

    for node in nodes:
        jump = node.jump
        if jump is None or jump.kind == "exit":
            node.exits = True
            node.term_kind = "exit"
            continue
        if jump.kind == "const":
            node.term_kind = "goto"
            node.goto_node = node_of(jump.target)
            if node.goto_node != EXIT_NODE:
                node.successors.append(node.goto_node)
                nodes[node.goto_node].entry_states.add(jump.target)
            else:
                node.term_kind = "exit"
                node.exits = True
        elif jump.kind == "cond":
            node.term_kind = "cond"
            node.cond_expr = jump.cond
            node.true_node = node_of(jump.true_target)
            node.false_node = node_of(jump.false_target)
            for state, target in ((jump.true_target, node.true_node), (jump.false_target, node.false_node)):
                if target == EXIT_NODE:
                    node.exits = True
                    if state is not None and state != EXIT:
                        node.unresolved = True
                else:
                    node.successors.append(target)
                    nodes[target].entry_states.add(state)
        else:
            node.unresolved = True
            node.term_kind = "exit"

    factories = list(factory_params)
    entries: list[tuple[int, str]] = []
    seen_states: set[int] = set()
    for item in _walk(root):
        if isinstance(item, Call) and isinstance(item.func, Name) and item.func.name in factories:
            if item.args:
                state = _as_int(item.args[0])
                if state is not None and state not in seen_states:
                    seen_states.add(state)
                    entries.append((state, item.func.name))

    functions: list[VmFunction] = []
    for state, factory in entries:
        entry_node = _leaf_for_state(dispatcher, state)
        function = VmFunction(entry_state=state, entry_node=entry_node, factory=factory)
        if entry_node is not None:
            nodes[entry_node].entry_states.add(state)
            seen = {entry_node}
            stack = [entry_node]
            while stack:
                current = stack.pop()
                for successor in nodes[current].successors:
                    if successor not in seen:
                        seen.add(successor)
                        stack.append(successor)
            function.nodes = sorted(seen)
        functions.append(function)

    return Cfg(
        dispatcher, nodes, functions, factories, dispatcher_name, unmatched,
        factory_params, registers + ["__c", RET], roles,
    )
