from __future__ import annotations

from typing import Callable

from .ast_nodes import (
    Assign, BinOp, Block, Call, CallStat, Do, Expr, Field, FunctionDecl,
    FunctionExpr, GenericFor, If, Index, LocalAssign, MethodCall, NumericFor,
    Paren, Repeat, Return, Stat, Table, UnOp, While,
)


def walk_block(block: Block, on_stat: Callable[[Stat], None]) -> None:
    for stat in block.stats:
        on_stat(stat)
        walk_stat_children(stat, on_stat)


def walk_stat_children(stat: Stat, on_stat: Callable[[Stat], None]) -> None:
    if isinstance(stat, Do):
        walk_block(stat.body, on_stat)
    elif isinstance(stat, While):
        walk_expr_for_functions(stat.cond, on_stat)
        walk_block(stat.body, on_stat)
    elif isinstance(stat, Repeat):
        walk_block(stat.body, on_stat)
        walk_expr_for_functions(stat.cond, on_stat)
    elif isinstance(stat, If):
        for clause in stat.clauses:
            walk_expr_for_functions(clause.cond, on_stat)
            walk_block(clause.body, on_stat)
        if stat.orelse is not None:
            walk_block(stat.orelse, on_stat)
    elif isinstance(stat, NumericFor):
        walk_block(stat.body, on_stat)
    elif isinstance(stat, GenericFor):
        walk_block(stat.body, on_stat)
    elif isinstance(stat, FunctionDecl):
        walk_block(stat.body, on_stat)
    elif isinstance(stat, LocalAssign):
        for expr in stat.exprs:
            walk_expr_for_functions(expr, on_stat)
    elif isinstance(stat, Assign):
        for expr in stat.exprs:
            walk_expr_for_functions(expr, on_stat)
    elif isinstance(stat, CallStat):
        walk_expr_for_functions(stat.call, on_stat)
    elif isinstance(stat, Return):
        for expr in stat.exprs:
            walk_expr_for_functions(expr, on_stat)


def walk_expr_for_functions(expr: Expr, on_stat: Callable[[Stat], None]) -> None:
    if isinstance(expr, FunctionExpr):
        walk_block(expr.body, on_stat)
    elif isinstance(expr, Call):
        walk_expr_for_functions(expr.func, on_stat)
        for arg in expr.args:
            walk_expr_for_functions(arg, on_stat)
    elif isinstance(expr, MethodCall):
        walk_expr_for_functions(expr.obj, on_stat)
        for arg in expr.args:
            walk_expr_for_functions(arg, on_stat)
    elif isinstance(expr, Paren):
        walk_expr_for_functions(expr.inner, on_stat)
    elif isinstance(expr, Index):
        walk_expr_for_functions(expr.obj, on_stat)
        walk_expr_for_functions(expr.key, on_stat)
    elif isinstance(expr, Field):
        walk_expr_for_functions(expr.obj, on_stat)
    elif isinstance(expr, BinOp):
        walk_expr_for_functions(expr.left, on_stat)
        walk_expr_for_functions(expr.right, on_stat)
    elif isinstance(expr, UnOp):
        walk_expr_for_functions(expr.operand, on_stat)
    elif isinstance(expr, Table):
        for field in expr.fields:
            if field.key is not None:
                walk_expr_for_functions(field.key, on_stat)
            walk_expr_for_functions(field.value, on_stat)


def find_while_loops(block: Block) -> list[While]:
    found: list[While] = []

    def on_stat(stat: Stat) -> None:
        if isinstance(stat, While):
            found.append(stat)

    walk_block(block, on_stat)
    return found
