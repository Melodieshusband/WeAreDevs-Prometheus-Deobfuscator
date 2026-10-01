from __future__ import annotations

from .ast_nodes import (
    Assign, BinOp, Block, Break, Call, CallStat, Do, Expr, Field, FunctionDecl,
    FunctionExpr, GenericFor, If, IfClause, Index, LocalAssign, MethodCall,
    NumericFor, Paren, Repeat, Return, Stat, Table, TableField, UnOp, While,
)
from .fold import fold_expr


def fold_block(block: Block) -> Block:
    return Block([fold_stat(stat) for stat in block.stats])


def fold_stat(stat: Stat) -> Stat:
    if isinstance(stat, LocalAssign):
        return LocalAssign(stat.names, [fold_expr_deep(e) for e in stat.exprs])
    if isinstance(stat, Assign):
        return Assign([fold_expr_deep(t) for t in stat.targets], [fold_expr_deep(e) for e in stat.exprs])
    if isinstance(stat, CallStat):
        return CallStat(fold_expr_deep(stat.call))
    if isinstance(stat, Do):
        return Do(fold_block(stat.body))
    if isinstance(stat, While):
        return While(fold_expr_deep(stat.cond), fold_block(stat.body))
    if isinstance(stat, Repeat):
        return Repeat(fold_block(stat.body), fold_expr_deep(stat.cond))
    if isinstance(stat, If):
        clauses = [IfClause(fold_expr_deep(c.cond), fold_block(c.body)) for c in stat.clauses]
        orelse = fold_block(stat.orelse) if stat.orelse is not None else None
        return If(clauses, orelse)
    if isinstance(stat, NumericFor):
        step = fold_expr_deep(stat.step) if stat.step is not None else None
        return NumericFor(stat.var, fold_expr_deep(stat.start), fold_expr_deep(stat.stop), step, fold_block(stat.body))
    if isinstance(stat, GenericFor):
        return GenericFor(stat.names, [fold_expr_deep(e) for e in stat.exprs], fold_block(stat.body))
    if isinstance(stat, FunctionDecl):
        return FunctionDecl(stat.target, stat.is_local, stat.params, stat.is_vararg, fold_block(stat.body))
    if isinstance(stat, Return):
        return Return([fold_expr_deep(e) for e in stat.exprs])
    if isinstance(stat, Break):
        return stat
    return stat


def fold_expr_deep(expr: Expr) -> Expr:
    if isinstance(expr, Call):
        expr = Call(fold_expr_deep(expr.func), [fold_expr_deep(a) for a in expr.args])
    elif isinstance(expr, MethodCall):
        expr = MethodCall(fold_expr_deep(expr.obj), expr.method, [fold_expr_deep(a) for a in expr.args])
    elif isinstance(expr, Index):
        expr = Index(fold_expr_deep(expr.obj), fold_expr_deep(expr.key))
    elif isinstance(expr, Field):
        expr = Field(fold_expr_deep(expr.obj), expr.name)
    elif isinstance(expr, Table):
        expr = Table([
            TableField(fold_expr_deep(f.key) if f.key is not None else None, fold_expr_deep(f.value))
            for f in expr.fields
        ])
    elif isinstance(expr, FunctionExpr):
        expr = FunctionExpr(expr.params, expr.is_vararg, fold_block(expr.body))
    elif isinstance(expr, Paren):
        expr = Paren(fold_expr_deep(expr.inner))
    elif isinstance(expr, UnOp):
        expr = UnOp(expr.op, fold_expr_deep(expr.operand))
    elif isinstance(expr, BinOp):
        expr = BinOp(expr.op, fold_expr_deep(expr.left), fold_expr_deep(expr.right))
    return fold_expr(expr)
