from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Expr:
    pass


@dataclass
class Nil(Expr):
    pass


@dataclass
class TrueLit(Expr):
    pass


@dataclass
class FalseLit(Expr):
    pass


@dataclass
class Vararg(Expr):
    pass


@dataclass
class Number(Expr):
    text: str


@dataclass
class String(Expr):
    value: str


@dataclass
class Name(Expr):
    name: str


@dataclass
class Index(Expr):
    obj: Expr
    key: Expr


@dataclass
class Field(Expr):
    obj: Expr
    name: str


@dataclass
class Call(Expr):
    func: Expr
    args: list[Expr]


@dataclass
class MethodCall(Expr):
    obj: Expr
    method: str
    args: list[Expr]


@dataclass
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class UnOp(Expr):
    op: str
    operand: Expr


@dataclass
class TableField:
    key: Expr | None
    value: Expr


@dataclass
class Table(Expr):
    fields: list[TableField]


@dataclass
class FunctionExpr(Expr):
    params: list[str]
    is_vararg: bool
    body: "Block"


@dataclass
class Paren(Expr):
    inner: Expr


class Stat:
    pass


@dataclass
class Block:
    stats: list[Stat] = field(default_factory=list)


@dataclass
class LocalAssign(Stat):
    names: list[str]
    exprs: list[Expr]


@dataclass
class Assign(Stat):
    targets: list[Expr]
    exprs: list[Expr]


@dataclass
class CallStat(Stat):
    call: Expr


@dataclass
class Do(Stat):
    body: Block


@dataclass
class While(Stat):
    cond: Expr
    body: Block


@dataclass
class Repeat(Stat):
    body: Block
    cond: Expr


@dataclass
class IfClause:
    cond: Expr
    body: Block


@dataclass
class If(Stat):
    clauses: list[IfClause]
    orelse: Block | None


@dataclass
class NumericFor(Stat):
    var: str
    start: Expr
    stop: Expr
    step: Expr | None
    body: Block


@dataclass
class GenericFor(Stat):
    names: list[str]
    exprs: list[Expr]
    body: Block


@dataclass
class FunctionDecl(Stat):

    target: Expr
    is_local: bool
    params: list[str]
    is_vararg: bool
    body: Block


@dataclass
class Return(Stat):
    exprs: list[Expr]


@dataclass
class Break(Stat):
    pass


@dataclass
class Continue(Stat):
    pass


@dataclass
class Raw(Stat):

    text: str
