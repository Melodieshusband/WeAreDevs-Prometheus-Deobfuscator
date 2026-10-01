from __future__ import annotations

from .ast_nodes import (
    Assign, BinOp, Block, Break, Call, Continue, CallStat, Do, Expr, FalseLit, Field,
    FunctionDecl, FunctionExpr, GenericFor, If, Index, LocalAssign,
    MethodCall, Name, Nil, Number, NumericFor, Paren, Repeat, Return, Stat,
    String, Table, TrueLit, UnOp, Vararg, While,
)

INDENT = "    "


def _needs_quote_free(name: str) -> bool:
    return name.isidentifier()


def _lua_string_literal(value: str) -> str:
    out = ['"']
    for ch in value:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif 32 <= code <= 126:
            out.append(ch)
        else:
            out.append(f"\\{code}")
    out.append('"')
    return "".join(out)


class Printer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.depth = 0

    def _prefix(self, expr) -> str:
        text = self.render_expr(expr)
        if isinstance(expr, (Name, Index, Field, Call, MethodCall, Paren)):
            return text
        return f"({text})"

    def render_block(self, block: Block, indent: int) -> str:
        lines = []
        for stat in block.stats:
            lines.append(self.render_stat(stat, indent))
        return "\n".join(lines)

    def render_stat(self, stat: Stat, indent: int) -> str:
        saved = self.depth
        self.depth = indent
        try:
            return self._render_stat(stat, indent)
        finally:
            self.depth = saved

    def _render_stat(self, stat: Stat, indent: int) -> str:
        pad = INDENT * indent
        if isinstance(stat, LocalAssign):
            names = ", ".join(stat.names)
            if not stat.exprs:
                return f"{pad}local {names}"
            exprs = ", ".join(self.render_expr(e) for e in stat.exprs)
            return f"{pad}local {names} = {exprs}"
        if isinstance(stat, Assign):
            targets = ", ".join(self.render_expr(t) for t in stat.targets)
            exprs = ", ".join(self.render_expr(e) for e in stat.exprs)
            return f"{pad}{targets} = {exprs}"
        if isinstance(stat, CallStat):
            return f"{pad}{self.render_expr(stat.call)}"
        if isinstance(stat, Do):
            inner = self.render_block(stat.body, indent + 1)
            body = f"\n{inner}\n" if inner else "\n"
            return f"{pad}do{body}{pad}end"
        if isinstance(stat, While):
            inner = self.render_block(stat.body, indent + 1)
            body = f"\n{inner}\n" if inner else "\n"
            return f"{pad}while {self.render_expr(stat.cond)} do{body}{pad}end"
        if isinstance(stat, Repeat):
            inner = self.render_block(stat.body, indent + 1)
            body = f"\n{inner}\n" if inner else "\n"
            return f"{pad}repeat{body}{pad}until {self.render_expr(stat.cond)}"
        if isinstance(stat, If):
            parts = []
            for index, clause in enumerate(stat.clauses):
                keyword = "if" if index == 0 else "elseif"
                inner = self.render_block(clause.body, indent + 1)
                body = f"\n{inner}\n" if inner else "\n"
                parts.append(f"{pad}{keyword} {self.render_expr(clause.cond)} then{body}")
            text = "".join(parts)
            if stat.orelse is not None:
                inner = self.render_block(stat.orelse, indent + 1)
                body = f"\n{inner}\n" if inner else "\n"
                text += f"{pad}else{body}"
            text += f"{pad}end"
            return text
        if isinstance(stat, NumericFor):
            step = f", {self.render_expr(stat.step)}" if stat.step is not None else ""
            inner = self.render_block(stat.body, indent + 1)
            body = f"\n{inner}\n" if inner else "\n"
            header = f"{pad}for {stat.var} = {self.render_expr(stat.start)}, {self.render_expr(stat.stop)}{step} do"
            return f"{header}{body}{pad}end"
        if isinstance(stat, GenericFor):
            names = ", ".join(stat.names)
            exprs = ", ".join(self.render_expr(e) for e in stat.exprs)
            inner = self.render_block(stat.body, indent + 1)
            body = f"\n{inner}\n" if inner else "\n"
            return f"{pad}for {names} in {exprs} do{body}{pad}end"
        if isinstance(stat, FunctionDecl):
            prefix = "local function" if stat.is_local else "function"
            name = self.render_expr(stat.target)
            params = list(stat.params)
            if not stat.is_local and params and params[0] == "self":
                params = params[1:]
            param_list = ", ".join(params) + (", ..." if stat.is_vararg and params else ("..." if stat.is_vararg else ""))
            inner = self.render_block(stat.body, indent + 1)
            body = f"\n{inner}\n" if inner else "\n"
            return f"{pad}{prefix} {name}({param_list}){body}{pad}end"
        if isinstance(stat, Return):
            if not stat.exprs:
                return f"{pad}return"
            exprs = ", ".join(self.render_expr(e) for e in stat.exprs)
            return f"{pad}return {exprs}"
        if isinstance(stat, Break):
            return f"{pad}break"
        if isinstance(stat, Continue):
            return f"{pad}continue"
        return f"{pad}-- <unhandled statement: {type(stat).__name__}>"


    def render_expr(self, expr: Expr) -> str:
        if isinstance(expr, Nil):
            return "nil"
        if isinstance(expr, TrueLit):
            return "true"
        if isinstance(expr, FalseLit):
            return "false"
        if isinstance(expr, Vararg):
            return "..."
        if isinstance(expr, Number):
            return expr.text
        if isinstance(expr, String):
            return _lua_string_literal(expr.value)
        if isinstance(expr, Name):
            return expr.name
        if isinstance(expr, Index):
            return f"{self._prefix(expr.obj)}[{self.render_expr(expr.key)}]"
        if isinstance(expr, Field):
            return f"{self._prefix(expr.obj)}.{expr.name}"
        if isinstance(expr, Call):
            args = ", ".join(self.render_expr(a) for a in expr.args)
            return f"{self._prefix(expr.func)}({args})"
        if isinstance(expr, MethodCall):
            args = ", ".join(self.render_expr(a) for a in expr.args)
            return f"{self._prefix(expr.obj)}:{expr.method}({args})"
        if isinstance(expr, BinOp):
            return f"{self.render_expr(expr.left)} {expr.op} {self.render_expr(expr.right)}"
        if isinstance(expr, UnOp):
            if expr.op == "not":
                return f"not {self.render_expr(expr.operand)}"
            operand = self.render_expr(expr.operand)
            if expr.op == "-" and operand.startswith("-"):
                return f"- {operand}"
            return f"{expr.op}{operand}"
        if isinstance(expr, Paren):
            return f"({self.render_expr(expr.inner)})"
        if isinstance(expr, Table):
            if not expr.fields:
                return "{}"
            parts = []
            for field in expr.fields:
                if field.key is None:
                    parts.append(self.render_expr(field.value))
                elif isinstance(field.key, String) and _needs_quote_free(field.key.value):
                    parts.append(f"{field.key.value} = {self.render_expr(field.value)}")
                else:
                    parts.append(f"[{self.render_expr(field.key)}] = {self.render_expr(field.value)}")
            return "{ " + ", ".join(parts) + " }"
        if isinstance(expr, FunctionExpr):
            params = ", ".join(expr.params) + ("..." if expr.is_vararg else "")
            inner = self.render_block(expr.body, self.depth + 1)
            body = f"\n{inner}\n" if inner else "\n"
            return f"function({params}){body}{INDENT * self.depth}end"
        return f"--[[unhandled expr: {type(expr).__name__}]]"


def render(block: Block) -> str:
    printer = Printer()
    return printer.render_block(block, 0)


def render_expr(expr: Expr) -> str:
    return Printer().render_expr(expr)
