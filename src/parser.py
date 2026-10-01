from __future__ import annotations

from .ast_nodes import (
    Assign, BinOp, Block, Break, Call, CallStat, Field, FunctionDecl,
    FunctionExpr, GenericFor, If, IfClause, Index, LocalAssign, MethodCall,
    Name, Nil, Number, NumericFor, Paren, Repeat, Return, Stat, String,
    Table, TableField, TrueLit, FalseLit, UnOp, Vararg, While,
)
from .lexer import Token, tokenize


class LuaParseError(ValueError):
    def __init__(self, message: str, token: Token):
        super().__init__(f"{message} near {token.kind}:{token.value!r} (line {token.line})")
        self.token = token


_BINOP_PRECEDENCE = {
    "or": 1,
    "and": 2,
    "<": 3, ">": 3, "<=": 3, ">=": 3, "~=": 3, "==": 3,
    "|": 4,
    "~": 5,
    "&": 6,
    "<<": 7, ">>": 7,
    "..": 9,
    "+": 10, "-": 10,
    "*": 11, "/": 11, "//": 11, "%": 11,
    "^": 14,
}
_RIGHT_ASSOC = {"..", "^"}
_UNARY_PRECEDENCE = 12


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0


    def peek(self, offset: int = 0) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.kind != "eof":
            self.pos += 1
        return tok

    def check(self, kind: str, value: str | None = None) -> bool:
        tok = self.peek()
        if tok.kind != kind:
            return False
        return value is None or tok.value == value

    def check_kw(self, *words: str) -> bool:
        tok = self.peek()
        return tok.kind == "keyword" and tok.value in words

    def check_sym(self, *symbols: str) -> bool:
        tok = self.peek()
        return tok.kind == "symbol" and tok.value in symbols

    def expect(self, kind: str, value: str | None = None) -> Token:
        if not self.check(kind, value):
            expected = value if value is not None else kind
            raise LuaParseError(f"Expected {expected!r}", self.peek())
        return self.advance()

    def expect_sym(self, symbol: str) -> Token:
        return self.expect("symbol", symbol)

    def expect_kw(self, word: str) -> Token:
        return self.expect("keyword", word)


    def parse_chunk(self) -> Block:
        block = self.parse_block()
        if not self.check("eof"):
            raise LuaParseError("Unexpected trailing tokens", self.peek())
        return block

    _BLOCK_END = {"end", "else", "elseif", "until"}

    def parse_block(self) -> Block:
        stats: list[Stat] = []
        while True:
            tok = self.peek()
            if tok.kind == "eof":
                break
            if tok.kind == "keyword" and tok.value in self._BLOCK_END:
                break
            if self.check_sym(";"):
                self.advance()
                continue
            if self.check_kw("return"):
                stats.append(self._parse_return())
                break
            stats.append(self._parse_statement())
        return Block(stats)


    def _parse_statement(self) -> Stat:
        if self.check_kw("local"):
            return self._parse_local()
        if self.check_kw("if"):
            return self._parse_if()
        if self.check_kw("while"):
            return self._parse_while()
        if self.check_kw("repeat"):
            return self._parse_repeat()
        if self.check_kw("for"):
            return self._parse_for()
        if self.check_kw("function"):
            return self._parse_function_decl()
        if self.check_kw("do"):
            self.advance()
            body = self.parse_block()
            self.expect_kw("end")
            from .ast_nodes import Do
            return Do(body)
        if self.check_kw("break"):
            self.advance()
            return Break()
        return self._parse_expr_statement()

    def _parse_local(self) -> Stat:
        self.advance()
        if self.check_kw("function"):
            self.advance()
            name = self.expect("name").value
            params, is_vararg, body = self._parse_function_body()
            return FunctionDecl(Name(name), True, params, is_vararg, body)
        names = [self.expect("name").value]
        self._skip_type_annotation()
        while self.check_sym(","):
            self.advance()
            names.append(self.expect("name").value)
            self._skip_type_annotation()
        exprs: list = []
        if self.check_sym("="):
            self.advance()
            exprs = self._parse_expr_list()
        return LocalAssign(names, exprs)

    def _skip_type_annotation(self) -> None:
        if self.check_sym(":"):
            self.advance()
            depth = 0
            while True:
                tok = self.peek()
                if tok.kind == "eof":
                    return
                if depth == 0 and (self.check_sym(",", "=") or self.check_kw("end") or self.check_sym(")")):
                    return
                if tok.kind == "symbol" and tok.value in "([{":
                    depth += 1
                elif tok.kind == "symbol" and tok.value in ")]}":
                    if depth == 0:
                        return
                    depth -= 1
                elif tok.kind == "symbol" and tok.value == "<":
                    depth += 1
                elif tok.kind == "symbol" and tok.value == ">":
                    depth = max(0, depth - 1)
                self.advance()

    def _parse_if(self) -> Stat:
        self.advance()
        clauses = []
        cond = self._parse_expr()
        self.expect_kw("then")
        body = self.parse_block()
        clauses.append(IfClause(cond, body))
        while self.check_kw("elseif"):
            self.advance()
            cond = self._parse_expr()
            self.expect_kw("then")
            body = self.parse_block()
            clauses.append(IfClause(cond, body))
        orelse = None
        if self.check_kw("else"):
            self.advance()
            orelse = self.parse_block()
        self.expect_kw("end")
        return If(clauses, orelse)

    def _parse_while(self) -> Stat:
        self.advance()
        cond = self._parse_expr()
        self.expect_kw("do")
        body = self.parse_block()
        self.expect_kw("end")
        return While(cond, body)

    def _parse_repeat(self) -> Stat:
        self.advance()
        body = self.parse_block()
        self.expect_kw("until")
        cond = self._parse_expr()
        return Repeat(body, cond)

    def _parse_for(self) -> Stat:
        self.advance()
        first_name = self.expect("name").value
        if self.check_sym("="):
            self.advance()
            start = self._parse_expr()
            self.expect_sym(",")
            stop = self._parse_expr()
            step = None
            if self.check_sym(","):
                self.advance()
                step = self._parse_expr()
            self.expect_kw("do")
            body = self.parse_block()
            self.expect_kw("end")
            return NumericFor(first_name, start, stop, step, body)
        names = [first_name]
        while self.check_sym(","):
            self.advance()
            names.append(self.expect("name").value)
        self.expect_kw("in")
        exprs = self._parse_expr_list()
        self.expect_kw("do")
        body = self.parse_block()
        self.expect_kw("end")
        return GenericFor(names, exprs, body)

    def _parse_function_decl(self) -> Stat:
        self.advance()
        target: object = Name(self.expect("name").value)
        is_method = False
        while self.check_sym(".", ":"):
            sep = self.advance().value
            field_name = self.expect("name").value
            target = Field(target, field_name)
            if sep == ":":
                is_method = True
                break
        params, is_vararg, body = self._parse_function_body()
        if is_method:
            params = ["self"] + params
        return FunctionDecl(target, False, params, is_vararg, body)

    def _parse_function_body(self) -> tuple[list[str], bool, "object"]:
        self.expect_sym("(")
        params: list[str] = []
        is_vararg = False
        if not self.check_sym(")"):
            while True:
                if self.check_sym("..."):
                    self.advance()
                    is_vararg = True
                    break
                params.append(self.expect("name").value)
                self._skip_type_annotation()
                if self.check_sym(","):
                    self.advance()
                    continue
                break
        self.expect_sym(")")
        self._skip_return_type_annotation()
        body = self.parse_block()
        self.expect_kw("end")
        return params, is_vararg, body

    def _skip_return_type_annotation(self) -> None:
        if self.check_sym(":"):
            self.advance()
            depth = 0
            while True:
                tok = self.peek()
                if tok.kind == "eof":
                    return
                if depth == 0 and (self.check_kw("do") or (tok.kind == "keyword" and tok.value not in ("nil", "true", "false"))):
                    return
                if tok.kind == "symbol" and tok.value in "([{<":
                    depth += 1
                elif tok.kind == "symbol" and tok.value in ")]}>":
                    depth = max(0, depth - 1)
                self.advance()

    def _parse_return(self) -> Stat:
        self.advance()
        exprs: list = []
        if not (self.check("eof") or (self.peek().kind == "keyword" and self.peek().value in self._BLOCK_END) or self.check_sym(";")):
            exprs = self._parse_expr_list()
        if self.check_sym(";"):
            self.advance()
        return Return(exprs)

    def _parse_expr_statement(self) -> Stat:
        expr = self._parse_suffixed_expr()
        if self.check_sym("=", ","):
            targets = [expr]
            while self.check_sym(","):
                self.advance()
                targets.append(self._parse_suffixed_expr())
            self.expect_sym("=")
            exprs = self._parse_expr_list()
            return Assign(targets, exprs)
        if isinstance(expr, (Call, MethodCall)):
            return CallStat(expr)
        raise LuaParseError("Expected assignment or call statement", self.peek())


    def _parse_expr_list(self) -> list:
        exprs = [self._parse_expr()]
        while self.check_sym(","):
            self.advance()
            exprs.append(self._parse_expr())
        return exprs

    def _parse_expr(self, min_prec: int = 0) -> object:
        left = self._parse_unary()
        while True:
            tok = self.peek()
            op = None
            if tok.kind == "symbol" and tok.value in _BINOP_PRECEDENCE:
                op = tok.value
            elif tok.kind == "keyword" and tok.value in ("and", "or"):
                op = tok.value
            if op is None:
                break
            prec = _BINOP_PRECEDENCE[op]
            if prec < min_prec:
                break
            self.advance()
            next_min = prec if op in _RIGHT_ASSOC else prec + 1
            right = self._parse_expr(next_min)
            left = BinOp(op, left, right)
        return left

    def _parse_unary(self) -> object:
        tok = self.peek()
        if (tok.kind == "symbol" and tok.value in ("-", "#", "~")) or (tok.kind == "keyword" and tok.value == "not"):
            self.advance()
            operand = self._parse_expr(_UNARY_PRECEDENCE)
            return UnOp(tok.value, operand)
        return self._parse_pow()

    def _parse_pow(self) -> object:
        return self._parse_suffixed_expr()

    def _parse_suffixed_expr(self) -> object:
        expr = self._parse_primary_expr()
        while True:
            if self.check_sym("."):
                self.advance()
                name = self.expect("name").value
                expr = Field(expr, name)
                continue
            if self.check_sym("["):
                self.advance()
                key = self._parse_expr()
                self.expect_sym("]")
                expr = Index(expr, key)
                continue
            if self.check_sym(":"):
                self.advance()
                method = self.expect("name").value
                args = self._parse_call_args()
                expr = MethodCall(expr, method, args)
                continue
            if self.check_sym("(") or self.check("string") or self.check_sym("{"):
                args = self._parse_call_args()
                expr = Call(expr, args)
                continue
            break
        return expr

    def _parse_call_args(self) -> list:
        if self.check("string"):
            tok = self.advance()
            return [String(tok.value)]
        if self.check_sym("{"):
            return [self._parse_table()]
        self.expect_sym("(")
        args: list = []
        if not self.check_sym(")"):
            args = self._parse_expr_list()
        self.expect_sym(")")
        return args

    def _parse_primary_expr(self) -> object:
        tok = self.peek()
        if tok.kind == "number":
            self.advance()
            return Number(tok.value)
        if tok.kind == "string":
            self.advance()
            return String(tok.value)
        if tok.kind == "keyword" and tok.value == "nil":
            self.advance()
            return Nil()
        if tok.kind == "keyword" and tok.value == "true":
            self.advance()
            return TrueLit()
        if tok.kind == "keyword" and tok.value == "false":
            self.advance()
            return FalseLit()
        if tok.kind == "symbol" and tok.value == "...":
            self.advance()
            return Vararg()
        if tok.kind == "keyword" and tok.value == "function":
            self.advance()
            params, is_vararg, body = self._parse_function_body()
            return FunctionExpr(params, is_vararg, body)
        if tok.kind == "name":
            self.advance()
            return Name(tok.value)
        if tok.kind == "symbol" and tok.value == "(":
            self.advance()
            inner = self._parse_expr()
            self.expect_sym(")")
            return Paren(inner)
        if tok.kind == "symbol" and tok.value == "{":
            return self._parse_table()
        raise LuaParseError("Unexpected token in expression", tok)

    def _parse_table(self) -> object:
        self.expect_sym("{")
        fields: list[TableField] = []
        while not self.check_sym("}"):
            if self.check_sym("["):
                self.advance()
                key = self._parse_expr()
                self.expect_sym("]")
                self.expect_sym("=")
                value = self._parse_expr()
                fields.append(TableField(key, value))
            elif self.check("name") and self.peek(1).kind == "symbol" and self.peek(1).value == "=":
                name_tok = self.advance()
                self.advance()
                value = self._parse_expr()
                fields.append(TableField(String(name_tok.value), value))
            else:
                value = self._parse_expr()
                fields.append(TableField(None, value))
            if self.check_sym(",", ";"):
                self.advance()
                continue
            break
        self.expect_sym("}")
        return Table(fields)


def parse_lua(source: str) -> Block:
    tokens = tokenize(source)
    return Parser(tokens).parse_chunk()


def parse_expression(source: str) -> object:
    tokens = tokenize(source)
    parser = Parser(tokens)
    expr = parser._parse_expr()
    if not parser.check("eof"):
        raise LuaParseError("Unexpected trailing tokens in expression", parser.peek())
    return expr
