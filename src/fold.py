from __future__ import annotations

from .ast_nodes import (
    BinOp, Expr, FalseLit, Nil, Number, Paren, String, TrueLit, UnOp,
)


def _parse_number(text: str) -> int | float:
    if text.lower().startswith("0x"):
        return int(text, 16)
    if "." in text or "e" in text.lower():
        return float(text)
    return int(text)


def _format_number(value: int | float) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return repr(int(value))
        return repr(value)
    return str(value)


class ConstFoldError(Exception):
    pass


def try_eval_const(expr: Expr) -> int | float | str | bool | None:
    if isinstance(expr, Number):
        return _parse_number(expr.text)
    if isinstance(expr, String):
        return expr.value
    if isinstance(expr, TrueLit):
        return True
    if isinstance(expr, FalseLit):
        return False
    if isinstance(expr, Nil):
        raise ConstFoldError("nil is not a fold target")
    if isinstance(expr, Paren):
        return try_eval_const(expr.inner)
    if isinstance(expr, UnOp):
        value = try_eval_const(expr.operand)
        if expr.op == "-":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return -value
            raise ConstFoldError("unary - on non-number")
        if expr.op == "not":
            return not value
        raise ConstFoldError(f"unsupported unary op {expr.op}")
    if isinstance(expr, BinOp):
        left = try_eval_const(expr.left)
        right = try_eval_const(expr.right)
        op = expr.op
        if op == "..":
            if isinstance(left, bool) or isinstance(right, bool):
                raise ConstFoldError("concat of bool")
            if isinstance(left, (int, float)):
                left = _format_number(left)
            if isinstance(right, (int, float)):
                right = _format_number(right)
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            raise ConstFoldError("concat of non-constant")
        if not isinstance(left, (int, float)) or isinstance(left, bool):
            raise ConstFoldError("left operand not numeric")
        if not isinstance(right, (int, float)) or isinstance(right, bool):
            raise ConstFoldError("right operand not numeric")
        try:
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            if op == "//":
                return left // right
            if op == "%":
                return left % right
            if op == "^":
                return float(left) ** float(right)
            if op == "==":
                return left == right
            if op == "~=":
                return left != right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
        except (ZeroDivisionError, OverflowError, ValueError) as exc:
            raise ConstFoldError(str(exc)) from exc
    raise ConstFoldError(f"unsupported node {type(expr).__name__}")


def fold_expr(expr: Expr) -> Expr:
    if isinstance(expr, BinOp):
        left = fold_expr(expr.left)
        right = fold_expr(expr.right)
        folded = BinOp(expr.op, left, right)
        try:
            value = try_eval_const(folded)
        except ConstFoldError:
            return folded
        return _literal_for(value)
    if isinstance(expr, UnOp):
        operand = fold_expr(expr.operand)
        folded = UnOp(expr.op, operand)
        try:
            value = try_eval_const(folded)
        except ConstFoldError:
            return folded
        return _literal_for(value)
    if isinstance(expr, Paren):
        inner = fold_expr(expr.inner)
        try:
            value = try_eval_const(inner)
            return _literal_for(value)
        except ConstFoldError:
            return Paren(inner)
    return expr


def _literal_for(value) -> Expr:
    if isinstance(value, bool):
        return TrueLit() if value else FalseLit()
    if isinstance(value, (int, float)):
        return Number(_format_number(value))
    if isinstance(value, str):
        return String(value)
    raise ConstFoldError(f"cannot build literal for {value!r}")
