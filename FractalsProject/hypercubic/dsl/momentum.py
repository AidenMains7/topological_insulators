import re
import numpy as np

from .expr import safe_eval, safe_format


_TOKEN = re.compile(r"\s*(?:(?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?j?)"
                    r"|(?P<func>cos|sin|exp)"
                    r"|(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
                    r"|(?P<op>\*\*|[+\-*/(),]))")


def parse_momentum(expr, dim_symbols, params=None):
    expr = safe_format(expr, params or {})
    tokens = _tokenize(expr)
    parser = _Parser(tokens, dim_symbols, params or {})
    result = parser.parse_expr()
    parser.expect_end()
    nd = len(dim_symbols)
    out = {}
    for shift, coef in result.items():
        if coef == 0:
            continue
        if shift not in out:
            out[shift] = complex(coef)
        else:
            out[shift] += complex(coef)
            if out[shift] == 0:
                del out[shift]
    if not out:
        return {tuple([0] * nd): 0.0} if False else {}
    return out


def _tokenize(s):
    tokens = []
    i = 0
    while i < len(s):
        m = _TOKEN.match(s, i)
        if not m:
            if s[i].isspace():
                i += 1
                continue
            raise SyntaxError(f"unexpected character {s[i]!r} at position {i}")
        i = m.end()
        if m.group("num"):
            v = m.group("num")
            tokens.append(("num", complex(v) if v.endswith("j") else float(v)))
        elif m.group("func"):
            tokens.append(("func", m.group("func")))
        elif m.group("name"):
            tokens.append(("name", m.group("name")))
        elif m.group("op"):
            tokens.append(("op", m.group("op")))
    tokens.append(("end", None))
    return tokens


class _Parser:
    def __init__(self, tokens, dim_symbols, params):
        self.tokens = tokens
        self.pos = 0
        self.dim_symbols = list(dim_symbols)
        self.k_index = {sym: i for i, sym in enumerate(dim_symbols)}
        self.nd = len(dim_symbols)
        self.params = params

    def peek(self):
        return self.tokens[self.pos]

    def consume(self):
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect_end(self):
        if self.peek()[0] != "end":
            raise SyntaxError(f"unexpected token {self.peek()}")

    # grammar: expr := term (('+'|'-') term)*
    def parse_expr(self):
        left = self.parse_term()
        while True:
            t, v = self.peek()
            if t == "op" and v in ("+", "-"):
                self.consume()
                right = self.parse_term()
                if v == "+":
                    left = _add(left, right)
                else:
                    left = _add(left, _scale(right, -1))
            else:
                return left

    # term := factor (('*'|'/') factor)*
    def parse_term(self):
        left = self.parse_factor()
        while True:
            t, v = self.peek()
            if t == "op" and v in ("*", "/"):
                self.consume()
                right = self.parse_factor()
                if v == "*":
                    left = _mul(left, right)
                else:
                    left = _div(left, right)
            else:
                return left

    # factor := unary ('**' factor)?
    def parse_factor(self):
        base = self.parse_unary()
        t, v = self.peek()
        if t == "op" and v == "**":
            self.consume()
            exp = self.parse_factor()
            return _pow(base, exp)
        return base

    # unary := ('+'|'-') unary | atom
    def parse_unary(self):
        t, v = self.peek()
        if t == "op" and v in ("+", "-"):
            self.consume()
            inner = self.parse_unary()
            return _scale(inner, -1) if v == "-" else inner
        return self.parse_atom()

    # atom := num | func '(' expr ')' | name | '(' expr ')'
    def parse_atom(self):
        t, v = self.consume()
        if t == "num":
            return _scalar(v, self.nd)
        if t == "func":
            self._expect_op("(")
            inner = self.parse_expr()
            self._expect_op(")")
            return _apply_func(v, inner, self.nd)
        if t == "name":
            if v in self.k_index:
                # bare k symbol -> momentum coordinate; only valid inside cos/sin
                # represent as a special marker that downstream cos/sin/exp can interpret
                return _k_marker(self.k_index[v], self.nd)
            if v in self.params:
                return _scalar(self.params[v], self.nd)
            raise NameError(f"unknown name '{v}' in momentum expression")
        if t == "op" and v == "(":
            inner = self.parse_expr()
            self._expect_op(")")
            return inner
        raise SyntaxError(f"unexpected token ({t}, {v})")

    def _expect_op(self, sym):
        t, v = self.consume()
        if t != "op" or v != sym:
            raise SyntaxError(f"expected '{sym}', got ({t}, {v})")


# ---------------------------------------------------------------- shift algebra
# Internal representation:
#   dict mapping shift_tuple (tuple[int]) -> complex coefficient
# Plus a special "k-marker" representation: a dict with key "_k" storing (dim_idx, multiplier)
# k-markers are linear combinations of momentum coordinates that have NO Fourier expansion
# until wrapped by cos/sin/exp. If arithmetic combines them with shift dicts in disallowed
# ways, we raise.


def _scalar(c, nd):
    z = tuple([0] * nd)
    return {z: complex(c)}


def _k_marker(dim_idx, nd):
    # represented as a dict-like sentinel: {"_k": [(dim_idx, multiplier)], "const": 0}
    return {"_k": [(dim_idx, 1)], "const": 0.0}


def _is_k(d):
    return isinstance(d, dict) and "_k" in d


def _is_shift(d):
    return isinstance(d, dict) and "_k" not in d


def _add(a, b):
    if _is_k(a) and _is_k(b):
        merged = {}
        for di, m in a["_k"] + b["_k"]:
            merged[di] = merged.get(di, 0) + m
        return {"_k": [(di, m) for di, m in merged.items() if m != 0],
                "const": a["const"] + b["const"]}
    if _is_k(a) and _is_shift(b):
        # k + scalar (shift {(0,..0): c}) -> k-marker with const += c
        if set(b.keys()) - {tuple([0] * len(next(iter(a["_k"]), (0, 0))[0:0]))}:
            pass
        return _add_k_scalar(a, b)
    if _is_shift(a) and _is_k(b):
        return _add_k_scalar(b, a)
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def _add_k_scalar(km, sh):
    # sh must contain only zero-shift keys
    nd = None
    z = None
    for key in sh.keys():
        nd = len(key)
        z = tuple([0] * nd)
        break
    for key in sh:
        if key != z:
            raise ValueError("cannot add k-symbol to a non-scalar Fourier expression")
    return {"_k": list(km["_k"]), "const": km["const"] + sh.get(z, 0)}


def _scale(d, c):
    if _is_k(d):
        return {"_k": [(di, m * c) for di, m in d["_k"]], "const": d["const"] * c}
    return {k: v * c for k, v in d.items()}


def _mul(a, b):
    if _is_k(a) and _is_k(b):
        raise ValueError("cannot multiply two momentum expressions outside cos/sin/exp")
    if _is_k(a):
        return _mul_k_scalar(a, b)
    if _is_k(b):
        return _mul_k_scalar(b, a)
    out = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            kc = tuple(x + y for x, y in zip(ka, kb))
            out[kc] = out.get(kc, 0) + va * vb
    return {k: v for k, v in out.items() if v != 0}


def _mul_k_scalar(km, sh):
    z = None
    for key in sh.keys():
        z = tuple([0] * len(key))
        break
    for key in sh:
        if key != z:
            raise ValueError("cannot multiply k-symbol by non-scalar Fourier expression")
    s = sh.get(z, 0)
    return {"_k": [(di, m * s) for di, m in km["_k"]], "const": km["const"] * s}


def _div(a, b):
    if _is_k(b):
        raise ValueError("cannot divide by k-symbol")
    if _is_shift(b):
        z = next(iter(b))
        if any(c != 0 for c in z) or len(b) != 1:
            raise ValueError("can only divide by scalar")
        c = b[z]
        if _is_k(a):
            return _scale(a, 1.0 / c)
        return {k: v / c for k, v in a.items()}
    raise TypeError


def _pow(a, b):
    if _is_k(a) or _is_k(b):
        raise ValueError("cannot exponentiate k-symbol expressions")
    if not (len(b) == 1):
        raise ValueError("exponent must be scalar")
    z = next(iter(b))
    if any(c != 0 for c in z):
        raise ValueError("exponent must be scalar (zero shift)")
    e = b[z]
    if not float(e.real).is_integer() or e.imag != 0:
        raise ValueError("only integer exponents supported")
    n = int(e.real)
    if n < 0:
        raise ValueError("negative exponents not supported")
    if n == 0:
        nd = len(next(iter(a)))
        return _scalar(1, nd)
    out = a
    for _ in range(n - 1):
        out = _mul(out, a)
    return out


def _apply_func(name, arg, nd):
    if not _is_k(arg):
        # constant under cos/sin/exp -> shift of zero with computed value
        z = next(iter(arg))
        if any(c != 0 for c in z) or len(arg) != 1:
            raise ValueError(f"argument to {name} must be a real linear combination of momentum symbols")
        c = arg[z]
        if name == "cos":
            return _scalar(np.cos(c), nd)
        if name == "sin":
            return _scalar(np.sin(c), nd)
        if name == "exp":
            return _scalar(np.exp(c), nd)
    # k-marker case
    const = arg["_k"], arg["const"]
    shift = [0] * nd
    for di, m in arg["_k"]:
        if not float(np.real(m)).is_integer() or np.imag(m) != 0:
            raise ValueError(f"non-integer momentum coefficient under {name}: {m}")
        shift[di] += int(np.real(m))
    shift = tuple(shift)
    neg_shift = tuple(-s for s in shift)
    phase = np.exp(1j * arg["const"]) if arg["const"] != 0 else 1.0
    if name == "cos":
        return {shift: 0.5 * phase, neg_shift: 0.5 * np.conj(phase)}
    if name == "sin":
        return {shift: -0.5j * phase, neg_shift: 0.5j * np.conj(phase)}
    if name == "exp":
        # exp(i * (k.shift + const)) -> single-shift of phase
        # but bare exp(...) here is ambiguous; assume the user meant exp(i*...).
        # Document: exp(...) inside momentum DSL is interpreted as exp(i*...) for k-args.
        return {shift: phase}
    raise ValueError(f"unknown function {name}")

