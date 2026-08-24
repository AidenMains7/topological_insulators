import ast
import re
import operator as op


_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def safe_eval(expr, variables):
    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body, variables)


def _eval(node, variables):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"unsupported literal: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id == "j":
            return 1j
        raise NameError(f"unknown variable '{node.id}'")
    if isinstance(node, ast.BinOp):
        opf = _BIN_OPS.get(type(node.op))
        if opf is None:
            raise ValueError(f"unsupported op: {type(node.op).__name__}")
        return opf(_eval(node.left, variables), _eval(node.right, variables))
    if isinstance(node, ast.UnaryOp):
        opf = _UNARY_OPS.get(type(node.op))
        if opf is None:
            raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
        return opf(_eval(node.operand, variables))
    raise ValueError(f"unsupported syntax: {ast.dump(node)}")


_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def safe_format(template, variables):
    def repl(m):
        return str(safe_eval(m.group(1), variables))
    return _PLACEHOLDER.sub(repl, template)


def make_coef_fn(spec):
    if spec is None:
        return lambda params: 1.0
    if isinstance(spec, (int, float, complex)):
        c = complex(spec)
        return lambda params, _c=c: _c
    if callable(spec):
        return spec
    s = str(spec)
    if not _PLACEHOLDER.search(s):
        try:
            v = complex(safe_eval(s, {}))
            return lambda params, _v=v: _v
        except Exception:
            pass
    def _eval_with_params(params, _s=s):
        formatted = safe_format(_s, params)
        return complex(safe_eval(formatted, params))
    return _eval_with_params


def make_kwarg_value_fn(spec):
    """Like ``make_coef_fn`` but does not coerce the result to ``complex``.

    Used for kwargs passed to real-space / hopping-modifier callables, where
    we want ``disorder_seed=42`` to arrive as ``int(42)`` (not ``42+0j``),
    a string mass like ``M=1.5`` to arrive as ``float``, etc. Values that
    genuinely need to be complex (e.g. expressions containing ``j``) still
    come through as ``complex``, since ``safe_eval`` returns whatever
    Python type the arithmetic produces.
    """
    if spec is None:
        return lambda params: 1.0
    if isinstance(spec, (int, float, complex)):
        return lambda params, _v=spec: _v
    if callable(spec):
        return spec
    s = str(spec)
    if not _PLACEHOLDER.search(s):
        try:
            v = safe_eval(s, {})
            return lambda params, _v=v: _v
        except Exception:
            pass
    def _eval_with_params(params, _s=s):
        formatted = safe_format(_s, params)
        return safe_eval(formatted, params)
    return _eval_with_params


def make_kwargs_fn(kwargs_spec):
    if not kwargs_spec:
        return lambda params: {}
    items = []
    for k, v in kwargs_spec.items():
        if isinstance(v, str):
            fn = make_kwarg_value_fn(v)
            items.append((k, fn, True))
        else:
            items.append((k, v, False))
    def _build(params):
        out = {}
        for k, fn, is_expr in items:
            out[k] = fn(params) if is_expr else fn
        return out
    return _build

