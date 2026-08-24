import re
import numpy as np

from .expr import safe_eval, safe_format, make_coef_fn, make_kwargs_fn
from .momentum import parse_momentum
from ..core.operators import OperatorTerm


# d-key parsing: "d_1", "d_1_2", "d_0" etc. Trailing integers are gamma indices.
_DKEY = re.compile(r"^d(?:_(\d+))+$")


def parse_d_key(key):
    m = _DKEY.match(key)
    if not m:
        raise ValueError(f"invalid d-key '{key}' (expected pattern d_<int>(_<int>)*)")
    parts = key.split("_")[1:]
    return tuple(int(p) for p in parts)


# real-space term syntax inside d-vector strings:
#   coeff[func_name, kwarg1=val1, kwarg2=val2]
# coefficient before the bracket is optional ("[func]" -> coefficient 1)
# multiple bracket terms can be separated with '+' or '-'
# momentum part is everything outside brackets

_BRACKET = re.compile(r"(?P<sign>[+-]?)\s*"
                      r"(?P<coef>(?:[^\[\]+\-]|\([^()]*\))+)?\s*"
                      r"\[\s*(?P<body>[^\]]+)\s*\]")


def split_dstring(s):
    momentum_parts = []
    site_terms = []
    last = 0
    for m in _BRACKET.finditer(s):
        before = s[last:m.start()].strip()
        if before:
            if not before.endswith(("+", "-")):
                momentum_parts.append(before)
            else:
                momentum_parts.append(before[:-1].strip())
        sign = m.group("sign") or "+"
        coef_raw = (m.group("coef") or "1").strip()
        if coef_raw.endswith("*"):
            coef_raw = coef_raw[:-1].strip()
        if not coef_raw:
            coef_raw = "1"
        if sign == "-":
            coef_raw = f"-({coef_raw})"
        body = m.group("body").strip()
        site_terms.append((coef_raw, body))
        last = m.end()
    tail = s[last:].strip()
    if tail and not tail.startswith(("+", "-", "")):
        momentum_parts.append(tail)
    elif tail:
        momentum_parts.append(tail)
    momentum_expr = " ".join(p for p in momentum_parts if p)
    momentum_expr = momentum_expr.strip()
    if momentum_expr in ("", "+", "-"):
        momentum_expr = None
    return momentum_expr, site_terms


def parse_site_term_body(body):
    # "func_name, kw1=expr, kw2=expr"
    parts = [p.strip() for p in body.split(",")]
    name = parts[0]
    kwargs = {}
    for kv in parts[1:]:
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"site-term arg '{kv}' missing '='")
        k, v = kv.split("=", 1)
        kwargs[k.strip()] = v.strip()
    return name, kwargs


def make_hops_factory(momentum_expr, dim_symbols):
    if momentum_expr is None or momentum_expr == "":
        return None
    expr = momentum_expr
    syms = tuple(dim_symbols)
    def factory(params, _e=expr, _s=syms):
        return parse_momentum(_e, _s, params)
    return factory


def make_site_terms(site_term_specs, registered_fns):
    out = []
    for coef_expr, body in site_term_specs:
        name, kwargs_spec = parse_site_term_body(body)
        if callable(name):
            fn = name
        else:
            if name not in registered_fns:
                raise KeyError(f"site function '{name}' not registered")
            fn = registered_fns[name]
        out.append((fn, make_kwargs_fn(kwargs_spec), make_coef_fn(coef_expr)))
    return out


def make_operator_term(d_key, value, dim_symbols, registered_fns,
                       edge_modifier=None, edge_modifier_keys=(),
                       edge_modifier_wants_ctx=False, selector_mask=None):
    gamma_indices = parse_d_key(d_key)

    if value is None:
        return None

    if isinstance(value, str):
        momentum_expr, site_specs = split_dstring(value)
    elif isinstance(value, dict):
        momentum_expr = value.get("momentum")
        site_specs = []
        for entry in value.get("real_terms", ()):
            if len(entry) == 3:
                fn, kwargs, coef = entry
            elif len(entry) == 2:
                fn, kwargs = entry
                coef = 1.0
            else:
                raise ValueError("real_terms entry must be (fn, kwargs[, coef])")
            site_specs.append((coef, fn, kwargs))
    elif isinstance(value, tuple) and len(value) == 2:
        momentum_expr, real_list = value
        site_specs = []
        for entry in real_list:
            if len(entry) == 3:
                fn, kwargs, coef = entry
            elif len(entry) == 2:
                fn, kwargs = entry
                coef = 1.0
            else:
                raise ValueError("real_terms entry must be (fn, kwargs[, coef])")
            site_specs.append((coef, fn, kwargs))
    else:
        raise TypeError(f"d-value of type {type(value).__name__} not supported")

    hops_factory = make_hops_factory(momentum_expr, dim_symbols)

    site_terms = []
    if isinstance(value, str):
        for coef_expr, body in site_specs:
            name, kwargs_spec = parse_site_term_body(body)
            fn = registered_fns.get(name)
            if fn is None:
                raise KeyError(f"site function '{name}' not registered")
            site_terms.append((fn, make_kwargs_fn(kwargs_spec), make_coef_fn(coef_expr)))
    else:
        for coef, fn, kwargs in site_specs:
            if isinstance(fn, str):
                if fn not in registered_fns:
                    raise KeyError(f"site function '{fn}' not registered")
                fn_obj = registered_fns[fn]
            else:
                fn_obj = fn
            site_terms.append((fn_obj, make_kwargs_fn(kwargs), make_coef_fn(coef)))

    return OperatorTerm(
        hops_factory=hops_factory,
        site_terms=site_terms,
        edge_modifier=edge_modifier,
        edge_modifier_keys=tuple(edge_modifier_keys),
        edge_modifier_wants_ctx=edge_modifier_wants_ctx,
        gamma_indices=gamma_indices,
        name=d_key,
        selector_mask=selector_mask,
    )

