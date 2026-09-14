"""
Differentiable MJX. 

1. Contact smoothing: replaces the hard contact switch `active = pos < 0` in
   MJX's constraint functions with a differentiable sigmoid approximation,
   following Schwarke et al. (CoRL 2025, arXiv 2404.02887).

2. We rewrite the call to run `_while_loop_scan` and have better Newton convergence.

MUST be called before any JIT compilation
    import smooth_mjx
    smooth_mjx.enable(kappa=300.0)
"""

import inspect
import textwrap

_PATCHED = False
_KAPPA = None
_STRAIGHT_THROUGH = None
_SOLVER_PATCHED = False

_TARGET_FUNCTIONS = [
    '_efc_contact_frictionless',
    '_efc_contact_pyramidal',
    '_efc_contact_elliptic',
]

_HARD_PATTERN = 'active = pos < 0'

# the single non-differentiable branch of solver.solve (the outer Newton loop).
_WHILE_LOOP_PATTERN = 'ctx = jax.lax.while_loop(cond, body, ctx)'
_WHILE_LOOP_SCAN = 'ctx = _while_loop_scan(cond, body, ctx, int(m.opt.iterations))'


def _make_soft_pattern(kappa: float, straight_through: bool = False) -> str:
    soft = f'jax.nn.sigmoid(-pos * {float(kappa)})'
    
    if straight_through:
        # forward: exact hard switch (pos < 1). zero physics distortion.
        # backward: d(active)/d(pos) = sigmoid'  (straight-through estimator,
        
        return (f'active = {soft}; '
                'active = active + jax.lax.stop_gradient((pos < 0) - active)')

    return f'active = {soft}'


def _recompile_in_module(ns, fn_name, patched_src, tag):
    """exec a rewritten function in its own module's namespace so all helpers
    are in scope, then rebind it in that module's globals."""
    exec_ns = dict(ns)
    exec(compile(patched_src, f'<smooth_mjx:{tag}>', 'exec'), exec_ns)
    ns[fn_name] = exec_ns[fn_name]


def _patch_newton_loop() -> None:
    """Route solver.solve's outer Newton while_loop through the scan-based
    bounded loop, so reverse-mode autodiff works at iterations > 1.

    solver.py contains exactly one `jax.lax.while_loop` (the Newton loop) --
    the linesearch calls `_while_loop_scan` directly and is unaffected.
    forward.py:452 resolves `solver.solve` at call time, so replacing the
    module attribute is sufficient.
    """
    global _SOLVER_PATCHED
    if _SOLVER_PATCHED:
        return

    import mujoco.mjx._src.solver as _smod

    ns = vars(_smod)
    src = textwrap.dedent(inspect.getsource(ns['solve']))

    if _WHILE_LOOP_PATTERN not in src:
        raise RuntimeError(
            f'[smooth_mjx] Pattern "{_WHILE_LOOP_PATTERN}" not found in '
            'solver.solve. MJX version may have changed.'
        )

    _recompile_in_module(
        ns, 'solve', src.replace(_WHILE_LOOP_PATTERN, _WHILE_LOOP_SCAN), 'solve'
    )

    _SOLVER_PATCHED = True


def enable(kappa: float = 300.0, straight_through: bool = False) -> None:
    """
    monkey-patching MJX contact functions to use sigmoid smoothing, and the MJX
    Newton solve to be reverse-mode differentiable at iterations > 1.

    straight_through=True keeps the HARD switch in the forward pass and applies
    the sigmoid only to the backward pass (gradient), so forward dynamics are
    bit-identical to unpatched MJX.
    """
    global _PATCHED, _KAPPA, _STRAIGHT_THROUGH

    _patch_newton_loop()

    if _PATCHED:
        if kappa != _KAPPA or straight_through != _STRAIGHT_THROUGH:
            raise RuntimeError(
                f'smooth_mjx already enabled with kappa={_KAPPA}, '
                f'straight_through={_STRAIGHT_THROUGH}. '
                'Cannot re-enable with different settings after JIT compilation.'
            )
        return

    import mujoco.mjx._src.constraint as _cmod

    ns = vars(_cmod)
    soft_pattern = _make_soft_pattern(kappa, straight_through)

    for fn_name in _TARGET_FUNCTIONS:
        fn = ns[fn_name]
        src = inspect.getsource(fn)
        src = textwrap.dedent(src)

        if _HARD_PATTERN not in src:
            raise RuntimeError(
                f'[smooth_mjx] Pattern "{_HARD_PATTERN}" not found in '
                f'{fn_name}. MJX version may have changed.'
            )

        patched_src = src.replace(_HARD_PATTERN, soft_pattern)

        # execute in the constraint module's own namespace so all helpers
        # (_row, support, jp, jax, Model, Data, etc.) are in scope.
        _recompile_in_module(ns, fn_name, patched_src, fn_name)

    _PATCHED = True
    _KAPPA = kappa
    _STRAIGHT_THROUGH = straight_through
    mode = 'straight-through (hard fwd / soft bwd)' if straight_through else 'sigmoid fwd+bwd'
    print(f'[smooth_mjx] Enabled contact smoothing (kappa={kappa}, {mode}) '
          'and reverse-mode Newton solve')


def is_enabled() -> bool:
    return _PATCHED


def newton_loop_is_patched() -> bool:
    return _SOLVER_PATCHED


def kappa() -> float:
    return _KAPPA
