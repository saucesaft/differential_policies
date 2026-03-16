"""smooth_mjx — sigmoid contact smoothing for MuJoCo MJX.

replaces the hard contact switch `active = pos < 0` in MJX's constraint
functions with a differentiable sigmoid approximation, following
Schwarke et al. (CoRL 2025, arXiv 2404.02887).

MUST be called before any JIT compilation
    import smooth_mjx
    smooth_mjx.enable(kappa=300.0)
"""

import inspect
import textwrap

_PATCHED = False
_KAPPA = None

_TARGET_FUNCTIONS = [
    '_efc_contact_frictionless',
    '_efc_contact_pyramidal',
    '_efc_contact_elliptic',
]

_HARD_PATTERN = 'active = pos < 0'


def _make_soft_pattern(kappa: float) -> str:
    return f'active = jax.nn.sigmoid(-pos * {float(kappa)})'


def enable(kappa: float = 300.0) -> None: 
    """
    monkey-patching MJX contact functions to use sigmoid smoothing
    """
    global _PATCHED, _KAPPA

    if _PATCHED:
        if kappa != _KAPPA:
            raise RuntimeError(
                f'smooth_mjx already enabled with kappa={_KAPPA}. '
                'Cannot re-enable with different kappa after JIT compilation.'
            )
        return

    import mujoco.mjx._src.constraint as _cmod

    ns = vars(_cmod)
    soft_pattern = _make_soft_pattern(kappa)

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
        exec_ns = dict(ns)
        exec(compile(patched_src, f'<smooth_mjx:{fn_name}>', 'exec'), exec_ns)

        # replace in the module's globals, make_constraint calls by name
        ns[fn_name] = exec_ns[fn_name]

    _PATCHED = True
    _KAPPA = kappa
    print(f'[smooth_mjx] Enabled sigmoid contact smoothing (kappa={kappa})')


def is_enabled() -> bool:
    return _PATCHED


def kappa() -> float:
    return _KAPPA
