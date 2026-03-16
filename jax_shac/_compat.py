"""Compatibility shim for brax 0.9.4 + JAX >= 0.5.

Patches deprecated/removed JAX attributes that brax v1 still uses.

Must be imported before brax.

Note: the old frozen-contact stop_gradient patch (for the RNN surrogate approach)
has been removed. MJX contact gradients are now handled by smooth_mjx.enable()
which replaces the hard contact switch with a differentiable sigmoid.
"""
import jax
import jax.interpreters.batching as _batching

# brax v1 jumpy.py uses jax.interpreters.batching.BatchTracer
if not hasattr(_batching, "BatchTracer"):
    _batching.BatchTracer = type("_DeprecatedBatchTracer", (), {})

# brax wrappers use jax.tree_map (removed in JAX 0.9)
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map
