"""Compatibility shim for brax 0.9.4 + JAX >= 0.5.

Patches deprecated/removed JAX attributes that brax v1 still uses.
Also patches MJX contact activation to use stop_gradient (frozen contact
backward pass) so that gradients flow through the constraint solver but
not through the binary contact on/off decision.

Must be imported before brax.
"""
import jax
import jax.interpreters.batching as _batching

# brax v1 jumpy.py uses jax.interpreters.batching.BatchTracer
if not hasattr(_batching, "BatchTracer"):
    _batching.BatchTracer = type("_DeprecatedBatchTracer", (), {})

# brax wrappers use jax.tree_map (removed in JAX 0.9)
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map


# ---------------------------------------------------------------------------
# MJX frozen-contact patch: stop_gradient on contact activation
# ---------------------------------------------------------------------------
def _patch_mjx_frozen_contacts():
    """Replace MJX contact constraint functions with versions that apply
    jax.lax.stop_gradient to the binary contact activation mask.

    Forward pass is unchanged (contacts activate/deactivate normally).
    Backward pass treats the contact set as fixed, so gradients only flow
    through the constraint solver (smooth) and not through the discrete
    contact on/off switch.
    """
    from jax import numpy as jp
    import numpy as np
    from typing import Optional

    import mujoco.mjx._src.constraint as _cmod
    from mujoco.mjx._src.types import (
        Contact, Data, DataJAX, Model, ModelJAX, OptionJAX,
    )
    from mujoco.mjx._src import support

    _row = _cmod._row
    _Efc = _cmod._Efc

    # --- frictionless (condim=1) ---
    _orig_frictionless = _cmod._efc_contact_frictionless

    def _efc_contact_frictionless(m: Model, d: Data) -> Optional[_Efc]:
        if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
            raise ValueError(
                '_efc_contact_frictionless requires JAX backend implementation.')
        con_id = np.nonzero(d._impl.contact.dim == 1)[0]
        if con_id.size == 0:
            return None

        @jax.vmap
        def rows(c: Contact):
            pos = c.dist - c.includemargin
            active = jax.lax.stop_gradient(pos < 0)  # FROZEN
            body1, body2 = jp.array(m.geom_bodyid)[c.geom]
            jac1p, _ = support.jac(m, d, c.pos, body1)
            jac2p, _ = support.jac(m, d, c.pos, body2)
            j = (c.frame @ (jac2p - jac1p).T)[0]
            invweight = m.body_invweight0[body1, 0] + m.body_invweight0[body2, 0]
            return _row(
                j * active, pos * active, pos, invweight,
                c.solref, c.solimp, c.includemargin, jp.zeros_like(pos),
            )

        contact = jax.tree_util.tree_map(lambda x: x[con_id], d._impl.contact)
        return rows(contact)

    # --- pyramidal (condim=3,4,6 — ANYmal uses condim=3) ---
    _orig_pyramidal = _cmod._efc_contact_pyramidal

    def _efc_contact_pyramidal(m: Model, d: Data, condim: int) -> Optional[_Efc]:
        if (
            not isinstance(m._impl, ModelJAX)
            or not isinstance(d._impl, DataJAX)
            or not isinstance(m.opt._impl, OptionJAX)
        ):
            raise ValueError(
                '_efc_contact_pyramidal requires JAX backend implementation.')
        con_id = np.nonzero(d._impl.contact.dim == condim)[0]
        if con_id.size == 0:
            return None

        @jax.vmap
        def rows(c: Contact):
            pos = c.dist - c.includemargin
            active = jax.lax.stop_gradient(pos < 0)  # FROZEN
            body1, body2 = jp.array(m.geom_bodyid)[c.geom]
            jac1p, jac1r = support.jac(m, d, c.pos, body1)
            jac2p, jac2r = support.jac(m, d, c.pos, body2)
            diff = c.frame @ (jac2p - jac1p).T
            if condim > 3:
                diff = jp.concatenate(
                    (diff, (c.frame @ (jac2r - jac1r).T)), axis=0)
            fri = jp.repeat(
                c.friction[:condim - 1], 2, axis=0).at[1::2].mul(-1)
            j = diff[0] + jp.repeat(diff[1:condim], 2, axis=0) * fri[:, None]
            invweight = (m.body_invweight0[body1, 0]
                         + m.body_invweight0[body2, 0])
            invweight = invweight + fri[0] * fri[0] * invweight
            invweight = invweight * 2 * fri[0] * fri[0] / m.opt.impratio
            return _row(
                j * active, pos * active, pos, invweight,
                c.solref, c.solimp, c.includemargin, jp.zeros_like(pos),
            )

        contact = jax.tree_util.tree_map(lambda x: x[con_id], d._impl.contact)
        return jax.tree_util.tree_map(jp.concatenate, rows(contact))

    # Apply patches
    _cmod._efc_contact_frictionless = _efc_contact_frictionless
    _cmod._efc_contact_pyramidal = _efc_contact_pyramidal

_patch_mjx_frozen_contacts()
