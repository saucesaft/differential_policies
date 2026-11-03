import dataclasses

import jax
import jax.numpy as jnp
import warp as wp

import mujoco
import mujoco_warp as mjw
from mujoco import mjx

from envs.g1_warp import make_g1_warp, set_domain_randomization, record_peak, check_overflow, NCONMAX, NJMAX

mjw.enable_grad()

SUBSTEPS = 10

# cache dicts
_G1 = {}
_BACKWARDS = {}
_STAGES = {}
_GRAPHS = {}

# replay each forward / backward as one captured CUDA graph instead of launching
# its few thousand kernels from Python (~10x faster at B=64). False = launch them
# one by one, same results; kept for testing the graphs against.
USE_GRAPHS = True

# warp kernel used for adding the final ctrl gradients
@wp.kernel
def _add_into(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float)):
    w, i = wp.tid() # get thread index and dimension (world, actuator)
    dst[w, i] += src[w, i]

# solver iterations: None = the XML's setting (3, what training uses).
# set to e.g. 50 for a converged solver
ITERATIONS = None

def _get_g1(B):
    key = (B, ITERATIONS)   # separate model per setting, so changing it never reuses a stale one
    if key not in _G1: _G1[key] = make_g1_warp( nworld=B, iterations=ITERATIONS, ls_iterations=ITERATIONS )
    return _G1[key]

@dataclasses.dataclass
class _Bwd: # everything the backward pass needs besides g1, allocated once per batch size
    d_out: mjw.Data                 # output buffer for the out-of-place taped step
    bc: object                      # IFT scratch, tied to _get_g1(B).d
    ck_qpos: wp.array               # (SUBSTEPS, B, nq) state before each substep
    ck_qvel: wp.array               # (SUBSTEPS, B, nv)
    ck_warm: wp.array               # (SUBSTEPS, B, nv)
    ck_time: wp.array               # (SUBSTEPS, B)
    carry_q: wp.array               # (B, nq) running cotangent dL/dqpos
    carry_v: wp.array               # (B, nv) running cotangent dL/dqvel

def _get_bwd(B): # just templating to not allocate on every call
    key = (B, ITERATIONS)   # same key as _G1: bc is tied to that g1.d
    if key not in _BACKWARDS:
        g1 = _get_g1( B )
        nq, nv = g1.mjm.nq, g1.mjm.nv

        # the arrays we differentiate w.r.t.: gives them a .grad buffer
        g1.d.qpos.requires_grad = True
        g1.d.qvel.requires_grad = True
        g1.d.ctrl.requires_grad = True

        _BACKWARDS[key] = _Bwd(
            d_out   = mjw.put_data(g1.mjm, mujoco.MjData(g1.mjm), nworld=B, nconmax=NCONMAX, njmax=NJMAX),
            bc      = mjw.create_backward_context(g1.m, g1.d),
            ck_qpos = wp.empty( (SUBSTEPS, B, nq), dtype=float ),
            ck_qvel = wp.empty( (SUBSTEPS, B, nv), dtype=float ),
            ck_warm = wp.empty( (SUBSTEPS, B, nv), dtype=float ),
            ck_time = wp.empty( (SUBSTEPS, B),     dtype=float ),
            carry_q = wp.zeros( (B, nq), dtype=float ),
            carry_v = wp.zeros( (B, nv), dtype=float ),
        )
    return _BACKWARDS[key]

@dataclasses.dataclass
class _Stage: # fixed-address copies of the JAX arguments: a captured graph bakes in
              # pointers, and XLA hands us different buffers from call to call
    qpos: wp.array          # (B, nq)
    qvel: wp.array          # (B, nv)
    warm: wp.array          # (B, nv)
    ctrl: wp.array          # (B, nu)
    friction: wp.array      # (B,)
    added_mass: wp.array    # (B,)
    ct_qpos: wp.array       # (B, nq) incoming cotangents (backward only)
    ct_qvel: wp.array       # (B, nv)
    g_ctrl: wp.array        # (B, nu) accumulated dL/dctrl (backward only)

def _get_stage(B):
    key = (B, ITERATIONS)
    if key not in _STAGES:
        mjm = _get_g1( B ).mjm
        nq, nv, nu = mjm.nq, mjm.nv, mjm.nu
        _STAGES[key] = _Stage(
            qpos       = wp.zeros( (B, nq), dtype=float ),
            qvel       = wp.zeros( (B, nv), dtype=float ),
            warm       = wp.zeros( (B, nv), dtype=float ),
            ctrl       = wp.zeros( (B, nu), dtype=float ),
            friction   = wp.zeros( B,       dtype=float ),
            added_mass = wp.zeros( B,       dtype=float ),
            ct_qpos    = wp.zeros( (B, nq), dtype=float ),
            ct_qvel    = wp.zeros( (B, nv), dtype=float ),
            g_ctrl     = wp.zeros( (B, nu), dtype=float ),
        )
    return _STAGES[key]

def _graph(kind, B, body):
    # body as a CUDA graph, captured on first use per (kind, B, ITERATIONS)

    # function runs once uncaptured first: that loads the kernel modules and lets
    # anything lazily allocated get allocated outside the capture

    key = (kind, B, ITERATIONS)
    if key not in _GRAPHS:
        body()
        with wp.ScopedCapture() as cap:
            body()
        _GRAPHS[key] = cap.graph
    return _GRAPHS[key]

def _run(kind, B, body):
    if USE_GRAPHS: wp.capture_launch( _graph(kind, B, body) )
    else:          body()

def _load_inputs(g1, st):
    set_domain_randomization( g1, st.friction, st.added_mass )
    wp.copy( g1.d.qpos, st.qpos )
    wp.copy( g1.d.qvel, st.qvel )
    wp.copy( g1.d.qacc_warmstart, st.warm )
    wp.copy( g1.d.ctrl, st.ctrl )

def _fwd_body(g1, st): # what the forward graph replays: only fixed-address arrays
    _load_inputs(g1, st)

    for _ in range(SUBSTEPS):
        mjw.step(g1.m, g1.d)

    record_peak(g1) # check_overflow() reads it later, from outside the graph

# you could say this is the most relevant function
# backwards loop, abstracted to manually capture it in a graph
def _bwd_body(g1, st, bw):
    B, nu = g1.nworld, g1.mjm.nu # relevant dimensions
    d_out, bc = bw.d_out, bw.bc # unpack
    ck_qpos, ck_qvel, ck_warm, ck_time = bw.ck_qpos, bw.ck_qvel, bw.ck_warm, bw.ck_time # unpack
    carry_q, carry_v = bw.carry_q, bw.carry_v # unpack

    # load inputs into g1.d
    _load_inputs(g1, st)

    # recompute forward pass and save intermediate steps (similar to jax.checkpoint)
    for j in range(SUBSTEPS):
        wp.copy( ck_qpos[j], g1.d.qpos )
        wp.copy( ck_qvel[j], g1.d.qvel )
        wp.copy( ck_warm[j], g1.d.qacc_warmstart )
        wp.copy( ck_time[j], g1.d.time )

        mjw.step(g1.m, g1.d)

    # initialize the carry
    wp.copy( carry_q, st.ct_qpos )
    wp.copy( carry_v, st.ct_qvel )

    st.g_ctrl.zero_() # accumulator: starts at zero every call

    # reverse loop (basically the chain rule through the substeps)
    for j in reversed(range(SUBSTEPS)):

        # restore inputs (so that state now matches what the step saw in the forward pass)
        wp.copy( g1.d.qpos,             ck_qpos[j] )
        wp.copy( g1.d.qvel,             ck_qvel[j] )
        wp.copy( g1.d.qacc_warmstart,   ck_warm[j] )
        wp.copy( g1.d.time,             ck_time[j] )

        # activate diff mode
        with mjw.backward_context(bc):
            tape = wp.Tape()
            with tape:
                mjw.step(g1.m, g1.d, d_out)

            tape.zero()

            # tell the tape what the loss's gradient is at this step's output
            wp.copy( d_out.qpos.grad, carry_q )
            wp.copy( d_out.qvel.grad, carry_v )

            # run the adjoint
            tape.backward()

        # re-set the carry
        wp.copy( carry_q, g1.d.qpos.grad )
        wp.copy( carry_v, g1.d.qvel.grad )

        # add the gradient accumulated per-substep
        wp.launch( _add_into, dim=(B, nu), inputs=[g1.d.ctrl.grad, st.g_ctrl])

def _fwd_impl(
               qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
               warm: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float),
               friction: wp.array(dtype=float), added_mass: wp.array(dtype=float),

               qpos_out: wp.array2d(dtype=float), qvel_out: wp.array2d(dtype=float),
               warm_out: wp.array2d(dtype=float)
            ):
    B = qpos.shape[0]
    g1, st = _get_g1( B ), _get_stage( B )

    # JAX buffers -> fixed staging buffers
    for src, dst in ((qpos, st.qpos), (qvel, st.qvel), (warm, st.warm), (ctrl, st.ctrl),
                     (friction, st.friction), (added_mass, st.added_mass)):
        wp.copy( dst, src )

    _run( 'fwd', B, lambda: _fwd_body(g1, st) )

    wp.copy(qpos_out, g1.d.qpos)
    wp.copy(qvel_out, g1.d.qvel)
    wp.copy(warm_out, g1.d.qacc_warmstart)

def _bwd_impl(
               qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
               warm: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float),
               friction: wp.array(dtype=float), added_mass: wp.array(dtype=float),
               ct_qpos: wp.array2d(dtype=float), ct_qvel: wp.array2d(dtype=float),

               g_qpos_out: wp.array2d(dtype=float), g_qvel_out: wp.array2d(dtype=float),
               g_ctrl_out: wp.array2d(dtype=float)
        ):

    B = qpos.shape[0] # relevant dimensions
    g1, st, bw = _get_g1( B ), _get_stage( B ), _get_bwd( B )

    for src, dst in ( (qpos, st.qpos), (qvel, st.qvel), (warm, st.warm), (ctrl, st.ctrl),
                     (friction, st.friction), (added_mass, st.added_mass),
                     (ct_qpos, st.ct_qpos), (ct_qvel, st.ct_qvel) ):
        wp.copy( dst, src )

    _run( 'bwd', B, lambda: _bwd_body(g1, st, bw) )

    # finalize output arrays by copying carry's values
    wp.copy( g_qpos_out, bw.carry_q )
    wp.copy( g_qvel_out, bw.carry_v )
    wp.copy( g_ctrl_out, st.g_ctrl )

def check_overflow_all():
    # raise if any cached model has overflowed its contact / constraint buffers
    for g1 in _G1.values():
        check_overflow(g1)

# expose warp function to jax as a callable primitive
_fwd_call = wp.jax_callable( _fwd_impl, num_outputs=3, graph_mode=wp.JaxCallableGraphMode.NONE )
_bwd_call = wp.jax_callable( _bwd_impl, num_outputs=3, graph_mode=wp.JaxCallableGraphMode.NONE )

def _fwd_per_env(qpos, qvel, warm, ctrl, friction, added_mass):
    # my other code is "per-env" with qpos of shape (nq,) for example, so i add a fake batch
    nq, nv = qpos.shape[0], qvel.shape[0]
    q, v, w = _fwd_call(
                            qpos[None], qvel[None], warm[None], ctrl[None], friction[None], added_mass[None],
                            output_dims={ 'qpos_out': (1, nq), 'qvel_out': (1, nv), 'warm_out': (1, nv) }
                        )
    return q[0], v[0], w[0]

@jax.custom_vjp
def warp_control_step(qpos, qvel, warm, ctrl, friction, added_mass):
    return _fwd_per_env(qpos, qvel, warm, ctrl, friction, added_mass)

def _vjp_fwd(*args):
    return warp_control_step(*args), args # residuals = the inputs

def _vjp_bwd(res, cts):
    qpos, qvel, warm, ctrl, friction, added_mass = res
    ct_qpos, ct_qvel, ct_warm = cts # warm start gets no gradient
    nq, nv, nu = qpos.shape[0], qvel.shape[0], ctrl.shape[0]
    g_q, g_v, g_c = _bwd_call(
                                qpos[None], qvel[None], warm[None], ctrl[None], friction[None], added_mass[None], ct_qpos[None], ct_qvel[None],
                                output_dims={ 'g_qpos_out': (1, nq), 'g_qvel_out': (1, nv), 'g_ctrl_out': (1, nu) }
                            )
    return ( g_q[0], g_v[0], jnp.zeros_like(warm), g_c[0], 
            jnp.zeros_like(friction), jnp.zeros_like(added_mass))

warp_control_step.defvjp( _vjp_fwd, _vjp_bwd )

def warp_pipeline_step(sys, data, ctrl, friction, added_mass):
    # one control step on MJWarp, returned as an mjx.Data: drop-in for
    # DiffG1._pipeline_step_dr(data, ctrl, sys), plus the DR values Warp needs.

    # Warp only hands back qpos/qvel/warm start; the fields the env reads
    # (xpos, xquat, cvel, subtree_com, geom_xpos, actuator_force) are rebuilt in
    # JAX so reward gradients reach warp_control_step's VJP through autodiff.
    # sys must be the env's randomized model (added mass changes subtree_com).
    # Fields not rebuilt (contacts, efc_*, qacc, time) stay stale: don't read them.

    qpos, qvel, warm = warp_control_step(data.qpos, data.qvel, data.qacc_warmstart,
                                         ctrl, friction, added_mass)
    data = data.replace(qpos=qpos, qvel=qvel, qacc_warmstart=warm, ctrl=ctrl)
    data = mjx.kinematics(sys, data)     # xpos, xquat, xipos, geom_xpos
    data = mjx.com_pos(sys, data)        # subtree_com, cinert, cdof
    data = mjx.transmission(sys, data)   # actuator_length, actuator_moment
    data = mjx.fwd_velocity(sys, data)   # cvel, actuator_velocity
    data = mjx.fwd_actuation(sys, data)  # actuator_force
    return data
