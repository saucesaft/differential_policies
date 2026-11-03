import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax, jax.numpy as jnp, numpy as np, mujoco, pytest
from envs import g1_warp_jax as gj
from envs.register_g1_29dof import FOOT_PAIRS

@pytest.fixture( scope='module' )
def keyframe():
    mjm = gj._get_g1(1).mjm
    mjd = mujoco.MjData(mjm)

    mujoco.mj_resetDataKeyframe( mjm, mjd, mujoco.mj_name2id( mjm, mujoco.mjtObj.mjOBJ_KEY, 'knees_bent' ))
    mujoco.mj_forward(mjm, mjd)

    return mjm, mjd

def _base_friction(mjm):
    pid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_PAIR, FOOT_PAIRS[0])
    return float(mjm.pair_friction[pid, 0])

def _inputs(mjd, vx=0.0):
    # keyframe state; vx > 0 gives the pelvis a forward push so the feet slide
    # and friction actually changes the result
    qvel = mjd.qvel.copy()
    qvel[0] = vx
    return (jnp.asarray(mjd.qpos), jnp.asarray(qvel),
            jnp.asarray(mjd.qacc_warmstart), jnp.asarray(mjd.ctrl))

def _cpu_step(mjm, mjd, vx=0.0):
    d = mujoco.MjData(mjm)
    d.qpos[:], d.qvel[:], d.ctrl[:] = mjd.qpos, mjd.qvel, mjd.ctrl
    d.qvel[0] = vx
    d.qacc_warmstart[:] = mjd.qacc_warmstart
    for _ in range(gj.SUBSTEPS):
        mujoco.mj_step(mjm, d)
    return d

@pytest.mark.parametrize('vx', [0.0, 0.5])
def test_single_env_matches_cpu(keyframe, vx):
    # no DR (base friction, +0 kg) -> must equal CPU MuJoCo x SUBSTEPS
    mjm, mjd = keyframe
    qpos, qvel, warm, ctrl = _inputs(mjd, vx)
    q, v, w = jax.jit(gj._fwd_per_env)(
        qpos, qvel, warm, ctrl, jnp.float32(_base_friction(mjm)), jnp.float32(0.0))

    assert q.shape == (mjm.nq,) and v.shape == (mjm.nv,) and w.shape == (mjm.nv,)
    ref = _cpu_step(mjm, mjd, vx)
    print(f'\nvx={vx}  max|dqpos| {np.abs(np.asarray(q) - ref.qpos).max():.2e}'
          f'  max|dqvel| {np.abs(np.asarray(v) - ref.qvel).max():.2e}')
    np.testing.assert_allclose(np.asarray(q), ref.qpos, atol=1e-3)
    np.testing.assert_allclose(np.asarray(v), ref.qvel, atol=1e-2)

def test_vmap_batch(keyframe):
    # B=4: world 0 no DR, world 1 low friction, world 2 high friction, world 3 +3 kg
    mjm, mjd = keyframe
    B, mu = 4, _base_friction(mjm)
    single = _inputs(mjd, vx=0.5)
    batch = [jnp.broadcast_to(x, (B,) + x.shape) for x in single]
    friction = jnp.array([mu, 0.2, 1.2, mu], jnp.float32)
    added_mass = jnp.array([0.0, 0.0, 0.0, 3.0], jnp.float32)

    q, v, w = jax.jit(jax.vmap(gj._fwd_per_env))(*batch, friction, added_mass)
    assert q.shape == (B, mjm.nq) and v.shape == (B, mjm.nv) and w.shape == (B, mjm.nv)

    # row 0 == the unbatched call (separate nworld=1 model). MJWarp isn't bitwise
    # deterministic: the same call repeated spreads by ~2e-6 qpos / ~7e-5 qvel here
    q0, v0, _ = jax.jit(gj._fwd_per_env)(*single, jnp.float32(mu), jnp.float32(0.0))
    np.testing.assert_allclose(np.asarray(q[0]), np.asarray(q0), atol=1e-4)
    np.testing.assert_allclose(np.asarray(v[0]), np.asarray(v0), atol=1e-3)

    # DR worlds must actually differ from world 0
    for i in range(1, B):
        dv = float(jnp.abs(v[i] - v[0]).max())
        print(f'\nworld {i}: max|dqvel| vs world 0 = {dv:.2e}')
        assert dv > 1e-2, f'world {i} ignored its domain randomization'

def _proj_loss(qpos, qvel, warm, ctrl, friction, added_mass, rq, rv):
    # random projection of the next state: its gradient is a VJP with cotangent (rq, rv)
    q, v, _ = gj.warp_control_step(qpos, qvel, warm, ctrl, friction, added_mass)
    return (q * rq).sum() + (v * rv).sum()

@pytest.fixture
def converged(monkeypatch):
    # finite differences only agree with the IFT gradient when the solver has converged;
    # at the XML's 3 iterations the forward is too bumpy in its inputs for FD to mean anything
    monkeypatch.setattr(gj, 'ITERATIONS', 50)

@pytest.mark.parametrize('wrt', ['ctrl', 'qvel', 'qpos'])
def test_grad_matches_fd(keyframe, converged, wrt):
    mjm, mjd = keyframe
    rng = np.random.default_rng(0)
    qpos, qvel, warm, ctrl = _inputs(mjd)
    mu, dm = jnp.float32(_base_friction(mjm)), jnp.float32(0.0)
    rq = jnp.asarray(rng.normal(size=mjm.nq), jnp.float32)
    rv = jnp.asarray(rng.normal(size=mjm.nv), jnp.float32)

    def loss(x):
        args = {'qpos': qpos, 'qvel': qvel, 'ctrl': ctrl} | {wrt: x}
        return _proj_loss(args['qpos'], args['qvel'], warm, args['ctrl'], mu, dm, rq, rv)

    x0 = {'qpos': qpos, 'qvel': qvel, 'ctrl': ctrl}[wrt]
    g = jax.jit(jax.grad(loss))(x0)
    assert g.shape == x0.shape and bool(jnp.isfinite(g).all())

    # free joint: perturb only the joint angles (qpos[7:]), not the base position/quaternion
    lo = 7 if wrt == 'qpos' else 0
    eps = 1e-3
    loss_j = jax.jit(loss)
    for k in range(3):
        d = np.zeros(x0.shape[0], np.float32)
        d[lo:] = rng.normal(size=x0.shape[0] - lo)
        d = jnp.asarray(d / np.linalg.norm(d))
        fd = float(loss_j(x0 + eps * d) - loss_j(x0 - eps * d)) / (2 * eps)
        an = float(g @ d)
        print(f'\n{wrt} dir {k}: fd {fd:+.4f}  analytic {an:+.4f}')
        assert abs(fd - an) <= 2e-2 * abs(an) + 2e-2

def test_grad_vmap_matches_single(keyframe, converged):
    # batched backward (B=4, each world with its own DR) must give each world the same
    # gradient as running it alone. Converged solver: at 3 iterations the run-to-run
    # noise (below) reaches ~1e-2 of the gradient's scale, which would hide real bugs.
    mjm, mjd = keyframe
    rng = np.random.default_rng(1)
    B, mu = 4, _base_friction(mjm)
    single = _inputs(mjd, vx=0.5)
    batch = [jnp.broadcast_to(x, (B,) + x.shape) for x in single]
    friction = jnp.array([mu, 0.2, 1.2, mu], jnp.float32)
    added_mass = jnp.array([0.0, 0.0, 0.0, 3.0], jnp.float32)
    rq = jnp.asarray(rng.normal(size=mjm.nq), jnp.float32)
    rv = jnp.asarray(rng.normal(size=mjm.nv), jnp.float32)

    grad = jax.grad(_proj_loss, argnums=(0, 1, 3))   # wrt qpos, qvel, ctrl
    gb = jax.jit(jax.vmap(grad, in_axes=(0, 0, 0, 0, 0, 0, None, None)))(
        *batch, friction, added_mass, rq, rv)
    # MJWarp isn't bitwise deterministic: the same call repeated gives slightly different
    # gradients (~5e-4 abs here). So compare relative to the gradient's overall scale.
    for i in range(B):
        gi = jax.jit(grad)(*single, friction[i], added_mass[i], rq, rv)
        for name, a, b in zip(('qpos', 'qvel', 'ctrl'), gb, gi):
            rel = float(jnp.abs(a[i] - b).max() / jnp.abs(b).max())
            print(f'\nworld {i} d/d{name}: max diff / max |grad| = {rel:.1e}')
            assert rel < 5e-3, f'world {i} d/d{name}'

def test_graphs_match_launches(keyframe, converged, monkeypatch):
    # replaying the captured graphs == launching the same kernels one by one, forward
    # and backward. Two different inputs back to back: a replay must read the new
    # values from the staging buffers, not reuse whatever was captured.
    mjm, mjd = keyframe
    rng = np.random.default_rng(2)
    B, mu = 4, _base_friction(mjm)
    friction = jnp.array([mu, 0.2, 1.2, mu], jnp.float32)
    added_mass = jnp.array([0.0, 0.0, 0.0, 3.0], jnp.float32)
    rq = jnp.asarray(rng.normal(size=mjm.nq), jnp.float32)
    rv = jnp.asarray(rng.normal(size=mjm.nv), jnp.float32)
    vg = jax.jit(jax.vmap(jax.value_and_grad(_proj_loss, argnums=(0, 1, 3)),
                          in_axes=(0, 0, 0, 0, 0, 0, None, None)))

    def run(vx, graphs):
        monkeypatch.setattr(gj, 'USE_GRAPHS', graphs)
        batch = [jnp.broadcast_to(x, (B,) + x.shape) for x in _inputs(mjd, vx)]
        val, grads = vg(*batch, friction, added_mass, rq, rv)
        return np.asarray(val), [np.asarray(g) for g in grads]

    for vx in (0.5, 0.0):
        val_g, grads_g = run(vx, True)
        val_l, grads_l = run(vx, False)
        print(f'\nvx={vx}: loss diff {np.abs(val_g - val_l).max():.1e}', end='')
        np.testing.assert_allclose(val_g, val_l, atol=1e-3)
        for name, a, b in zip(('qpos', 'qvel', 'ctrl'), grads_g, grads_l):
            rel = np.abs(a - b).max() / np.abs(b).max()
            print(f'  d/d{name} {rel:.1e}', end='')
            assert rel < 5e-3, f'vx={vx} d/d{name}'
    assert ('fwd', B, 50) in gj._GRAPHS and ('bwd', B, 50) in gj._GRAPHS
    gj.check_overflow_all()

from mujoco import mjx
from envs.register_g1_29dof import FOOT_GEOMS, PELVIS_BODY

def _mjx_sys_data(mjm, mjd, added_mass=0.0):
    # same randomization as DiffG1._build_dr_sys (friction doesn't affect the rebuilt fields)
    sys = mjx.put_model(mjm)
    pelvis = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY)
    sys = sys.replace(body_mass=sys.body_mass.at[pelvis].add(added_mass))
    return sys, mjx.put_data(mjm, mjd)

REBUILT = ('xpos', 'xquat', 'cvel', 'subtree_com', 'geom_xpos', 'actuator_force')

@pytest.mark.parametrize('added_mass', [0.0, 3.0])
def test_pipeline_step_fields_match_forward(keyframe, added_mass):
    # the rebuilt fields must equal a full mjx.forward on the stepped state
    mjm, mjd = keyframe
    sys, data = _mjx_sys_data(mjm, mjd, added_mass)
    data = data.replace(qvel=data.qvel.at[0].set(0.5))
    mu = jnp.float32(_base_friction(mjm))
    out = jax.jit(gj.warp_pipeline_step)(sys, data, data.ctrl, mu, jnp.float32(added_mass))
    ref = jax.jit(mjx.forward)(sys, out)
    assert not np.allclose(np.asarray(out.qpos), np.asarray(data.qpos)), 'state did not advance'
    for f in REBUILT:
        a, b = np.asarray(getattr(out, f)), np.asarray(getattr(ref, f))
        print(f'\n+{added_mass} kg {f}: max diff {np.abs(a - b).max():.1e}')
        np.testing.assert_allclose(a, b, atol=1e-4, err_msg=f)

def test_pipeline_step_uses_randomized_mass(keyframe):
    # subtree_com[0] (whole-robot CoM) must be weighted with the +3 kg pelvis from sys
    mjm, mjd = keyframe
    sys, data = _mjx_sys_data(mjm, mjd, 3.0)
    mu = jnp.float32(_base_friction(mjm))
    out = jax.jit(gj.warp_pipeline_step)(sys, data, data.ctrl, mu, jnp.float32(3.0))
    xipos = np.asarray(out.xipos)
    m_dr, m_base = np.asarray(sys.body_mass), mjm.body_mass
    com_dr = (m_dr[:, None] * xipos).sum(0) / m_dr.sum()
    com_base = (m_base[:, None] * xipos).sum(0) / m_base.sum()
    got = np.asarray(out.subtree_com[0])
    print(f'\nCoM shift from +3 kg: {np.linalg.norm(com_dr - com_base):.1e} m, '
          f'error vs randomized {np.linalg.norm(got - com_dr):.1e} m')
    np.testing.assert_allclose(got, com_dr, atol=1e-5)
    assert np.linalg.norm(got - com_base) > 10 * np.linalg.norm(got - com_dr)

def test_pipeline_step_grad_reaches_warp(keyframe):
    # reward on a rebuilt field -> mjx.kinematics -> ct_qpos -> _bwd_impl -> d/dctrl
    mjm, mjd = keyframe
    sys, data = _mjx_sys_data(mjm, mjd)
    mu = jnp.float32(_base_friction(mjm))
    feet = jnp.array([mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOMS])

    def foot_height(ctrl):
        out = gj.warp_pipeline_step(sys, data, ctrl, mu, jnp.float32(0.0))
        return out.geom_xpos[feet, 2].sum()

    def torque(ctrl):   # actuator_force depends on ctrl directly and through the state
        out = gj.warp_pipeline_step(sys, data, ctrl, mu, jnp.float32(0.0))
        return jnp.square(out.actuator_force).sum()

    for name, f in (('foot_height', foot_height), ('torque', torque)):
        g = jax.jit(jax.grad(f))(data.ctrl)
        print(f'\n{name}: |d/dctrl| = {float(jnp.linalg.norm(g)):.3e}')
        assert bool(jnp.isfinite(g).all()) and float(jnp.linalg.norm(g)) > 0, name
