"""DiffG1(backend='mjwarp') end to end: same env, physics swapped.

    uv run --group warp pytest envs/tests/test_diffg1_backend.py -q -s
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

from envs import g1_warp_jax  # noqa: F401  first: mjw.enable_grad() before any kernel

import smooth_mjx
smooth_mjx.enable(kappa=300, straight_through=True)   # as train_g1.py: MJX reverse mode needs it, forward unchanged

import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax import envs
from mujoco import mjx

from envs import register_g1_29dof  # noqa: F401  registers g1_29dof

B = 8
KW = dict(use_domain_randomization=True, kick_prob=0.3, obs_noise=0.0, action_delay_prob=0.0)


@pytest.fixture(scope='module')
def envs_by_backend():
  return {b: envs.get_environment('g1_29dof', backend=b, **KW) for b in ('mjx', 'mjwarp')}


def _rollout(env, state, actions):
  def body(s, a):
    s = env.step(s, a)
    return s, s.reward
  return jax.lax.scan(body, state, actions)


def test_bad_backend_raises():
  with pytest.raises(ValueError):
    envs.get_environment('g1_29dof', backend='warp')


def _refresh(sys, d):
  #recompute the derived fields from the final qpos/qvel. mjx.step leaves them one substep stale TODO future me
  for f in (mjx.kinematics, mjx.com_pos, mjx.transmission, mjx.fwd_velocity):
    d = f(sys, d)
  return d


def test_steps_match_mjx(envs_by_backend):
  keys = jax.random.split(jax.random.PRNGKey(0), B)
  actions = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (B, 29))
  out = {}
  for name, env in envs_by_backend.items():
    s0 = jax.jit(jax.vmap(env.reset))(keys)
    out[name] = jax.jit(jax.vmap(env.step))(s0, actions)
  w = out['mjwarp']
  m = jax.jit(jax.vmap(lambda d: _refresh(envs_by_backend['mjx'].sys, d)))(out['mjx'].pipeline_state)
  pen = np.asarray(s0.pipeline_state.contact.dist.min(axis=1))
  ok = pen > -5e-3
  diff = lambda f: np.abs(np.asarray(getattr(m, f) - getattr(w.pipeline_state, f))).reshape(B, -1).max(1)
  dq, dv, dcv, dg = diff('qpos'), diff('qvel'), diff('cvel'), diff('geom_xpos')
  print(f'\n  start penetration {pen.round(3)}\n  dqpos {dq.round(4)}\n  dqvel {dv.round(3)}'
        f'\n  dcvel {dcv.round(3)}\n  dgeom_xpos {dg.round(4)}')
  assert np.isfinite(np.asarray(w.obs)).all() and np.isfinite(np.asarray(w.reward)).all()
  assert ok.sum() >= 2, 'too few clean starts in this batch, change the seed'
  assert (dq[ok] < 1e-3).all() and (dv[ok] < 5e-2).all()
  assert (dcv[ok] < 5e-2).all() and (dg[ok] < 1e-3).all()


@pytest.mark.parametrize('H', [8])
def test_rollout_grad(envs_by_backend, H):
  #d(sum reward)/d(actions) through jit(vmap(grad(scan(env.step))))
  keys = jax.random.split(jax.random.PRNGKey(2), B)
  actions = 0.1 * jax.random.normal(jax.random.PRNGKey(3), (B, H, 29))
  grads = {}
  for name, env in envs_by_backend.items():
    s0 = jax.jit(jax.vmap(env.reset))(keys)
    loss = lambda s, a: _rollout(env, s, a)[1].sum()
    g = jax.jit(jax.vmap(jax.value_and_grad(loss, argnums=1)))(s0, actions)[1]
    grads[name] = np.asarray(g, np.float64)
  g_w, g_m = grads['mjwarp'], grads['mjx']
  assert np.isfinite(g_w).all() and np.abs(g_w).max() > 0
  cos = [float(g_w[b].ravel() @ g_m[b].ravel()
               / (np.linalg.norm(g_w[b]) * np.linalg.norm(g_m[b]) + 1e-12)) for b in range(B)]
  print(f'\n  H={H}: |g| mjwarp {np.linalg.norm(g_w.reshape(B, -1), axis=1).round(2)}  '
        f'mjx {np.linalg.norm(g_m.reshape(B, -1), axis=1).round(2)}  cos {np.round(cos, 3)}')
