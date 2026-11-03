"""
measure cost of: jit(vmap(value_and_grad(sum of rewards over an
H-step scan of env.step))) w.r.t. the actions, for MJX and MJWarp.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')   # share the GPU with Warp

import argparse
import json
import pathlib
import time

from envs import g1_warp_jax as gj   # first: mjw.enable_grad() before any kernel

import smooth_mjx
smooth_mjx.enable(kappa=300, straight_through=True)   # as train_g1.py

import jax
import jax.numpy as jp
import numpy as np
from brax import envs

from envs import register_g1_29dof  # noqa: F401  registers g1_29dof

ENV_KW = dict(termination_height=0.5, physics_steps_per_control_step=10, action_scale=0.5,
              use_domain_randomization=True, kick_prob=0.3, obs_noise=1.0, action_delay_prob=0.5)


def timed(f, *args, reps):
  jax.block_until_ready(f(*args))            # compile + warm up (and CUDA graph capture)
  ts = []
  for _ in range(reps):
    t = time.perf_counter()
    jax.block_until_ready(f(*args))
    ts.append(time.perf_counter() - t)
  return float(np.median(ts))


def bench(env, B, H, reps, remat):
  s0 = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), B))
  acts = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (B, H, env.action_size))

  step = jax.checkpoint(env.step) if remat else env.step

  def ret(s, a):
    def body(s, a):
      s = step(s, a)
      return s, s.reward
    return jax.lax.scan(body, s, a)[1].sum()

  fwd = jax.jit(jax.vmap(ret))
  grad = jax.jit(jax.vmap(jax.value_and_grad(ret, argnums=1)))
  t_f = timed(fwd, s0, acts, reps=reps)
  t_g = timed(grad, s0, acts, reps=reps)
  g = np.asarray(grad(s0, acts)[1])
  return dict(fwd_ms=1e3 * t_f, grad_ms=1e3 * t_g, sps=B * H / t_g, finite=bool(np.isfinite(g).all()))


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--backends', default='mjx,mjwarp')
  p.add_argument('--horizons', default='12,32')
  p.add_argument('--iterations', default='3,50', help='mjwarp solver iterations (mjx: XML setting)')
  p.add_argument('--envs', type=int, default=64)
  p.add_argument('--reps', type=int, default=5)
  p.add_argument('--out', default='outputs/throughput_g1.json')
  args = p.parse_args()
  horizons = [int(h) for h in args.horizons.split(',')]

  rows = []
  print(f'{"backend":>14} {"H":>3} {"fwd ms":>8} {"fwd+grad ms":>12} {"env-steps/s":>12}')
  for backend in args.backends.split(','):
    env = envs.get_environment('g1_29dof', backend=backend, **ENV_KW)
    its = [int(i) for i in args.iterations.split(',')] if backend == 'mjwarp' else [None]
    for it in its:
      gj.ITERATIONS = None if it == 3 else it    # None = the XML's 3
      tag = backend if it is None else f'{backend} it{it}'
      for H in horizons:
        r = bench(env, args.envs, H, args.reps, remat=backend == 'mjx')
        rows.append(dict(backend=backend, iterations=it, H=H, B=args.envs, **r))
        print(f'{tag:>14} {H:3d} {r["fwd_ms"]:8.0f} {r["grad_ms"]:12.0f} {r["sps"]:12.0f}'
              + ('' if r['finite'] else '  NON-FINITE GRAD'))
  if 'mjwarp' in args.backends:
    gj.check_overflow_all()
  pathlib.Path(args.out).parent.mkdir(exist_ok=True)
  pathlib.Path(args.out).write_text(json.dumps(rows, indent=1))
  print(f'saved {args.out}')


if __name__ == '__main__':
  main()
