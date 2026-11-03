"""
Measures d L / d a, where a is an open-loop action sequence of H control steps
applied the way the G1 env does it (ctrl = default_pose + action_scale * a,
held for 10 physics steps), and L is a SHAC-like tracking loss averaged over
the H control steps. That is the gradient a SHAC policy update flows through.

Finite differences are central differences along K random directions shared by
all horizons (later actions cannot change earlier losses, so one rollout of
H_max control steps gives every shorter horizon). The reported cosine between
an analytic gradient g and FD is the cosine between [g . u_k]_k and [FD_k]_k,
a random projection of cos(g, g_fd) with error ~1/sqrt(K). Everything is
float32, so FD is reported at two step sizes; cos(FD eps1, FD eps2) says how far
to trust it at a given horizon.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')  # share the GPU with Warp

import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

from envs.g1_model import build_g1_mj_model

KEYFRAME = 'knees_bent'
SUBSTEPS = 10          # physics steps per control step (env default)
ACTION_SCALE = 0.5     # env default

# stage loss: (vx - v*)^2 + vy^2 + W_Z (z - z0)^2 + W_W (wx^2 + wy^2)
V_TARGET = 0.5
W_Z = 10.0
W_W = 0.1


def stage_loss(qpos, qvel, z0):
  return ((qvel[..., 0] - V_TARGET) ** 2 + qvel[..., 1] ** 2
          + W_Z * (qpos[..., 2] - z0) ** 2
          + W_W * (qvel[..., 3] ** 2 + qvel[..., 4] ** 2))


def stage_loss_grad(qpos, qvel, z0):
  #(dl/dqpos, dl/dqvel) of stage_loss, same shapes as qpos, qvel
  gq = np.zeros_like(qpos)
  gv = np.zeros_like(qvel)
  gv[..., 0] = 2.0 * (qvel[..., 0] - V_TARGET)
  gv[..., 1] = 2.0 * qvel[..., 1]
  gq[..., 2] = 2.0 * W_Z * (qpos[..., 2] - z0)
  gv[..., 3] = 2.0 * W_W * qvel[..., 3]
  gv[..., 4] = 2.0 * W_W * qvel[..., 4]
  return gq, gv


def cosine(a, b):
  na, nb = np.linalg.norm(a), np.linalg.norm(b)
  if na == 0.0 or nb == 0.0 or not np.isfinite(na + nb):
    return np.nan
  return float(np.dot(a, b) / (na * nb))


# ----------------------------------------------------------------------------
# initial states and nominal action sequences (CPU MuJoCo, float64)
# ----------------------------------------------------------------------------

def sample_problems(mjm, args, rng):
  #S (qpos, qvel, qacc_warmstart) states mid-gait-ish and an AR(1) action sequence of H_max steps continuing from each."""
  kid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME)
  default = mjm.key_qpos[kid][7:].copy()
  rho, std = args.action_rho, args.action_std
  mjd = mujoco.MjData(mjm)

  def ar1(a):
    return rho * a + np.sqrt(1.0 - rho ** 2) * std * rng.normal(size=mjm.nu)

  qpos0, qvel0, warm0, actions = [], [], [], []
  tries = 0
  while len(qpos0) < args.states:
    tries += 1
    if tries > 20 * args.states:
      raise RuntimeError('robot keeps falling during burn-in; lower --action-std')
    mujoco.mj_resetDataKeyframe(mjm, mjd, kid)
    mjd.qvel[:] += rng.normal(0.0, 0.05, mjm.nv)
    a = np.zeros(mjm.nu)
    fell = False
    for _ in range(args.burn_in):
      a = ar1(a)
      mjd.ctrl[:] = default + ACTION_SCALE * a
      for _ in range(SUBSTEPS):
        mujoco.mj_step(mjm, mjd)
      if mjd.qpos[2] < 0.6:
        fell = True
        break
    if fell:
      continue
    qpos0.append(mjd.qpos.copy())
    qvel0.append(mjd.qvel.copy())
    warm0.append(mjd.qacc_warmstart.copy())
    seq = []
    for _ in range(args.h_max):
      a = ar1(a)
      seq.append(a.copy())
    actions.append(np.stack(seq))
  return (np.stack(qpos0), np.stack(qvel0), np.stack(warm0),
          np.stack(actions), default)


def sample_policy_problems(mjm, args, rng):
  #S states from the deployed walking policy
  from deploy.controller import Controller
  from deploy.policy import NumpyPolicy
  from deploy.transport.mujoco_sim import MujocoTransport, SimConfig

  policy = NumpyPolicy.from_npz(args.policy)
  kid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME)
  default = mjm.key_qpos[kid][7:].copy()
  cmd = np.array([V_TARGET, 0.0, 0.0])

  qpos0, qvel0, warm0, actions = [], [], [], []
  seed = 0
  while len(qpos0) < args.states:
    seed += 1
    if seed > 20 * args.states:
      raise RuntimeError('policy keeps falling during burn-in')
    # no jitter, so a tick is exactly SUBSTEPS physics steps and the bench's
    # replay of the recorded ctrl reproduces the closed-loop trajectory
    tr = MujocoTransport(SimConfig(jitter_s=0.0, seed=int(rng.integers(1 << 30))), model=mjm)
    ctl = Controller(policy, speed_source='odometry')
    # stagger the burn-in so states land at different gait phases
    burn = args.burn_in + int(rng.integers(0, 50))
    state = seq = None
    for k in range(burn + args.h_max):
      if k == burn:
        state = (tr.data.qpos.copy(), tr.data.qvel.copy(), tr.data.qacc_warmstart.copy())
        seq = []
      out = ctl.step(tr.read(), cmd, loop_dt=tr.cfg.dt, true_v_body=tr.true_velocity_body())
      tr.send_position(out.q_target, ctl.table.kp, ctl.table.kd)
      tr.wait_tick()
      if seq is not None:
        seq.append((tr.data.ctrl - default) / ACTION_SCALE)
      if tr.fallen(0.5):
        break
    if seq is None or len(seq) < args.h_max:
      continue
    qpos0.append(state[0])
    qvel0.append(state[1])
    warm0.append(state[2])
    actions.append(np.stack(seq))
  return (np.stack(qpos0), np.stack(qvel0), np.stack(warm0),
          np.stack(actions), default)


def contact_switches(mjm, qpos0, qvel0, warm0, actions, default):
  mjd = mujoco.MjData(mjm)
  out = np.zeros(actions.shape[:2], dtype=int)
  max_ncon = max_nefc = 0

  def active():
    return {tuple(sorted((c.geom1, c.geom2))) for c in mjd.contact[:mjd.ncon] if c.dist < 0}

  for s in range(len(qpos0)):
    mujoco.mj_resetData(mjm, mjd)
    mjd.qpos[:], mjd.qvel[:], mjd.qacc_warmstart[:] = qpos0[s], qvel0[s], warm0[s]
    mujoco.mj_forward(mjm, mjd)
    prev, n = active(), 0
    for k in range(actions.shape[1]):
      mjd.ctrl[:] = default + ACTION_SCALE * actions[s, k]
      for _ in range(SUBSTEPS):
        mujoco.mj_step(mjm, mjd)
        cur = active()
        max_ncon, max_nefc = max(max_ncon, mjd.ncon), max(max_nefc, mjd.nefc)
        n += len(cur ^ prev)
        prev = cur
      out[s, k] = n
  return out, max_ncon, max_nefc


# ----------------------------------------------------------------------------
# MJWarp: analytic IFT gradient (PR #1535) and batched rollouts
# ----------------------------------------------------------------------------

class MjwBackend:
  name = 'mjwarp'

  def __init__(self, mjm, default, z0, args):
    import warp as wp
    import mujoco_warp as mjw
    # must precede the first kernel launch: swaps in the adjoint-safe sqrt and
    # registers the step backward hook.
    mjw.enable_grad()
    self.wp, self.mjw = wp, mjw
    self.mjm, self.z0, self.args = mjm, z0, args
    self.m = mjw.put_model(mjm)
    # the XML's iterations=3 hits the solver/linesearch limits on most steps;
    # the per-step print drowns the table. Convergence is what --iterations is for.
    self.m.opt.warn_overflow &= ~int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
    self.default = default.astype(np.float32)
    self._data = {}

    @wp.kernel
    def add_loss_adjoint(adj: wp.array3d(dtype=float), k: int, carry: wp.array2d(dtype=float)):
      w, i = wp.tid()
      carry[w, i] += adj[k, w, i]

    @wp.kernel
    def accum_action_grad(ctrl_grad: wp.array2d(dtype=float), k: int, scale: float,
                          g: wp.array3d(dtype=float)):
      w, i = wp.tid()
      g[w, k, i] += scale * ctrl_grad[w, i]

    self._add_loss_adjoint = add_loss_adjoint
    self._accum_action_grad = accum_action_grad

  def _make_data(self, nworld):
    kw = dict(nworld=nworld)
    if self.args.nconmax:
      kw['nconmax'] = self.args.nconmax
    if self.args.njmax:
      kw['njmax'] = self.args.njmax
    return self.mjw.put_data(self.mjm, mujoco.MjData(self.mjm), **kw)

  def _data_pair(self, nworld):
    if nworld not in self._data:
      self._data[nworld] = (self._make_data(nworld), self._make_data(nworld))
    return self._data[nworld]

  def _load(self, d, qpos, qvel, warm):
    d.qpos.assign(qpos.astype(np.float32))
    d.qvel.assign(qvel.astype(np.float32))
    d.qacc_warmstart.assign(warm.astype(np.float32))
    d.time.zero_()

  def _check_overflow(self, d):
    # the counts saturate at capacity, so reaching it means rows were dropped
    # and the physics (and every gradient) is wrong.
    nacon = int(d.nacon.numpy()[0])
    nefc = int(d.nefc.numpy().max())
    if nacon >= d.naconmax or nefc >= d.njmax:
      raise RuntimeError(f'MJWarp buffer full: nacon {nacon}/{d.naconmax}, nefc {nefc}/{d.njmax}; '
                         'raise --nconmax/--njmax')

  def rollout_losses(self, qpos, qvel, warm, actions):
    #(B, H) per-control-step stage losses for B rollouts, run in chunks
    wp = self.wp
    B, H, _ = actions.shape
    nw = min(self.args.fd_batch, B)
    d, _ = self._data_pair(nw)
    out = np.zeros((B, H))
    for lo in range(0, B, nw):
      idx = np.arange(lo, lo + nw) % B  # pad the last chunk with repeats
      self._load(d, qpos[idx], qvel[idx], warm[idx])
      ctrl = self.default + ACTION_SCALE * actions[idx].astype(np.float32)
      for k in range(H):
        d.ctrl.assign(ctrl[:, k])
        for _ in range(SUBSTEPS):
          self.mjw.step(self.m, d)
        losses = stage_loss(d.qpos.numpy().astype(np.float64), d.qvel.numpy().astype(np.float64), self.z0)
        n = min(nw, B - lo)
        out[lo:lo + n, k] = losses[:n]
      self._check_overflow(d)
    wp.synchronize()
    return out

  def grad(self, qpos, qvel, warm, actions, H):
    #(losses (S,), dL_H/da (S, H, nu), backward seconds)
    wp, mjw = self.wp, self.mjw
    S, nu = len(qpos), self.mjm.nu
    nq, nv = self.mjm.nq, self.mjm.nv
    N = H * SUBSTEPS
    d, d_out = self._data_pair(S)
    for arr in (d.qpos, d.qvel, d.ctrl):
      arr.requires_grad = True

    ck_qpos = wp.empty((N + 1, S, nq), dtype=float)
    ck_qvel = wp.empty((N + 1, S, nv), dtype=float)
    ck_warm = wp.empty((N + 1, S, nv), dtype=float)
    ck_time = wp.empty((N + 1, S), dtype=float)
    ctrl_seq = wp.array((self.default + ACTION_SCALE * actions[:, :H].astype(np.float32)).transpose(1, 0, 2))

    # forward, checkpointing the step inputs
    self._load(d, qpos, qvel, warm)
    for j in range(N):
      if j % SUBSTEPS == 0:
        wp.copy(d.ctrl, ctrl_seq[j // SUBSTEPS])
      wp.copy(ck_qpos[j], d.qpos)
      wp.copy(ck_qvel[j], d.qvel)
      wp.copy(ck_warm[j], d.qacc_warmstart)
      wp.copy(ck_time[j], d.time)
      mjw.step(self.m, d)
    wp.copy(ck_qpos[N], d.qpos)
    wp.copy(ck_qvel[N], d.qvel)
    self._check_overflow(d)

    hq = ck_qpos.numpy()[SUBSTEPS::SUBSTEPS].astype(np.float64)  # (H, S, nq)
    hv = ck_qvel.numpy()[SUBSTEPS::SUBSTEPS].astype(np.float64)
    losses = stage_loss(hq, hv, self.z0).mean(axis=0)
    gq, gv = stage_loss_grad(hq, hv, self.z0)
    adj_q = wp.array((gq / H).astype(np.float32))
    adj_v = wp.array((gv / H).astype(np.float32))

    # backward
    carry_q = wp.zeros((S, nq), dtype=float)
    carry_v = wp.zeros((S, nv), dtype=float)
    g = wp.zeros((S, H, nu), dtype=float)
    bc = mjw.create_backward_context(self.m, d)
    wp.synchronize()
    t0 = time.perf_counter()
    for j in reversed(range(N)):
      k = j // SUBSTEPS
      wp.copy(d.qpos, ck_qpos[j])
      wp.copy(d.qvel, ck_qvel[j])
      wp.copy(d.qacc_warmstart, ck_warm[j])
      wp.copy(d.time, ck_time[j])
      wp.copy(d.ctrl, ctrl_seq[k])
      if (j + 1) % SUBSTEPS == 0:  # d_out is the state at the end of control step k
        wp.launch(self._add_loss_adjoint, dim=(S, nq), inputs=[adj_q, k, carry_q])
        wp.launch(self._add_loss_adjoint, dim=(S, nv), inputs=[adj_v, k, carry_v])
      tape = wp.Tape()
      with mjw.backward_context(bc):
        with tape:
          mjw.step(self.m, d, d_out)
        tape.zero()
        wp.copy(d_out.qpos.grad, carry_q)
        wp.copy(d_out.qvel.grad, carry_v)
        tape.backward()
      wp.copy(carry_q, d.qpos.grad)
      wp.copy(carry_v, d.qvel.grad)
      wp.launch(self._accum_action_grad, dim=(S, nu), inputs=[d.ctrl.grad, k, ACTION_SCALE, g])
    wp.synchronize()
    return losses, g.numpy().astype(np.float64), time.perf_counter() - t0


# ----------------------------------------------------------------------------
# MJX: autodiff through the smooth_mjx-patched solver
# ----------------------------------------------------------------------------

class MjxBackend:
  name = 'mjx'

  def __init__(self, mjm, default, z0, args):
    import smooth_mjx
    if args.mjx_mode == 'st':
      smooth_mjx.enable(kappa=args.kappa, straight_through=True)
    elif args.mjx_mode == 'soft':
      smooth_mjx.enable(kappa=args.kappa, straight_through=False)
    else:  # hard contact switch, only make the Newton loop reverse-differentiable
      smooth_mjx._patch_newton_loop()
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    self.jax, self.args = jax, args
    mx = mjx.put_model(mjm)
    d0 = mjx.put_data(mjm, mujoco.MjData(mjm))
    default = jp.asarray(default, dtype=jp.float32)

    def rollout_losses(qpos0, qvel0, warm0, actions):
      d = d0.replace(qpos=qpos0, qvel=qvel0, qacc_warmstart=warm0)

      @jax.checkpoint
      def control_step(d, a):
        ctrl = default + ACTION_SCALE * a

        def physics_step(d, _):
          return mjx.step(mx, d.replace(ctrl=ctrl)), None

        d, _ = jax.lax.scan(physics_step, d, None, length=SUBSTEPS)
        return d, stage_loss(d.qpos, d.qvel, z0)

      _, losses = jax.lax.scan(control_step, d, actions)
      return losses

    def weighted_loss(qpos0, qvel0, warm0, actions, w):
      return jp.dot(w, rollout_losses(qpos0, qvel0, warm0, actions))

    # one compile for every horizon: L_H puts weight 1/H on the first H steps
    self._grad = jax.jit(jax.vmap(jax.value_and_grad(weighted_loss, argnums=3),
                                  in_axes=(0, 0, 0, 0, None)))
    self._rollout = jax.jit(jax.vmap(rollout_losses))

  def rollout_losses(self, qpos, qvel, warm, actions):
    B = len(actions)
    nb = min(self.args.fd_batch, B)
    out = np.zeros(actions.shape[:2])
    for lo in range(0, B, nb):
      idx = np.arange(lo, lo + nb) % B
      res = self._rollout(*(x[idx].astype(np.float32) for x in (qpos, qvel, warm, actions)))
      n = min(nb, B - lo)
      out[lo:lo + n] = np.asarray(res, dtype=np.float64)[:n]
    return out

  def grad(self, qpos, qvel, warm, actions, H):
    w = np.zeros(actions.shape[1], dtype=np.float32)
    w[:H] = 1.0 / H
    args = [x.astype(np.float32) for x in (qpos, qvel, warm, actions)] + [w]
    self._grad(*args)[1].block_until_ready()  # compile / warm up outside the timer
    t0 = time.perf_counter()
    losses, g = self._grad(*args)
    g.block_until_ready()
    dt = time.perf_counter() - t0
    return np.asarray(losses, np.float64), np.asarray(g, np.float64)[:, :H], dt


# ----------------------------------------------------------------------------

def finite_differences(backend, qpos, qvel, warm, actions, dirs, eps_list):
  # fd[e, s, k, h] = central difference of L_{h+1} along dirs[k], step eps_list[e]
  S, K = len(qpos), len(dirs)
  E = len(eps_list)
  # rows ordered (eps, sign, state, dir)
  eps = np.asarray(eps_list)[:, None, None, None, None, None]
  sign = np.array([1.0, -1.0])[None, :, None, None, None, None]
  pert = actions[None, None, :, None] + sign * eps * dirs[None, None, None]
  rep = lambda x: np.broadcast_to(x[None, None, :, None], (E, 2, S, K) + x.shape[1:]).reshape(-1, x.shape[-1])
  L = backend.rollout_losses(rep(qpos), rep(qvel), rep(warm), pert.reshape(-1, *actions.shape[1:]))
  L = L.reshape(E, 2, S, K, -1)
  L_h = np.cumsum(L, axis=-1) / np.arange(1, L.shape[-1] + 1)  # L_H for every horizon at once
  return (L_h[:, 0] - L_h[:, 1]) / (2.0 * np.asarray(eps_list)[:, None, None, None])


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--backends', default='mjwarp,mjx')
  p.add_argument('--horizons', default='1,2,4,8,16,32', help='control steps (x10 physics steps)')
  p.add_argument('--states', type=int, default=8)
  p.add_argument('--dirs', type=int, default=128, help='random FD directions')
  p.add_argument('--eps', default='1e-3,1e-2', help='FD steps in action units (x0.5 rad)')
  p.add_argument('--states-from', choices=('random', 'policy'), default='random',
                 help='random: AR(1) actions from the keyframe; policy: --policy walking closed loop at vx=0.5')
  p.add_argument('--policy', default='deploy/artifacts/v14.npz')
  p.add_argument('--burn-in', type=int, default=None,
                 help='control steps before t=0 (default 50 random, 150 policy)')
  p.add_argument('--action-std', type=float, default=0.3)
  p.add_argument('--action-rho', type=float, default=0.8, help='AR(1) action correlation')
  p.add_argument('--iterations', type=int, default=None, help='override opt.iterations (XML: 3)')
  p.add_argument('--ls-iterations', type=int, default=None, help='override opt.ls_iterations (XML: 5)')
  p.add_argument('--mjx-mode', choices=('st', 'soft', 'hard'), default='st',
                 help='st: hard forward, sigmoid backward (train_g1.py); soft: sigmoid both; hard: no smoothing')
  p.add_argument('--kappa', type=float, default=300.0)
  p.add_argument('--fd-batch', type=int, default=1024, help='worlds per FD chunk')
  p.add_argument('--nconmax', type=int, default=None, help='per world (default: 2x peak in nominal rollouts)')
  p.add_argument('--njmax', type=int, default=None, help='per world (default: 2x peak in nominal rollouts)')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', default=None, help='json results (default outputs/grad_bench_<tag>.json)')
  args = p.parse_args()

  horizons = sorted(int(h) for h in args.horizons.split(','))
  args.h_max = horizons[-1]
  eps_list = [float(e) for e in args.eps.split(',')]
  backends = args.backends.split(',')

  mjm = build_g1_mj_model()
  if args.iterations is not None:
    mjm.opt.iterations = args.iterations
  if args.ls_iterations is not None:
    mjm.opt.ls_iterations = args.ls_iterations
  kid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME)
  z0 = float(mjm.key_qpos[kid][2])

  rng = np.random.default_rng(args.seed)
  if args.burn_in is None:
    args.burn_in = 150 if args.states_from == 'policy' else 50
  sampler = sample_policy_problems if args.states_from == 'policy' else sample_problems
  qpos, qvel, warm, actions, default = sampler(mjm, args, rng)
  switches, max_ncon, max_nefc = contact_switches(mjm, qpos, qvel, warm, actions, default)
  # MJWarp allocates per-world buffers up front (default njmax=64, below the
  # G1's ~61 rows at rest); size them from the nominal rollouts with 2x headroom
  # for FD perturbations.
  args.nconmax = args.nconmax or 2 * max_ncon
  args.njmax = args.njmax or -(-2 * max_nefc // 32) * 32
  dirs = rng.normal(size=(args.dirs, args.h_max, mjm.nu))
  dirs /= np.linalg.norm(dirs.reshape(args.dirs, -1), axis=1)[:, None, None]

  print(f'G1 nv={mjm.nv} nu={mjm.nu} | iterations={mjm.opt.iterations} ls_iterations={mjm.opt.ls_iterations} '
        f'tol={mjm.opt.tolerance:g} dt={mjm.opt.timestep:g} | peak ncon={max_ncon} nefc={max_nefc} -> '
        f'nconmax={args.nconmax} njmax={args.njmax} | {args.states} {args.states_from} states, '
        f'{args.dirs} FD dirs, eps={eps_list}')

  results = {'args': vars(args), 'horizons': horizons, 'eps': eps_list,
             'contact_switches': switches.tolist(), 'backends': {}}
  grads, fds = {}, {}
  for name in backends:
    backend = {'mjwarp': MjwBackend, 'mjx': MjxBackend}[name](mjm, default, z0, args)
    label = name if name == 'mjwarp' else f'mjx[{args.mjx_mode}' + (']' if args.mjx_mode == 'hard' else f' k={args.kappa:g}]')
    print(f'\n== {label}: finite differences ({len(eps_list) * 2 * args.states * args.dirs} rollouts of {args.h_max} steps)')
    t0 = time.perf_counter()
    fd = finite_differences(backend, qpos, qvel, warm, actions, dirs, eps_list)
    print(f'   {time.perf_counter() - t0:.1f}s')
    fds[name] = fd
    backend.grad(qpos, qvel, warm, actions, horizons[0])  # compile the backward outside the timer
    rows = []
    grads[name] = {}
    for H in horizons:
      losses, g, t_bwd = backend.grad(qpos, qvel, warm, actions, H)
      grads[name][H] = g
      proj = np.einsum('shu,khu->sk', g, dirs[:, :H])  # (S, K)
      fd_h = fd[..., H - 1]                              # (E, S, K)
      row = {
          'H': H,
          'loss': losses.tolist(),
          'grad_norm': np.linalg.norm(g.reshape(len(g), -1), axis=1).tolist(),
          'nan_frac': float(np.mean(~np.isfinite(g))),
          'cos_fd': [[cosine(proj[s], fd_h[e, s]) for s in range(len(g))] for e in range(len(eps_list))],
          'cos_fd_self': [cosine(fd_h[0, s], fd_h[-1, s]) for s in range(len(g))],
          'backward_s': t_bwd,
      }
      rows.append(row)
      fmt = lambda v: f'{np.nanmedian(v):+.3f}'
      cos_cols = '  '.join(f'cos(g,FD {e:g})={fmt(row["cos_fd"][i])}' for i, e in enumerate(eps_list))
      print(f'   H={H:3d} sw={np.median(switches[:, H - 1]):4.0f}  L={np.mean(losses):.4f}  '
            f'|g|={np.median(row["grad_norm"]):.3e}  {cos_cols}  cos(FD,FD)={fmt(row["cos_fd_self"])}  '
            f'nan={row["nan_frac"]:.2f}  bwd={t_bwd:.2f}s')
    results['backends'][label] = rows

  if len(grads) == 2:
    print('\n== mjwarp IFT vs MJX (full-gradient cosine, median over states)')
    a, b = grads['mjwarp'], grads['mjx']
    results['cross'] = []
    for H in horizons:
      cs = [cosine(a[H][s].ravel(), b[H][s].ravel()) for s in range(args.states)]
      # each analytic gradient against the OTHER backend's FD too: both simulate
      # the same physics, so the less noisy FD is the fairer reference for both
      ref = {}
      for g_name, g in (('mjwarp', a[H]), ('mjx', b[H])):
        proj = np.einsum('shu,khu->sk', g, dirs[:, :H])
        for fd_name, fd in fds.items():
          ref[f'{g_name}_vs_fd_{fd_name}'] = [[cosine(proj[s], fd[e, s, :, H - 1]) for s in range(args.states)]
                                              for e in range(len(eps_list))]
      results['cross'].append({'H': H, 'cos': cs, **ref})
      fmt = lambda v: f'{np.nanmedian(v[-1]):+.3f}'
      print(f'   H={H:3d}  cos={np.nanmedian(cs):+.3f}  [min {np.nanmin(cs):+.3f}, max {np.nanmax(cs):+.3f}]  '
            f'| vs FD(mjwarp, eps {eps_list[-1]:g}): ift={fmt(ref["mjwarp_vs_fd_mjwarp"])} mjx={fmt(ref["mjx_vs_fd_mjwarp"])}  '
            f'| vs FD(mjx): ift={fmt(ref["mjwarp_vs_fd_mjx"])} mjx={fmt(ref["mjx_vs_fd_mjx"])}')

  tag = '_'.join(backends) + (f'_{args.mjx_mode}' if 'mjx' in backends else '') + \
        (f'_it{args.iterations}' if args.iterations else '') + \
        ('_policy' if args.states_from == 'policy' else '')
  out = pathlib.Path(args.out or f'outputs/grad_bench_{tag}.json')
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(results, indent=1))
  print(f'\nsaved {out}')


if __name__ == '__main__':
  main()
