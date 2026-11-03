"""
G1 on MuJoCo Warp
- model/data loader
- per-world domain randomization matching (foot friction, base mass)
- buffer overflow check.
replaces: mjx.put_model / mjx.make_data / _build_dr_sys plumbing
"""
import dataclasses

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjw

from .g1_model import build_g1_mj_model
from .register_g1_29dof import FOOT_PAIRS, KEYFRAME, PELVIS_BODY

# MJWarp allocates contact/constraint buffers per world up front (its default
# njmax=64 is below the G1's ~61 rows standing). Peaks measured in
# bench_grad_g1.py: walking ncon 6 / nefc 53, random flailing ncon 8 / nefc 63.
# 3x headroom for kicks and falls
NCONMAX = 24
NJMAX = 192


@dataclasses.dataclass
class G1Warp:
  mjm: mujoco.MjModel
  m: mjw.Model
  d: mjw.Data
  nworld: int
  foot_pair_ids: wp.array        # (2,) int
  base_body_id: int
  base_pair_friction: wp.array   # (npair,) vec5, unrandomized
  base_body_mass: wp.array       # (nbody,) float, unrandomized
  peak: wp.array                 # (2,) int: max nacon, max nefc seen by record_peak


def make_g1_warp(nworld: int, *, iterations: int | None = None,
                 ls_iterations: int | None = None, nconmax: int = NCONMAX,
                 njmax: int = NJMAX, domain_randomization: bool = True) -> G1Warp:
  """model + data for nworld envs, reset to the knees_bent keyframe."""
  mjw.enable_grad()

  mjm = build_g1_mj_model()
  if iterations is not None:
    mjm.opt.iterations = iterations
  if ls_iterations is not None:
    mjm.opt.ls_iterations = ls_iterations

  # per-world copies of only the fields DiffG1 randomizes; everything else
  # stays shared (leading dim 1, kernels index worldid % shape[0]).
  batch = {'pair_friction': nworld, 'body_mass': nworld} if domain_randomization else {}
  m = mjw.put_model(mjm, batch_sizes=batch)
  # the XML's iterations=3 hits the solver limits on most steps; the warning
  # is printed per step. Unconverged solves are a modelling choice, not a bug.
  m.opt.warn_overflow &= ~int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)

  mjd = mujoco.MjData(mjm)
  kid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME)
  mujoco.mj_resetDataKeyframe(mjm, mjd, kid)
  mujoco.mj_forward(mjm, mjd)
  d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)

  pair_ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_PAIR, n) for n in FOOT_PAIRS]
  base_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY)
  if min(pair_ids) < 0 or base_id < 0:
    raise ValueError(f'G1 model is missing {FOOT_PAIRS} or {PELVIS_BODY!r}')

  return G1Warp(
      mjm=mjm, m=m, d=d, nworld=nworld,
      foot_pair_ids=wp.array(pair_ids, dtype=int),
      base_body_id=base_id,
      base_pair_friction=wp.array(mjm.pair_friction.astype(np.float32), dtype=wp.types.vector(5, float)),
      base_body_mass=wp.array(mjm.body_mass.astype(np.float32), dtype=float),
      peak=wp.zeros(2, dtype=int),
  )


@wp.kernel
def _set_dr(
    foot_friction: wp.array(dtype=float),
    added_mass: wp.array(dtype=float),
    foot_pair_ids: wp.array(dtype=int),
    base_body_id: int,
    base_pair_friction: wp.array(dtype=wp.types.vector(5, float)),
    base_body_mass: wp.array(dtype=float),
    pair_friction: wp.array2d(dtype=wp.types.vector(5, float)),
    body_mass: wp.array2d(dtype=float),
):
  worldid, i = wp.tid()
  # columns 0:2 of pair_friction are the two tangential (sliding) coefficients,
  # same as DiffG1._build_dr_sys.
  pairid = foot_pair_ids[i]
  f = base_pair_friction[pairid]
  f[0] = foot_friction[worldid]
  f[1] = foot_friction[worldid]
  pair_friction[worldid, pairid] = f
  if i == 0:
    body_mass[worldid, base_body_id] = base_body_mass[base_body_id] + added_mass[worldid]


def set_domain_randomization(g1: G1Warp, foot_friction: wp.array, added_mass: wp.array):
  """write per-world foot friction and added base mass, both (nworld,) float.

  Like DiffG1, only body_mass changes: derived constants (subtree mass,
  invweight) are not recomputed, matching MJX's behaviour in _build_dr_sys.
  """
  if g1.m.body_mass.shape[0] != g1.nworld:
    raise RuntimeError('built with domain_randomization=False')
  wp.launch(
      _set_dr, dim=(g1.nworld, len(FOOT_PAIRS)),
      inputs=[foot_friction, added_mass, g1.foot_pair_ids, g1.base_body_id,
              g1.base_pair_friction, g1.base_body_mass],
      outputs=[g1.m.pair_friction, g1.m.body_mass],
  )


@wp.kernel
def _record_peak(nacon: wp.array(dtype=int), nefc: wp.array(dtype=int), peak: wp.array(dtype=int)):
  worldid = wp.tid()
  wp.atomic_max(peak, 1, nefc[worldid])
  if worldid == 0:
    wp.atomic_max(peak, 0, nacon[0])


def record_peak(g1: G1Warp, d: mjw.Data | None = None):
  #no host sync,
  d = d or g1.d
  wp.launch(_record_peak, dim=g1.nworld, inputs=[d.nacon, d.nefc], outputs=[g1.peak])


def check_overflow(g1: G1Warp, d: mjw.Data | None = None):
  #host syncs
  d = d or g1.d
  record_peak(g1, d)
  nacon, nefc = (int(x) for x in g1.peak.numpy())
  if nacon >= d.naconmax or nefc >= d.njmax:
    raise RuntimeError(
        f'MJWarp buffer overflow: nacon {nacon}/{d.naconmax}, nefc {nefc}/{d.njmax}; '
        'raise nconmax/njmax in make_g1_warp')
