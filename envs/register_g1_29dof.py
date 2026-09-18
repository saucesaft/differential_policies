"""
unitree G1 29-DOF velocity tracking

obs (112-dim):
    linear base velocity (body frame)     3
    angular base velocity (body frame)    3
    projected gravity (pelvis frame)      3
    velocity command [vx*, vy*, wz*]      3
    joint positions                      29
    joint velocities                     29
    previous action                      29
    gait phase [sin, cos]                 2
    projected gravity (torso frame)       3
    foot positions rel. pelvis (body)     6
    CoM xy rel. feet midpoint (body)      2
    yotal                               112
"""
import warnings
from typing import Any, Dict

from brax import envs
from brax import math
from brax.base import Motion, Transform
from ml_collections import config_dict
from mujoco import mjx
import jax
from jax import numpy as jp
import mujoco
import numpy as np

from .g1_model import build_g1_mj_model
from .mjx_envs import State, MjxEnv
from jax_shac.utils.math_utils import axis_angle_to_quaternion

# body / geom / pair / keyframe names this env resolves at construction.
PELVIS_BODY = 'pelvis'
TORSO_BODY = 'torso_link'
FOOT_GEOMS = ('left_foot', 'right_foot')
FOOT_PAIRS = ('left_foot_floor', 'right_foot_floor')
KEYFRAME = 'knees_bent'
N_LEG_JOINTS = 12  # qpos[7:19]: left leg 6, right leg 6. The remaining 17 are
                   # waist (3) + arms/wrists (14), commanded but posture-penalised.


# nan_to_num with well-defined derivatives: where the input was NaN,
# the tangent/cotangent is zeroed.  Using custom_jvp so it works with
# both forward-mode (jacfwd, used for Jacobian diagnostics) and
# reverse-mode (grad, used for policy training) autodiff.
@jax.custom_jvp
def safe_nan_to_num(x):
    return jp.nan_to_num(x)

@safe_nan_to_num.defjvp
def _safe_nan_jvp(primals, tangents):
    x, = primals
    t, = tangents
    return jp.nan_to_num(x), jp.where(jp.isnan(x), 0.0, t)


def get_config():
  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            scales=config_dict.ConfigDict(
                dict(
                    # --- velocity tracking ---
                    lin_vel_tracking=1.0,    # exp(-||v_xy - v_xy*||² / 0.25)
                    ang_vel_tracking=0.75,   # exp(-(ωz - ωz*)² / 0.25)
                    # --- foot height tracking (cubic Bezier prescription) ---
                    foot_height=1.0,         # exp(-Σ_j (z_j - z*_j)² / 0.01), z* = rest at zero cmd
                    # --- velocity penalties ---
                    lin_vel_error=0.0,       # -vz²                 (off)
                    ang_vel_error=0.15,      # -||ωxy||²
                    # --- base stability ---
                    base_height=0.0,         # exp(-(z-z_home)²/0.1) (off)
                    fall_barrier=5.0,        # -((z_b - z)/(z_b - z_term))²
                    base_orientation=0.5,    # -||g_xy||²           (pelvis)
                    torso_orientation=2.0,   # -||up_torso - up*||²  (torso)
                    # --- humanoid posture ---
                    upper_body_posture=0.5,  # -||q_upper - q_upper*||²
                    # --- smoothness ---
                    # action_magnitude term they collapsed action_size from 0.71
                    # to 0.36 in a single epoch and flattened the policy gradient.
                    action_rate=0.05,        # -||a - a_prev||²
                    joint_acceleration=0.0,  # -||q̈||²              (off)
                    joint_torque=0.0,        # -||τ||²              (off)
                )
            ),
        )
    )
    return default_config

  default_config = config_dict.ConfigDict(
      dict(rewards=get_default_rewards_config(),))

  return default_config


class DiffG1(MjxEnv):
  """
  velocity commands are randomised in [-1,1] m/s (linear) and [-1,1] rad/s (yaw),
  resampled every 10-15 s
  """

  def __init__(
      self,
      action_scale: float = 0.5,
      termination_height: float = 0.5,
      barrier_height: float = 0.65,
      s_afilt_buf: float = 1,
      smooth_sigma_q: float = 0.0,
      smooth_sigma_v: float = 0.0,
      swing_height: float = 0.15,      # bezier foot target amplitude (m)
      cmd_deadband: float = 0.1,       # ||vel_cmd|| below this counts as no command
      gait_freq_min: float = 1.0,      # gait cycles/s at zero command
      gait_freq_max: float = 1.8,      # gait cycles/s at full command
      kick_prob: float = 0.3,          # per-opportunity chance of a base velocity kick
      reward_scales: dict = None,
      use_domain_randomization: bool = True,
      **kwargs,
  ):
    self.early_termination = kwargs.get('early_termination', True)

    mj_model = build_g1_mj_model()

    self.s_afilt_buf = s_afilt_buf
    if s_afilt_buf > 1:
      warnings.warn("s_afilt_buf > 1 gives undefined observations")

    # 10 x 0.002s = 0.02s control dt
    physics_steps_per_control_step = 10
    kwargs['physics_steps_per_control_step'] = kwargs.get(
        'physics_steps_per_control_step', physics_steps_per_control_step)
    kwargs.pop('early_termination', None)
    super().__init__(mj_model=mj_model, **kwargs)

    self.action_scale = action_scale
    self.termination_height = termination_height
    self.barrier_height = barrier_height
    self.smooth_sigma_q = smooth_sigma_q
    self.smooth_sigma_v = smooth_sigma_v
    self.swing_height = swing_height
    self.cmd_deadband = cmd_deadband
    self.gait_freq_min = gait_freq_min
    self.gait_freq_max = gait_freq_max

    self.n_joints = mj_model.nq - 7          # 29
    self.n_upper = self.n_joints - N_LEG_JOINTS  # 17

    # --- ids, all resolved by name ---
    def body_id(name):
      i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
      if i < 0:
        raise ValueError(f'body {name!r} not in the G1 model')
      return i

    def geom_id(name):
      i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
      if i < 0:
        raise ValueError(f'geom {name!r} not in the G1 model')
      return i

    def pair_id(name):
      i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_PAIR, name)
      if i < 0:
        raise ValueError(f'contact pair {name!r} not in the G1 model')
      return i

    self._base_body_id = body_id(PELVIS_BODY)
    self._torso_body_id = body_id(TORSO_BODY)
    # _pos_vel() drops the world body, so x/xd are indexed one lower than mj ids.
    self._base_x_idx = self._base_body_id - 1
    self._torso_x_idx = self._torso_body_id - 1
    self.foot_geom_ids = jp.array([geom_id(n) for n in FOOT_GEOMS])
    self._foot_pair_ids = jp.array([pair_id(n) for n in FOOT_PAIRS])

    # biped: the two feet are exactly out of phase.
    self.foot_phase_offsets = jp.array([0.0, jp.pi])

    kid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME)
    if kid < 0:
      raise ValueError(f'keyframe {KEYFRAME!r} not in the G1 model')
    self._init_q = mj_model.key_qpos[kid].copy()
    # this model's key_ctrl equals key_qpos[7:], so the nominal position targets
    # and the nominal joint angles are the same vector.
    self._default_ap_pose = mj_model.key_qpos[kid][7:].copy()
    self._default_upper_pose = jp.array(self._default_ap_pose[N_LEG_JOINTS:])
    # joint 0 is the free base; joints 1..29 map to qpos[7:] in order.
    self._jnt_lo = jp.array(mj_model.jnt_range[1:, 0])
    self._jnt_hi = jp.array(mj_model.jnt_range[1:, 1])
    self._home_base_h = float(mj_model.key_qpos[kid][2])   # 0.755
    # foot geom height with the feet planted in the keyframe (~0.007 m): the
    # stance value of the foot height target.
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, kid)
    mujoco.mj_forward(mj_model, mj_data)
    self._foot_rest_z = float(
        np.mean(mj_data.geom_xpos[np.array(self.foot_geom_ids), 2]))

    self.reward_config = get_config()
    if reward_scales:
      for k, v in reward_scales.items():
        self.reward_config.rewards.scales[k] = v

    # velocity command ranges and per-component "keep nonzero" probabilities,
    self._cmd_ranges = jp.array([[-1.0, 1.0], [-0.5, 0.5], [-1.0, 1.0]])
    self._cmd_keep_prob = jp.array([0.9, 0.25, 0.5])
    self._cmd_zero_prob = 0.1

    self._resample_every = int(round(1.0 / self.dt))
    self._resample_prob = 0.4

    # domain randomization
    # base values stored at init -- always randomize relative to these.
    self._use_dr = use_domain_randomization
    self._dr_mass_range = (-2.0, 2.0)
    self._dr_friction_range = (0.5, 1.25)
    # every _kick_every steps, apply a base velocity kick with probability
    # _kick_prob.
    self._kick_every = int(round(1.0 / self.dt))
    self._kick_prob = kick_prob

  # --------------------------------------------------------------------------
  # helpers
  # --------------------------------------------------------------------------

  def _to_body_frame(self, v_world: jax.Array, q_body: jax.Array) -> jax.Array:
    """rotate a world-frame vector into the robot body frame.

    q_body is the body-to-world quaternion (w, x, y, z).
    Inverse = (w, -x, -y, -z) for unit quaternions.
    """
    q_inv = jp.array([q_body[0], -q_body[1], -q_body[2], -q_body[3]])
    return math.rotate(v_world, q_inv)

  def _sample_command(self, rng: jax.Array) -> jax.Array:
    """[vx*, vy*, ωz*], each uniform in its own range then zeroed with
    probability 1 - keep_prob."""
    key_u, key_b, key_z = jax.random.split(rng, 3)
    lo, hi = self._cmd_ranges[:, 0], self._cmd_ranges[:, 1]
    cmd = jax.random.uniform(key_u, (3,), minval=lo, maxval=hi)
    keep = jax.random.uniform(key_b, (3,)) < self._cmd_keep_prob
    cmd = jp.where(keep, cmd, 0.0)
    all_zero = jax.random.uniform(key_z, ()) < self._cmd_zero_prob
    return jp.where(all_zero, 0.0, cmd)

  def _build_dr_sys(self, foot_friction: jax.Array, added_mass: jax.Array):
    """return a per-env MJX model with randomized friction and base mass."""
    # columns 0:2 of pair_friction are the two tangential (sliding) coefficients.
    new_pair_friction = self.sys.pair_friction.at[self._foot_pair_ids, 0:2].set(foot_friction)
    new_body_mass = self.sys.body_mass.at[self._base_body_id].add(added_mass)
    return self.sys.replace(pair_friction=new_pair_friction, body_mass=new_body_mass)

  def _pipeline_step_dr(self, data: mjx.Data, ctrl: jax.Array, sys) -> mjx.Data:
    """physics step using a per-env model (DR variant of pipeline_step)."""
    def f(data, _):
      data = data.replace(ctrl=ctrl)
      return mjx.step(sys, data), None
    data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
    return data

  # --------------------------------------------------------------------------
  # reset
  # --------------------------------------------------------------------------

  def reset(self, rng: jax.Array) -> State:
    rng, key_xyz, key_ang, key_ax, key_q, key_qd, key_cmd, key_t, \
        key_fr, key_mass, key_kick, key_phase = jax.random.split(rng, 12)

    nj = self.n_joints
    qpos = jp.array(self._init_q)
    qvel = jp.zeros(self.sys.nv)

    # randomise initial state
    r_xyz   = 0.2 * (jax.random.uniform(key_xyz, (3,)) - 0.5)
    r_angle = (jp.pi / 12) * (jax.random.uniform(key_ang, (1,)) - 0.5)
    r_axis  = jax.random.uniform(key_ax, (3,)) - 0.5
    r_axis  = r_axis / jp.linalg.norm(r_axis)
    r_quat  = axis_angle_to_quaternion(r_axis, r_angle)
    r_joint_q  = 0.2 * (jax.random.uniform(key_q,  (nj,)) - 0.5)
    r_joint_qd = 0.5 * (jax.random.uniform(key_qd, (nj,)) - 0.5)

    qpos = qpos.at[0:3].set(qpos[0:3] + r_xyz)
    qpos = qpos.at[3:7].set(r_quat)
    qpos = qpos.at[7:].set(
        jp.clip(qpos[7:] + r_joint_q, self._jnt_lo, self._jnt_hi))
    qvel = qvel.at[6:].set(qvel[6:] + r_joint_qd)

    data = self.pipeline_init(qpos, qvel)

    # initial random velocity command: [vx*, vy*, ωz*] in body frame.
    init_cmd = self._sample_command(key_cmd)
    init_countdown = jax.random.randint(
        key_t, shape=(), minval=1, maxval=self._resample_every + 1)

    # domain randomization: sample per-env physics params
    dr_foot_friction = jax.lax.cond(
        self._use_dr,
        lambda: jax.random.uniform(key_fr, (), minval=self._dr_friction_range[0],
                                   maxval=self._dr_friction_range[1]),
        lambda: self.sys.pair_friction[self._foot_pair_ids[0], 0],
    )
    dr_added_mass = jax.lax.cond(
        self._use_dr,
        lambda: jax.random.uniform(key_mass, (), minval=self._dr_mass_range[0],
                                   maxval=self._dr_mass_range[1]),
        lambda: jp.zeros(()),
    )
    vel_kick_countdown = jax.random.randint(
        key_kick, shape=(), minval=1, maxval=self._kick_every + 1)
    init_phase = jax.random.uniform(key_phase, (), maxval=2.0 * jp.pi)

    state_info = {
        'rng': rng,
        'reward_tuple': {k: 0.0 for k in self.reward_config.rewards.scales},
        'last_action':       jp.array(self._default_ap_pose),
        'afilt_buf':         jp.tile(jp.array(self._default_ap_pose)[None], (self.s_afilt_buf, 1)),
        'step_count':        jp.array(0, dtype=jp.int32),
        'phase_rad':         init_phase,
        'vel_cmd':           init_cmd,            # [vx*, vy*, ωz*]
        'resample_countdown': init_countdown,     # steps until next resample
        'last_joint_vel':    jp.zeros(nj),        # for joint acceleration
        'dr_foot_friction':  dr_foot_friction,    # scalar in [0.5, 1.25]
        'dr_added_mass':     dr_added_mass,       # scalar in [-2, +2] kg
        'vel_kick_countdown': vel_kick_countdown, # steps until next base vel kick
    }

    x, xd = self._pos_vel(data)
    obs = self._get_obs(data, x, xd, state_info, init_phase)
    reward, done = jp.zeros(2)
    metrics = {k: state_info['reward_tuple'][k] for k in state_info['reward_tuple']}
    return State(data, obs, reward, done, metrics, state_info)

  # --------------------------------------------------------------------------
  # termination
  # --------------------------------------------------------------------------

  def compute_termination(self, x: Any, obs: jax.Array, data: Any):
    done = 0.0
    done = jp.where(x.pos[self._base_x_idx, 2] < self.termination_height, 1.0, done)

    joint_qd     = data.qvel[6:]
    joint_angles = data.qpos[7:]

    nonfinite_mask = jp.any(~jp.isfinite(data.qpos))
    nonfinite_mask = jp.any(~jp.isfinite(data.qvel)) | nonfinite_mask
    nonfinite_mask = jp.any(~jp.isfinite(obs))       | nonfinite_mask

    invalid_value_mask = jp.any(jp.abs(joint_angles) > 10)
    invalid_value_mask = jp.any(jp.abs(joint_qd) > 100)     | invalid_value_mask
    invalid_value_mask = jp.any(jp.abs(obs) > 1000)         | invalid_value_mask
    ahac_done = nonfinite_mask | invalid_value_mask

    done = jp.logical_or(done, ahac_done)
    done = jp.array(done, dtype=jp.float32)

    up = jp.array([0.0, 0.0, 1.0])
    done = jp.where(jp.dot(math.rotate(up, x.rot[self._base_x_idx]), up) < 0, 1.0, done)
    done = jp.where(self.early_termination, done, 0.0)
    return done

  # --------------------------------------------------------------------------
  # step
  # --------------------------------------------------------------------------

  def step(self, state: State, action: jax.Array) -> State:

    # action processing
    action    = jp.clip(action, -1, 1)
    # PD position control: the model's actuators are <position>, so ctrl is a
    # joint target. target = default_angles + a * action_scale
    action_target = jp.array(self._default_ap_pose) + action * self.action_scale
    afilt_buf = state.info['afilt_buf']
    afilt_buf = jp.roll(afilt_buf, shift=1, axis=0)
    afilt_buf = afilt_buf.at[0, :].set(action_target)
    f_action  = jp.mean(afilt_buf, axis=0)     # filtered position target
    state.info['afilt_buf'] = afilt_buf

    # domain randomization
    dr_foot_friction = jax.lax.stop_gradient(state.info['dr_foot_friction'])
    dr_added_mass    = jax.lax.stop_gradient(state.info['dr_added_mass'])
    env_sys = self._build_dr_sys(dr_foot_friction, dr_added_mass)

    # velocity kick (before MJX compilation so it is taken into account)
    rng, kick_key, kick_p_key = jax.random.split(state.info['rng'], 3)
    state.info['rng'] = rng
    vel_kick = jax.random.uniform(kick_key, (3,), minval=-0.5, maxval=0.5)
    kick_check = state.info['vel_kick_countdown'] <= 0
    should_kick = kick_check & (
        jax.random.uniform(kick_p_key, ()) < self._kick_prob) & self._use_dr
    ps = state.pipeline_state
    kicked_qvel = jp.where(should_kick, ps.qvel.at[:3].add(vel_kick), ps.qvel)
    ps = ps.replace(qvel=kicked_qvel)
    state.info['vel_kick_countdown'] = jp.where(
        kick_check, self._kick_every, state.info['vel_kick_countdown'] - 1)

    # mjx step
    nj = self.n_joints
    if self.smooth_sigma_q > 0.0:
      rng, key = jax.random.split(state.info['rng'])
      state.info['rng'] = rng
      eps_q = jax.random.normal(key, (nj,)) * self.smooth_sigma_q
      eps_v = jax.random.normal(key, (nj,)) * self.smooth_sigma_v
      data_p = self._pipeline_step_dr(
          ps.replace(qpos=ps.qpos.at[7:].add(+eps_q),
                     qvel=ps.qvel.at[6:].add(+eps_v)), f_action, env_sys)
      data_m = self._pipeline_step_dr(
          ps.replace(qpos=ps.qpos.at[7:].add(-eps_q),
                     qvel=ps.qvel.at[6:].add(-eps_v)), f_action, env_sys)
      data = jax.tree_util.tree_map(
          lambda a, b: 0.5 * (a + b) if jp.issubdtype(a.dtype, jp.floating) else a,
          data_p, data_m)
    else:
      data = self._pipeline_step_dr(ps, f_action, env_sys)

    step_count = state.info['step_count'] + 1
    state.info['step_count'] = step_count

    # vel_cmd
    rng, key_cmd, key_p = jax.random.split(state.info['rng'], 3)
    state.info['rng'] = rng
    new_cmd      = self._sample_command(key_cmd)
    should_check = state.info['resample_countdown'] <= 0
    do_resample  = should_check & (
        jax.random.uniform(key_p, ()) < self._resample_prob)
    vel_cmd      = jp.where(do_resample, new_cmd, state.info['vel_cmd'])
    resample_countdown = jp.where(
        should_check, self._resample_every,
        state.info['resample_countdown'] - 1)
    state.info['vel_cmd']            = vel_cmd
    state.info['resample_countdown'] = resample_countdown

    # gait phase, advanced at a command-dependent frequency
    phase_rad = jp.mod(
        state.info['phase_rad']
        + 2.0 * jp.pi * self._gait_freq(vel_cmd) * self.dt,
        2.0 * jp.pi)
    state.info['phase_rad'] = phase_rad

    # obs & termination
    x, xd = self._pos_vel(data)
    obs   = self._get_obs(data, x, xd, state.info, phase_rad)
    done  = self.compute_termination(x, obs, data)

    # clean NaNs out
    data = jax.tree_util.tree_map(
        lambda v: safe_nan_to_num(v) if jp.issubdtype(v.dtype, jp.floating) else v,
        data)
    x, xd = self._pos_vel(data)
    obs   = self._get_obs(data, x, xd, state.info, phase_rad)

    # calculate some velocities
    q           = x.rot[self._base_x_idx]
    v_lin_body  = self._to_body_frame(xd.vel[self._base_x_idx], q)   # body-frame linear vel
    v_ang_body  = self._to_body_frame(xd.ang[self._base_x_idx], q)   # body-frame angular vel
    gravity     = jp.array([0., 0., -1.])
    g_body      = self._to_body_frame(gravity, q)                    # projected gravity
    g_torso     = self._to_body_frame(gravity, x.rot[self._torso_x_idx])

    # joint accelerations
    joint_vel      = data.qvel[6:]
    joint_vel_prev = state.info['last_joint_vel']
    joint_acc      = (joint_vel - joint_vel_prev) / self.dt

    # rewards
    s = self.reward_config.rewards.scales
    reward_tuple = {
        'lin_vel_tracking': (
            self._reward_lin_vel_tracking(v_lin_body, v_ang_body, vel_cmd)
            * s.lin_vel_tracking
        ),
        'ang_vel_tracking': (
            self._reward_ang_vel_tracking(v_ang_body, vel_cmd)
            * s.ang_vel_tracking
        ),
        'foot_height': (
            self._reward_foot_height(data, phase_rad, vel_cmd)
            * s.foot_height
        ),
        'lin_vel_error': (
            self._reward_lin_vel_error(v_lin_body)
            * s.lin_vel_error
        ),
        'ang_vel_error': (
            self._reward_ang_vel_error(v_ang_body)
            * s.ang_vel_error
        ),
        'base_height': (
            self._reward_base_height(x)
            * s.base_height
        ),
        'fall_barrier': (
            self._reward_fall_barrier(x)
            * s.fall_barrier
        ),
        'base_orientation': (
            self._reward_base_orientation(g_body)
            * s.base_orientation
        ),
        'torso_orientation': (
            self._reward_torso_orientation(g_torso)
            * s.torso_orientation
        ),
        'upper_body_posture': (
            self._reward_upper_body_posture(data)
            * s.upper_body_posture
        ),
        'action_rate': (
            self._reward_action_rate(f_action, state.info['last_action'])
            * s.action_rate
        ),
        'joint_acceleration': (
            self._reward_joint_acceleration(joint_acc)
            * s.joint_acceleration
        ),
        'joint_torque': (
            self._reward_joint_torque(data)
            * s.joint_torque
        ),
    }

    reward = sum(reward_tuple.values())

    # state
    state.info['reward_tuple']  = reward_tuple
    state.info['last_action']   = f_action
    state.info['last_joint_vel'] = joint_vel

    for k in reward_tuple:
      state.metrics[k] = reward_tuple[k]

    return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

  # --------------------------------------------------------------------------
  # observation
  # --------------------------------------------------------------------------

  def _get_obs(self, data: mjx.Data,
               x: Transform, xd: Motion,
               state_info: Dict[str, Any],
               phase_rad: jax.Array = None) -> jax.Array:
    """
      0:  3   linear base velocity (body frame)
      3:  6   angular base velocity (body frame)
      6:  9   projected gravity (pelvis frame)
      9: 12   velocity command [vx*, vy*, ωz*]
     12: 41   joint positions
     41: 70   joint velocities
     70: 99   previous action (normalised)
     99:101   gait phase [sin, cos]
    101:104   projected gravity (torso frame)
    104:110   foot positions relative to the pelvis (body frame)
    110:112   CoM xy relative to the feet midpoint (body frame)
    """
    q          = x.rot[self._base_x_idx]
    v_lin_body = self._to_body_frame(xd.vel[self._base_x_idx], q)
    v_ang_body = self._to_body_frame(xd.ang[self._base_x_idx], q)
    gravity    = jp.array([0., 0., -1.])
    g_body     = self._to_body_frame(gravity, q)
    g_torso    = self._to_body_frame(gravity, x.rot[self._torso_x_idx])

    if phase_rad is None:
      # fallback for reset() which calls _get_obs before step_count exists
      phase_rad = jp.array(0.0)

    base_pos = x.pos[self._base_x_idx]
    foot_pos = data.geom_xpos[self.foot_geom_ids]                  # (2, 3) world
    feet_rel = jax.vmap(self._to_body_frame, in_axes=(0, None))(
        foot_pos - base_pos, q)                                     # (2, 3) body

    # subtree_com[0] is the CoM of the whole model (the world body's subtree).
    com_rel_feet = self._to_body_frame(
        data.subtree_com[0] - jp.mean(foot_pos, axis=0), q)

    return jp.concatenate([
        v_lin_body,                                   #   0:3
        v_ang_body,                                   #   3:6
        g_body,                                       #   6:9
        state_info['vel_cmd'],                        #   9:12
        data.qpos[7:],                                #  12:41 joint positions
        data.qvel[6:],                                #  41:70 joint velocities
        (state_info['last_action'] - jp.array(self._default_ap_pose)) / self.action_scale,
                                                      #  70:99 previous action
        jp.array([jp.sin(phase_rad), jp.cos(phase_rad)]),
                                                      #  99:101 phase (sin alone
                                                      #  is ambiguous: two phases
                                                      #  share every value)
        g_torso,                                      # 101:104
        feet_rel.reshape(-1),                         # 104:110
        com_rel_feet[:2],                             # 110:112
    ])

  # --------------------------------------------------------------------------
  # reward functions
  # --------------------------------------------------------------------------

  def _reward_lin_vel_tracking(self, v_lin_body, v_ang_body, vel_cmd):
    """exp(-||v_xy - v_xy*||² / 0.25) - linear velocity tracking."""
    v_xy     = v_lin_body[:2]
    v_xy_cmd = vel_cmd[:2]
    return jp.exp(-jp.sum(jp.square(v_xy - v_xy_cmd)) / 0.25)

  def _reward_ang_vel_tracking(self, v_ang_body, vel_cmd):
    """exp(-(ωz - ωz*)² / 0.25) - yaw rate tracking."""
    omega_z     = v_ang_body[2]
    omega_z_cmd = vel_cmd[2]
    return jp.exp(-jp.square(omega_z - omega_z_cmd) / 0.25)

  def _reward_foot_height(self, data, phase_rad: jax.Array, vel_cmd: jax.Array):
    """exp(-Σ_j (z_j - z*_j)² / 0.01). The command gates the target, not the
    reward: inside the deadband both feet are asked to stay planted, so
    standing still is what pays.
    uses the foot collision-box positions (z~0.007m at rest)."""
    foot_z = data.geom_xpos[self.foot_geom_ids, 2]               # (2,) actual
    swing = self._swing_profile(phase_rad + self.foot_phase_offsets)
    z_star = self._foot_rest_z + swing * self._cmd_active(vel_cmd)
    return jp.exp(-jp.sum(jp.square(foot_z - z_star)) / 0.01)

  def _swing_profile(self, phi: jax.Array) -> jax.Array:
    """foot lift above its rest height, one value per element of phi.
    first half of the cycle is swing (cubic Bezier 0 -> swing_height -> 0), second half
    is stance (0). With the feet pi apart one of them is always in stance; a
    profile that spends the whole cycle in the air asks for hopping."""
    def bezier(y0, y1, x):
      return y0 + (y1 - y0) * (x ** 3 + 3.0 * (x ** 2) * (1.0 - x))

    # wrap phi into [0, 2π) then map to x ∈ [0, 1); s ∈ [0, 1] runs over the swing half.
    x = jp.mod(phi, 2.0 * jp.pi) / (2.0 * jp.pi)
    s = jp.clip(2.0 * x, 0.0, 1.0)
    up = bezier(0.0, self.swing_height, 2.0 * s)
    down = bezier(self.swing_height, 0.0, 2.0 * s - 1.0)
    return jp.where(x < 0.5, jp.where(s <= 0.5, up, down), 0.0)

  def _gait_freq(self, vel_cmd: jax.Array) -> jax.Array:
    """gait cycles/s, interpolated between gait_freq_min and gait_freq_max by
    the commanded speed so stride length stays sane across the range."""
    speed = jp.clip(jp.linalg.norm(jax.lax.stop_gradient(vel_cmd)), 0.0, 1.0)
    return self.gait_freq_min + (self.gait_freq_max - self.gait_freq_min) * speed

  def _cmd_active(self, vel_cmd: jax.Array) -> jax.Array:
    """1.0 while a velocity is commanded, 0.0 inside the deadband."""
    return jp.asarray(jp.linalg.norm(vel_cmd) > self.cmd_deadband, jp.float32)

  def _reward_lin_vel_error(self, v_lin_body):
    """-vz² - penalise vertical CoM velocity. Clipped to prevent critic divergence."""
    return jp.maximum(-jp.square(v_lin_body[2]), -25.0)

  def _reward_ang_vel_error(self, v_ang_body):
    """-||ωxy||² - penalise pitch and roll rate. Clipped to prevent critic divergence."""
    return jp.maximum(-jp.sum(jp.square(v_ang_body[:2])), -200.0)

  def _reward_base_height(self, x: Transform):
    """exp(-(z - z_home)² / 0.1) - target the keyframe pelvis height (0.755 m)."""
    z = x.pos[self._base_x_idx, 2]
    return jp.exp(-jp.square(z - self._home_base_h) / 0.1)

  def _reward_fall_barrier(self, x: Transform):
    """-d², d = max(0, (barrier_height - z) / (barrier_height - termination_height)).
    Zero above barrier_height, -1 at the termination height."""
    z = x.pos[self._base_x_idx, 2]
    d = jp.maximum(
        0.0,
        (self.barrier_height - z) / (self.barrier_height - self.termination_height))
    return jp.maximum(-jp.square(d), -4.0)

  def _reward_base_orientation(self, g_body):
    """-||g_xy||² - penalise pelvis tilt (gravity leaking into xy in body frame)."""
    return -jp.sum(jp.square(g_body[:2]))

  def _reward_torso_orientation(self, g_torso):
    """-||up_torso - up*||², up* = [0.073, 0, 1]"""
    up_torso = -g_torso
    return -jp.sum(jp.square(up_torso - jp.array([0.073, 0.0, 1.0])))

  def _reward_upper_body_posture(self, data):
    """-||q_upper - q_upper*||² - hold the 17 waist/arm/wrist joints near the
    keyframe pose. The policy commands all 29 joints, and nothing else in the
    reward gives the arms a reason not to flail."""
    q_upper = data.qpos[7 + N_LEG_JOINTS:]
    return -jp.sum(jp.square(q_upper - self._default_upper_pose))

  def _reward_action_rate(self, act, last_act):
    """-||a - a_prev||² - penalise jerky target changes."""
    return -jp.sum(jp.square(act - last_act))

  def _reward_joint_acceleration(self, joint_acc):
    """-||q̈||² - penalise high joint accelerations. clipped to prevent critic divergence."""
    return jp.maximum(-jp.sum(jp.square(joint_acc)), -1e7)

  def _reward_joint_torque(self, data):
    """-||τ||² - penalise high applied torques (actual actuator forces)."""
    return -jp.sum(jp.square(data.actuator_force))


envs.register_environment('g1_29dof', DiffG1)
