import sys
from pathlib import Path
import warnings
import functools

from brax import envs
from mujoco import mjx
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Tuple, Union

from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, State
from ml_collections import config_dict
import mujoco

from . import anymal_xml
from .mjx_envs import State, MjxEnv
from jax_shac.utils.math_utils import axis_angle_to_quaternion

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
  """Reward config for ANYmal velocity-tracking task (Schwarke et al. CoRL 2025)."""

  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            scales=config_dict.ConfigDict(
                dict(
                    # --- velocity tracking ---
                    lin_vel_tracking=1.0,    # exp(-||v_xy - v_xy*||² / 0.25)
                    ang_vel_tracking=0.5,    # exp(-(ωz - ωz*)² / 0.25)
                    # --- foot height tracking (sinusoidal prescription) ---
                    foot_height=3.0,         # Σ_j (z*_j/0.1) exp(-(z_j - z*_j)²/0.05)
                    # --- velocity penalties ---
                    lin_vel_error=2.0,       # -vz²
                    ang_vel_error=0.05,      # -||ωxy||²
                    # --- base stability ---
                    base_height=1.0,         # exp(-(z - 0.45)² / 0.1)
                    base_orientation=0.5,    # -||g_xy||²
                    # --- smoothness ---
                    action_magnitude=0.05,   # -Σ|a_i|
                    action_rate=0.01,        # -||a - a_prev||²
                    joint_acceleration=2.5e-7,  # -||q̈||²
                    joint_torque=2.5e-5,     # -||τ||²
                )
            ),
        )
    )
    return default_config

  default_config = config_dict.ConfigDict(
      dict(rewards=get_default_rewards_config(),))

  return default_config


class DiffAnymal(MjxEnv):
  """ANYmal velocity-tracking environment matching Schwarke et al. (CoRL 2025).

  Obs (49-dim):
    linear base velocity (body frame)   3
    angular base velocity (body frame)  3
    projected gravity (body frame)      3
    velocity command [vx*, vy*, ωz*]    3
    joint positions                     12
    joint velocities                    12
    previous action                     12
    phase sin(4πt)                      1
    Total                               49

  Velocity commands are randomised in [-1,1] m/s (linear) and [-1,1] rad/s (yaw), resampled every 10–15 s.
  """

  def __init__(
      self,
      action_scale: float = 0.5,
      termination_height: float = 0.25,
      s_afilt_buf: float = 1,
      smooth_sigma_q: float = 0.0,
      smooth_sigma_v: float = 0.0,
      swing_height: float = 0.1,       # sinusoidal foot target amplitude (m)
      reward_scales: dict = None,
      use_domain_randomization: bool = True,
      **kwargs,
  ):
    self.model_variant = kwargs.get('model_variant', 'anymal')

    match self.model_variant:
      case "anymal":
        f_path = anymal_xml
      case _:
        raise ValueError("Invalid model specified!")

    self.early_termination = kwargs.get('early_termination', True)

    mj_model = mujoco.MjModel.from_xml_path(f_path)

    self.s_afilt_buf = s_afilt_buf
    if s_afilt_buf > 1:
      warnings.warn("s_afilt_buf > 1 gives undefined observations")

    physics_steps_per_control_step = 10
    kwargs['physics_steps_per_control_step'] = kwargs.get(
        'physics_steps_per_control_step', physics_steps_per_control_step)
    super().__init__(mj_model=mj_model, **kwargs)

    self.action_scale = action_scale
    self.termination_height = termination_height
    self.smooth_sigma_q = smooth_sigma_q
    self.smooth_sigma_v = smooth_sigma_v
    self.swing_height = swing_height

    # trot gait: LF+RH in phase (offset=0), RF+LH offset by π.
    # order: [LF, RF, LH, RH]
    self.foot_phase_offsets = jp.array([0.0, jp.pi, jp.pi, 0.0])
    # foot contact geom indices (lowest geom on each shank, z≈0.03m at standing).
    # LF=21, RF=28, LH=35, RH=42 — verified via mj_data.geom_xpos at keyframe 0.
    # using geom positions instead of shank body CoM (z≈0.26m) so the 0.1m
    # sinusoidal swing target is relative to the actual contact point.
    self.foot_geom_ids = jp.array([21, 28, 35, 42])

    self._init_q = mj_model.keyframe('standing').qpos
    self._default_ap_pose = mj_model.keyframe('standing').qpos[7:]
    self.reward_config = get_config()
    if reward_scales:
      for k, v in reward_scales.items():
        self.reward_config.rewards.scales[k] = v

    # velocity command resampling bounds (in steps).
    self._resample_min = int(round(10.0 / self.dt))
    self._resample_max = int(round(15.0 / self.dt))

    # used for termination.
    self.lowers = self._default_ap_pose - jp.array([0.2, 0.8, 0.8] * 4)
    self.uppers = self._default_ap_pose + jp.array([0.2, 0.8, 0.8] * 4)

    # domain randomization
    # base values stored at init — always randomize relative to these.
    self._use_dr = use_domain_randomization
    self._base_body_id = 1  # body 0 = world; body 1 = ANYmal trunk
    # velocity kick interval bounds (in steps)
    self._kick_min = int(round(10.0 / self.dt))
    self._kick_max = int(round(15.0 / self.dt))

  # --------------------------------------------------------------------------
  # Helpers
  # --------------------------------------------------------------------------

  def _to_body_frame(self, v_world: jax.Array, q_body: jax.Array) -> jax.Array:
    """Rotate a world-frame vector into the robot body frame.

    q_body is the body-to-world quaternion (w, x, y, z).
    Inverse = (w, -x, -y, -z) for unit quaternions.
    """
    q_inv = jp.array([q_body[0], -q_body[1], -q_body[2], -q_body[3]])
    return math.rotate(v_world, q_inv)

  def _build_dr_sys(self, foot_friction: jax.Array, added_mass: jax.Array):
    """Return a per-env MJX model with randomized friction and base mass."""
    new_geom_friction = self.sys.geom_friction.at[self.foot_geom_ids, 0].set(foot_friction)
    new_body_mass = self.sys.body_mass.at[self._base_body_id].add(added_mass)
    return self.sys.replace(geom_friction=new_geom_friction, body_mass=new_body_mass)

  def _pipeline_step_dr(self, data: mjx.Data, ctrl: jax.Array, sys) -> mjx.Data:
    """Physics step using a per-env model (DR variant of pipeline_step)."""
    def f(data, _):
      data = data.replace(ctrl=ctrl)
      return mjx.step(sys, data), None
    data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
    return data

  # --------------------------------------------------------------------------
  # Reset
  # --------------------------------------------------------------------------

  def reset(self, rng: jax.Array) -> State:
    rng, key_xyz, key_ang, key_ax, key_q, key_qd, key_cmd, key_t, \
        key_fr, key_mass, key_kick = jax.random.split(rng, 11)

    qpos = jp.array(self._init_q)
    qvel = jp.zeros(18)

    # Randomise initial state
    r_xyz   = 0.2 * (jax.random.uniform(key_xyz, (3,)) - 0.5)
    r_angle = (jp.pi / 12) * (jax.random.uniform(key_ang, (1,)) - 0.5)
    r_axis  = jax.random.uniform(key_ax, (3,)) - 0.5
    r_axis  = r_axis / jp.linalg.norm(r_axis)
    r_quat  = axis_angle_to_quaternion(r_axis, r_angle)
    r_joint_q  = 0.2 * (jax.random.uniform(key_q,  (12,)) - 0.5)
    r_joint_qd = 0.5 * (jax.random.uniform(key_qd, (12,)) - 0.5)

    qpos = qpos.at[0:3].set(qpos[0:3] + r_xyz)
    qpos = qpos.at[3:7].set(r_quat)
    qpos = qpos.at[7:19].set(qpos[7:19] + r_joint_q)
    qvel = qvel.at[6:18].set(qvel[6:18] + r_joint_qd)

    data = self.pipeline_init(qpos, qvel)

    # Initial random velocity command: [vx*, vy*, ωz*] in body frame.
    init_cmd = jax.random.uniform(key_cmd, (3,)) * 2.0 - 1.0  # [-1, 1]
    init_countdown = jax.random.randint(
        key_t, shape=(), minval=self._resample_min, maxval=self._resample_max)

    # Domain randomization: sample per-env physics params
    dr_foot_friction = jax.lax.cond(
        self._use_dr,
        lambda: jax.random.uniform(key_fr, (), minval=0.5, maxval=1.25),
        lambda: self.sys.geom_friction[self.foot_geom_ids[0], 0],
    )
    dr_added_mass = jax.lax.cond(
        self._use_dr,
        lambda: jax.random.uniform(key_mass, (), minval=-5.0, maxval=5.0),
        lambda: jp.zeros(()),
    )
    vel_kick_countdown = jax.random.randint(
        key_kick, shape=(), minval=self._kick_min, maxval=self._kick_max)

    state_info = {
        'rng': rng,
        'reward_tuple': {
            'lin_vel_tracking': 0.0,
            'ang_vel_tracking': 0.0,
            'foot_height':      0.0,
            'lin_vel_error':    0.0,
            'ang_vel_error':    0.0,
            'base_height':      0.0,
            'base_orientation': 0.0,
            'action_magnitude': 0.0,
            'action_rate':      0.0,
            'joint_acceleration': 0.0,
            'joint_torque':     0.0,
        },
        'last_action':       jp.array(self._default_ap_pose),
        'afilt_buf':         jp.tile(jp.array(self._default_ap_pose)[None], (self.s_afilt_buf, 1)),
        'step_count':        jp.array(0, dtype=jp.int32),
        'vel_cmd':           init_cmd,            # [vx*, vy*, ωz*]
        'resample_countdown': init_countdown,     # steps until next resample
        'last_joint_vel':    jp.zeros(12),        # for joint acceleration
        'dr_foot_friction':  dr_foot_friction,    # scalar in [0.5, 1.25]
        'dr_added_mass':     dr_added_mass,       # scalar in [-5, +5] kg
        'vel_kick_countdown': vel_kick_countdown, # steps until next base vel kick
    }

    x, xd = self._pos_vel(data)
    obs = self._get_obs(data.qpos, data.qvel, x, xd, state_info)
    reward, done = jp.zeros(2)
    metrics = {k: state_info['reward_tuple'][k] for k in state_info['reward_tuple']}
    return State(data, obs, reward, done, metrics, state_info)

  # --------------------------------------------------------------------------
  # Termination
  # --------------------------------------------------------------------------

  def compute_termination(self, x: Any, obs: jax.Array, data: Any):
    done = 0.0
    done = jp.where(x.pos[0, 2] < self.termination_height, 1.0, done)

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
    done = jp.where(jp.dot(math.rotate(up, x.rot[0]), up) < 0, 1.0, done)
    done = jp.where(self.early_termination, done, 0.0)
    return done

  # --------------------------------------------------------------------------
  # Step
  # --------------------------------------------------------------------------

  def step(self, state: State, action: jax.Array) -> State:

    # action processing
    action    = jp.clip(action, -1, 1)
    raw_action = action                         # keep [-1,1] for action_magnitude
    # PD position control 
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
    rng, kick_key, kick_t_key = jax.random.split(state.info['rng'], 3)
    state.info['rng'] = rng
    vel_kick = jax.random.uniform(kick_key, (3,), minval=-0.5, maxval=0.5)
    should_kick = state.info['vel_kick_countdown'] <= 0
    ps = state.pipeline_state
    kicked_qvel = jp.where(should_kick, ps.qvel.at[:3].add(vel_kick), ps.qvel)
    ps = ps.replace(qvel=kicked_qvel)
    new_kick_countdown = jax.random.randint(
        kick_t_key, shape=(), minval=self._kick_min, maxval=self._kick_max)
    state.info['vel_kick_countdown'] = jp.where(
        should_kick, new_kick_countdown, state.info['vel_kick_countdown'] - 1)

    # mjx step
    if self.smooth_sigma_q > 0.0:
      rng, key = jax.random.split(state.info['rng'])
      state.info['rng'] = rng
      eps_q = jax.random.normal(key, (12,)) * self.smooth_sigma_q
      eps_v = jax.random.normal(key, (12,)) * self.smooth_sigma_v
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

    # phase maths
    step_count = state.info['step_count'] + 1
    state.info['step_count'] = step_count
    phase_rad = 4.0 * jp.pi * jp.asarray(step_count, jp.float32) * self.dt

    # vel_cmd
    rng, key_cmd, key_t = jax.random.split(state.info['rng'], 3)
    state.info['rng'] = rng
    new_cmd      = jax.random.uniform(key_cmd, (3,)) * 2.0 - 1.0
    should_resample = state.info['resample_countdown'] <= 0
    vel_cmd      = jp.where(should_resample, new_cmd, state.info['vel_cmd'])
    new_countdown = jax.random.randint(
        key_t, shape=(), minval=self._resample_min, maxval=self._resample_max)
    resample_countdown = jp.where(
        should_resample, new_countdown, state.info['resample_countdown'] - 1)
    state.info['vel_cmd']            = vel_cmd
    state.info['resample_countdown'] = resample_countdown

    # obs & termination
    x, xd = self._pos_vel(data)
    obs   = self._get_obs(data.qpos, data.qvel, x, xd, state.info, phase_rad)
    done  = self.compute_termination(x, obs, data)

    # clean NaNs out
    data = jax.tree_util.tree_map(
        lambda v: safe_nan_to_num(v) if jp.issubdtype(v.dtype, jp.floating) else v,
        data)
    x, xd = self._pos_vel(data)
    obs   = self._get_obs(data.qpos, data.qvel, x, xd, state.info, phase_rad)

    # calculate some velocities
    q           = x.rot[0]
    v_lin_body  = self._to_body_frame(xd.vel[0], q)   # (3,) body-frame linear vel
    v_ang_body  = self._to_body_frame(xd.ang[0], q)   # (3,) body-frame angular vel
    g_body      = self._to_body_frame(jp.array([0., 0., -1.]), q)  # projected gravity

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
            self._reward_foot_height(data, phase_rad)
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
        'base_orientation': (
            self._reward_base_orientation(g_body)
            * s.base_orientation
        ),
        'action_magnitude': (
            self._reward_action_magnitude(raw_action)
            * s.action_magnitude
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
  # Observation
  # --------------------------------------------------------------------------

  def _get_obs(self, qpos: jax.Array, qvel: jax.Array,
               x: Transform, xd: Motion,
               state_info: Dict[str, Any],
               phase_rad: jax.Array = None) -> jax.Array:
    """
      0: 3   linear base velocity (body frame)
      3: 6   angular base velocity (body frame)
      6: 9   projected gravity (body frame)
      9:12   velocity command [vx*, vy*, ωz*]
     12:24   joint positions
     24:36   joint velocities
     36:48   previous action (normalised)
     48:49   phase sin(4πt)
    """
    q          = x.rot[0]
    v_lin_body = self._to_body_frame(xd.vel[0], q)
    v_ang_body = self._to_body_frame(xd.ang[0], q)
    g_body     = self._to_body_frame(jp.array([0., 0., -1.]), q)

    if phase_rad is None:
      # fallback for reset() which calls _get_obs before step_count exists
      phase_rad = jp.array(0.0)

    return jp.concatenate([
        v_lin_body,                                   # 0:3
        v_ang_body,                                   # 3:6
        g_body,                                       # 6:9
        state_info['vel_cmd'],                        # 9:12
        qpos[7:],                                     # 12:24 joint positions
        qvel[6:],                                     # 24:36 joint velocities
        (state_info['last_action'] - jp.array(self._default_ap_pose)) / self.action_scale,  # 36:48
        jp.sin(phase_rad).reshape(1),                 # 48:49
    ])

  # --------------------------------------------------------------------------
  # reward functions
  # --------------------------------------------------------------------------

  def _reward_lin_vel_tracking(self, v_lin_body, v_ang_body, vel_cmd):
    """exp(-||v_xy - v_xy*||² / 0.25)  — linear velocity tracking."""
    v_xy     = v_lin_body[:2]
    v_xy_cmd = vel_cmd[:2]
    return jp.exp(-jp.sum(jp.square(v_xy - v_xy_cmd)) / 0.25)

  def _reward_ang_vel_tracking(self, v_ang_body, vel_cmd):
    """exp(-(ωz - ωz*)² / 0.25)  — yaw rate tracking."""
    omega_z     = v_ang_body[2]
    omega_z_cmd = vel_cmd[2]
    return jp.exp(-jp.square(omega_z - omega_z_cmd) / 0.25)

  def _reward_foot_height(self, data, phase_rad: jax.Array):
    """Σ_j (z*_j / 0.1) · exp(-(z_j - z*_j)² / 0.05)

    sinusoidal foot target: z*_j = max(0, swing_height * sin(phase + offset_j))
    during stance (sin < 0): z*_j = 0, weight = 0 → no contribution.
    during swing (sin > 0): foot rewarded for tracking the sinusoidal lift.
    uses geom positions (z≈0.03m at standing) not shank CoM (z≈0.26m).
    """
    foot_z  = data.geom_xpos[self.foot_geom_ids, 2]             # (4,) actual
    z_star  = jp.maximum(
        0.0,
        self.swing_height * jp.sin(phase_rad + self.foot_phase_offsets)
    )                                                             # (4,) target
    weight  = z_star / 0.1                                       # normalised
    return jp.sum(weight * jp.exp(-jp.square(foot_z - z_star) / 0.05))

  def _reward_lin_vel_error(self, v_lin_body):
    """-vz²  — penalise vertical CoM velocity. Clipped to prevent critic divergence."""
    return jp.maximum(-jp.square(v_lin_body[2]), -25.0)

  def _reward_ang_vel_error(self, v_ang_body):
    """-||ωxy||²  — penalise pitch and roll rate. Clipped to prevent critic divergence."""
    return jp.maximum(-jp.sum(jp.square(v_ang_body[:2])), -200.0)

  def _reward_base_height(self, x: Transform):
    """exp(-(z - 0.45)² / 0.1)  — target height 0.45 m."""
    z = x.pos[0, 2]
    return jp.exp(-jp.square(z - 0.45) / 0.1)

  def _reward_base_orientation(self, g_body):
    """-||g_xy||²  — penalise body tilt (gravity leaking into xy in body frame)."""
    return -jp.sum(jp.square(g_body[:2]))

  def _reward_action_magnitude(self, raw_action):
    """-Σ|a_i|  — L1 penalty on normalised actions."""
    return -jp.sum(jp.abs(raw_action))

  def _reward_action_rate(self, act, last_act):
    """-||a - a_prev||²  — penalise jerky torque changes."""
    return -jp.sum(jp.square(act - last_act))

  def _reward_joint_acceleration(self, joint_acc):
    """-||q̈||²  — penalise high joint accelerations. Clipped to prevent critic divergence."""
    return jp.maximum(-jp.sum(jp.square(joint_acc)), -1e7)

  def _reward_joint_torque(self, data):
    """-||τ||²  — penalise high applied torques (actual actuator forces)."""
    return -jp.sum(jp.square(data.actuator_force))


envs.register_environment('ahac_anymal', DiffAnymal)
