#!/usr/bin/env python3
"""
interactive ANYmal driver — gamepad control.

  Left  stick  : forward / backward / strafe  (vx, vy)
  Right stick  : rotate left / right           (wz)
  A / Cross    : reset episode
  B / Circle   : quit
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"   # joystick only, no pygame window

import jax
jax.config.update("jax_compilation_cache_dir", os.path.expanduser(".cache/jax"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import smooth_mjx
smooth_mjx.enable(kappa=300.0)

import time
import argparse
import pickle
import functools
import numpy as np
import pygame
import mujoco
import mujoco.viewer
import jax
import jax.numpy as jp
from brax import envs
from brax.training.acme import running_statistics
from envs import register_ahac_anymal
import shac.networks as shac_networks

parser = argparse.ArgumentParser()
parser.add_argument("policy", help="Path to saved policy .pkl")
parser.add_argument("--speed",    type=float, default=1.0)
parser.add_argument("--seed",     type=int,   default=0)
parser.add_argument("--max-vel",  type=float, default=1.0)
parser.add_argument("--deadzone", type=float, default=0.12)
args = parser.parse_args()

pygame.init()
pygame.joystick.init()

def _find_joystick():
    pygame.joystick.quit()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        return None
    j = pygame.joystick.Joystick(0)
    j.init()
    return j

print("Looking for controller...")
joy = _find_joystick()
while joy is None:
    print("  No controller found — plug one in and press Enter...")
    input()
    joy = _find_joystick()
print(f"  Connected: {joy.get_name()}\n")

def _axis(joy, i):
    v = joy.get_axis(i)
    return v if abs(v) > args.deadzone else 0.0

def _read_cmd(joy):
    pygame.event.pump()
    V  = args.max_vel
    vx = -_axis(joy, 1) * V     # left stick Y, up = forward
    vy = -_axis(joy, 0) * V     # left stick X, left = strafe-left
    wz = -_axis(joy, 3) * V     # right stick X, left = turn-left
    reset = joy.get_button(0)   # A / Cross
    quit_ = joy.get_button(1)   # B / Circle
    return np.array([vx, vy, wz], dtype=np.float32), reset, quit_

env = envs.get_environment("ahac_anymal",
    termination_height=0.25,
    physics_steps_per_control_step=4,
    model_variant="anymal",
    smooth_sigma_q=0.0,
    smooth_sigma_v=0.0,
)
mj_model   = env.model
mj_data    = mujoco.MjData(mj_model)
control_dt = float(env.dt)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

print(f"Loading {args.policy} ...")
with open(args.policy, "rb") as f:
    policy_params = pickle.load(f)

_net = functools.partial(
    shac_networks.make_shac_networks,
    policy_hidden_layer_sizes=(128, 64, 32),
    value_hidden_layer_sizes=(256, 128, 64),
    scalar_var=False, layer_norm=True,
)(env.observation_size, env.action_size,
  preprocess_observations_fn=running_statistics.normalize)

inference_fn = jax.jit(shac_networks.make_inference_fn(_net)(policy_params))
jit_reset    = jax.jit(env.reset)
jit_step     = jax.jit(env.step)

print("Compiling...")
rng   = jax.random.PRNGKey(args.seed)
state = jit_reset(rng)
action, _ = inference_fn(state.obs, rng)
state = jit_step(state, action)
state = jit_reset(rng)
print("Ready!\n")
print("  Left stick=move  Right stick=rotate  A=reset  B=quit")
print("─" * 60)

def _inject_cmd(state, jax_cmd):
    new_info = {**state.info, 'vel_cmd': jax_cmd,
                'resample_countdown': jp.array(9999)}
    return state.replace(info=new_info, obs=state.obs.at[9:12].set(jax_cmd))

def _sync(state):
    ps = state.pipeline_state
    mj_data.qpos[:] = np.array(ps.qpos)
    mj_data.qvel[:] = np.array(ps.qvel)
    mujoco.mj_kinematics(mj_model, mj_data)

episode_steps = 0
episode_count = 0
prev_reset_btn = False   # edge-detect to avoid repeated resets

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 1
    viewer.cam.distance    = 3.5
    viewer.cam.elevation   = -20.0

    while viewer.is_running():
        t0 = time.perf_counter()

        cmd, reset_btn, quit_btn = _read_cmd(joy)

        if quit_btn:
            break

        if reset_btn and not prev_reset_btn:
            cmd[:] = 0.0
            rng, key = jax.random.split(rng)
            state = jit_reset(key)
            episode_count += 1
            episode_steps = 0
            print(f"\n── Manual reset (episode {episode_count}) ──")
        prev_reset_btn = bool(reset_btn)

        jax_cmd = jp.array(cmd)
        state   = _inject_cmd(state, jax_cmd)

        rng, key = jax.random.split(rng)
        action, _ = inference_fn(state.obs, key)
        state = jit_step(state, action)
        episode_steps += 1

        _sync(state)
        viewer.sync()

        if episode_steps % 10 == 0:
            obs = np.array(state.obs)
            print(f"\r  cmd [{cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f}]"
                  f"  actual [{obs[0]:+.2f} {obs[1]:+.2f}]"
                  f"  step {episode_steps:4d}  ", end="", flush=True)

        if float(state.done) > 0.5:
            print(f"\n── Fell after {episode_steps} steps "
                  f"(h={float(state.pipeline_state.qpos[2]):.3f} m) ──")
            rng, key = jax.random.split(rng)
            state = jit_reset(key)
            cmd = np.zeros(3, dtype=np.float32)
            episode_steps = 0
            episode_count += 1

        elapsed = time.perf_counter() - t0
        sleep_t = (control_dt / args.speed) - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

pygame.quit()
print("\nBye!")
