#!/usr/bin/env python3
"""
live MuJoCo viewer for trained ANYmal policy.
"""
import os
import jax
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import smooth_mjx
smooth_mjx.enable(kappa=300.0)

import sys
import time
import argparse
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import jax
import jax.numpy as jp
import functools
from brax import envs
from brax.training.acme import running_statistics
from envs import register_ahac_anymal
import shac.networks as shac_networks

parser = argparse.ArgumentParser()
parser.add_argument("policy", help="Path to saved policy .pkl")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--print-every", type=int, default=10, metavar="N",
                    help="Print per-step state every N steps (0 = off)")
args = parser.parse_args()

env_kwargs = dict(
    termination_height=0.25,
    physics_steps_per_control_step=4,   # 4 × 0.005s = 0.02s control dt
    model_variant="anymal",
    smooth_sigma_q=0.0,
    smooth_sigma_v=0.0,
)
env = envs.get_environment("ahac_anymal", **env_kwargs)
mj_model = env.model
mj_data  = mujoco.MjData(mj_model)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

make_networks_factory = functools.partial(
    shac_networks.make_shac_networks,
    policy_hidden_layer_sizes=(128, 64, 32),
    value_hidden_layer_sizes=(256, 128, 64),
    scalar_var=False,
    layer_norm=True,
)

print(f"Loading policy from {args.policy} ...")
with open(args.policy, "rb") as f:
    policy_params = pickle.load(f)

_net = make_networks_factory(
    env.observation_size,
    env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
)
make_inference_fn = shac_networks.make_inference_fn(_net)
inference_fn = jax.jit(make_inference_fn(policy_params))

jit_reset = jax.jit(env.reset)
jit_step  = jax.jit(env.step)

rng   = jax.random.PRNGKey(args.seed)
state = jit_reset(rng)
print("JIT-compiling step + inference (first step is slow)...")

## obs
# 0:3   v_lin_body   3:6   v_ang_body   6:9  g_body
# 9:12  vel_cmd      12:24 qpos[7:]     24:36 qvel[6:]
# 36:48 last_action  48:49 sin(phase)

def sync_viewer(state, mj_model, mj_data):
    ps = state.pipeline_state
    mj_data.qpos[:] = np.array(ps.qpos)
    mj_data.qvel[:] = np.array(ps.qvel)
    mujoco.mj_kinematics(mj_model, mj_data)

control_dt    = float(env.dt)
episode_steps = 0
episode_count = 0

print(f"Control dt={control_dt:.4f}s  |  speed={args.speed}x  |  Press Esc to quit")
print("─" * 70)

action, _ = inference_fn(state.obs, rng)
state = jit_step(state, action)
state = jit_reset(rng)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 1

    if args.print_every > 0:
        print(f"{'step':>5}  "
              f"{'vx':>6} {'vy':>6}  "
              f"{'cmd_vx':>7} {'cmd_vy':>7} {'cmd_wz':>7}  "
              f"{'height':>6}  {'phase':>6}")
        print("─" * 70)

    while viewer.is_running():
        t0 = time.perf_counter()

        rng, key = jax.random.split(rng)
        action, _ = inference_fn(state.obs, key)
        state = jit_step(state, action)
        episode_steps += 1

        sync_viewer(state, mj_model, mj_data)
        viewer.sync()

        if args.print_every > 0 and episode_steps % args.print_every == 0:
            obs    = np.array(state.obs)
            vx, vy = obs[0], obs[1]
            cmd_vx, cmd_vy, cmd_wz = obs[9], obs[10], obs[11]
            height = float(state.pipeline_state.qpos[2])
            phase  = float(obs[48])
            print(f"{episode_steps:>5}  "
                  f"{vx:>+6.3f} {vy:>+6.3f}  "
                  f"{cmd_vx:>+7.3f} {cmd_vy:>+7.3f} {cmd_wz:>+7.3f}  "
                  f"{height:>6.3f}  {phase:>+6.3f}")

        if float(state.done) > 0.5:
            episode_count += 1
            height = float(state.pipeline_state.qpos[2])
            print(f"\n── Episode {episode_count:3d} ended after {episode_steps:4d} steps  "
                  f"(height: {height:.3f}m) ──\n")
            if args.print_every > 0:
                print(f"{'step':>5}  "
                      f"{'vx':>6} {'vy':>6}  "
                      f"{'cmd_vx':>7} {'cmd_vy':>7} {'cmd_wz':>7}  "
                      f"{'height':>6}  {'phase':>6}")
                print("─" * 70)
            rng, key = jax.random.split(rng)
            state = jit_reset(key)
            episode_steps = 0

        elapsed = time.perf_counter() - t0
        sleep_t = (control_dt / args.speed) - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
