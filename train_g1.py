import smooth_mjx
smooth_mjx.enable(kappa=300, straight_through=True)

import sys
import argparse
import functools
import pickle
import optax

import mujoco
from brax import envs
import jax
import jax.numpy as jp

from envs import register_g1_29dof
import shac.networks as shac_networks
from shac.train import SHAC

env_name = 'g1_29dof'

parser = argparse.ArgumentParser()
parser.add_argument("--name", default='g1_h32_e64_10k_v15b',
                    help="experiment name (tensorboard dir and saved policy)")
parser.add_argument("--init-from", default=None,
                    help="SHAC checkpoint .pkl to warm-start the policy and obs "
                         "normalizer from. Copy it out of shac/checkpoints/ first: "
                         "train() empties that directory when it starts.")
parser.add_argument("--value-burn-in", type=int, default=None,
                    help="training steps that update only the critic "
                         "(default: 200 with --init-from, else 0)")
parser.add_argument("--feet-velocity", type=float, default=0.0,
                    help="weight of the feet_velocity reward (0 = off)")
args = parser.parse_args()

# warm start: the critic is not restored (a changed reward invalidates it), so
# it gets a burn-in against the loaded policy before that policy is updated.
policy_init_params = normalizer_init_params = None
if args.init_from is not None:
    with open(args.init_from, "rb") as f:
        _init = pickle.load(f)["training_state"]
    policy_init_params = _init.policy_params
    normalizer_init_params = _init.normalizer_params
    print(f"warm start from {args.init_from} "
          f"({int(_init.env_steps):,} env steps)")
value_burn_in = args.value_burn_in
if value_burn_in is None:
    value_burn_in = 200 if args.init_from is not None else 0

# Wider than the ANYmal nets: 112-dim obs / 29-dim action against 48/12, and a
# biped has to learn balance on top of the gait.
make_networks_factory = functools.partial(
    shac_networks.make_shac_networks,
    policy_hidden_layer_sizes=(256, 128, 64),
    value_hidden_layer_sizes=(128, 128),
    scalar_var=False,
    layer_norm=True,
)

unroll_length = 32
num_envs = 64
episode_length = 1000          # 1000 x 0.02s = 20s

num_training_steps = 10_000
num_timesteps = num_training_steps * num_envs * unroll_length
num_evals = 50

num_critic_minibatches = 4
critic_batch_size = (num_envs * unroll_length) // num_critic_minibatches

_actor_lr = optax.cosine_decay_schedule(init_value=5e-3, decay_steps=num_training_steps, alpha=1e-5/5e-3)
_critic_lr = optax.cosine_decay_schedule(init_value=2e-3, decay_steps=num_training_steps, alpha=1e-5/2e-3)

env_kwargs = {
    # pelvis starts at 0.755 (knees_bent); 0.5 is a fall, not a deep crouch.
    "termination_height": 0.5,
    "physics_steps_per_control_step": 10,   # 10 x 0.002s = 0.02s control dt (playground)
    "action_scale": 0.5,                    # target = default_angles + a * 0.5
    "swing_height": 0.15,
    "smooth_sigma_q": 0.0,
    "smooth_sigma_v": 0.0,
    "use_domain_randomization": True,
    "kick_prob": 0.3,
    "obs_lin_vel": False,
    "step_speed_gate": 0.2,
    "step_speed_gate_start": 0.35,
    "freeze_clock_at_rest": True,
    "obs_noise": 1.0,
    "action_delay_prob": 0.5,
    "reward_scales": {"action_rate": 0.5, "straightness": 0.5, "feet_separation": 5.0},
}
if args.feet_velocity:
    env_kwargs["reward_scales"]["feet_velocity"] = args.feet_velocity
eval_env_kwargs = {**env_kwargs, "use_domain_randomization": False}

env = envs.get_environment(env_name, **env_kwargs)
eval_env = envs.get_environment(env_name, **eval_env_kwargs)

print(f"Obs size: {env.observation_size}  |  Action size: {env.action_size}")

EXPERIMENT_NAME = args.name

trainer = SHAC(
    environment=env,
    eval_env=eval_env,
    num_timesteps=num_timesteps,
    episode_length=episode_length,
    num_envs=num_envs,
    num_eval_envs=64,
    unroll_length=unroll_length,
    critic_batch_size=critic_batch_size,
    critic_epochs=16,
    target_critic_alpha=0.995,
    discounting=0.99,
    lambda_=0.95,
    normalize_observations=True,
    reward_scaling=1.0,
    network_factory=make_networks_factory,
    actor_learning_rate=_actor_lr,
    critic_learning_rate=_critic_lr,
    entropy_cost=0.0,
    deterministic_eval=True,   # eval the mean action, not a std~0.7 sample of it
    seed=0,
    num_evals=num_evals,
    use_tbx=True,
    tbx_logdir=f'{env_name}_log',
    tbx_experiment_name=EXPERIMENT_NAME,
    resample_init=True,
    scramble_initial_times=True,
    save_all_checkpoints=False,
    checkpoint_every=10,
    polgrad_thresh=1e6,
    grad_clip_norm=None,
    policy_init_params=policy_init_params,
    normalizer_init_params=normalizer_init_params,
    value_burn_in=value_burn_in,
)

make_inference_fn, policy_params, value_params, _ =  trainer.train()

import pathlib
pathlib.Path("saved_policies").mkdir(exist_ok=True)

_save_path = "saved_policies/" + EXPERIMENT_NAME

with open(_save_path, "wb") as f:
    pickle.dump(policy_params, f)

print(f"stage 1 params saved → {_save_path}")
