import smooth_mjx
smooth_mjx.enable(kappa=300, straight_through=True)

import sys
import functools
import optax

import mujoco
from brax import envs
import jax
import jax.numpy as jp

from envs import register_g1_29dof
import shac.networks as shac_networks
from shac.train import SHAC

env_name = 'g1_29dof'

# Wider than the ANYmal nets: 112-dim obs / 29-dim action against 48/12, and a
# biped has to learn balance on top of the gait.
make_networks_factory = functools.partial(
    shac_networks.make_shac_networks,
    policy_hidden_layer_sizes=(256, 128, 64),
    value_hidden_layer_sizes=(128, 128),
    scalar_var=False,
    layer_norm=True,
)

unroll_length = 16
num_envs = 128
episode_length = 1000          # 1000 x 0.02s = 20s

num_training_steps = 40_000
num_timesteps = num_training_steps * num_envs * unroll_length
num_evals = 200

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
}
eval_env_kwargs = {**env_kwargs, "use_domain_randomization": False}

env = envs.get_environment(env_name, **env_kwargs)
eval_env = envs.get_environment(env_name, **eval_env_kwargs)

print(f"Obs size: {env.observation_size}  |  Action size: {env.action_size}")

EXPERIMENT_NAME = 'g1_h32_e64_40k_st'

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
    seed=0,
    num_evals=num_evals,
    use_tbx=True,
    tbx_logdir=f'{env_name}_log',
    tbx_experiment_name=EXPERIMENT_NAME,
    resample_init=True,
    scramble_initial_times=True,
    save_all_checkpoints=False,
    checkpoint_every=50,
    polgrad_thresh=1e6,
    grad_clip_norm=1.0,
)

make_inference_fn, policy_params, value_params, _ =  trainer.train()

import pickle, pathlib
pathlib.Path("saved_policies").mkdir(exist_ok=True)

_save_path = "saved_policies/" + EXPERIMENT_NAME

with open(_save_path, "wb") as f:
    pickle.dump(policy_params, f)

print(f"stage 1 params saved → {_save_path}")
