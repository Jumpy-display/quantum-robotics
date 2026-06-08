import os
import time
import random
import numpy as np
import gymnasium as gym
import metaworld
import torch
import csv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import sys
from pathlib import Path
cwd = Path(__file__).parent.resolve()
sys.path.append(cwd)

from finalAgent3_metaworld import make_metaworld_env, HybridContinuousAgent, RunningMeanStd, seed_everything
from finalAgent3_metaworld import BATCH_SIZE, NUM_ENVS, HIDDEN_DIM, BETA_MAX, BETA_MIN, SCALE_REWARD, STEPS_PER_UPDATE
from SAC_metaworld import SuccessBonusVecWrapper, LogCallback



# =============================================================================
# Setup
# =============================================================================
# PPO: use cpu is more efficient for MLP
DEVICE = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


env_name = "assembly-v3"
SEED = 1
TOTAL_TIMESTEPS = 20_000_000
print('ENV:', env_name)




def train():
	# =============================================================================
	# Vectorized environment
	# =============================================================================
	# env = gym.vector.SyncVectorEnv(
	# 	[make_metaworld_env(env_name, SEED, i) for i in range(NUM_ENVS)]
	# )
	# env = make_vec_env(
	#     lambda: make_metaworld_env(env_name, SEED, 0), 
	#     n_envs=NUM_ENVS
	# )
	env_fns = [make_metaworld_env(env_name, SEED, idx) for idx in range(NUM_ENVS)]
	# env = SubprocVecEnv(env_fns)
	env = DummyVecEnv(env_fns)
	# Proper reward shaping
	print('Using reward shaping for PPO.')
	env = SuccessBonusVecWrapper(
		env,
		beta_start=BETA_MAX, # 10.0, set to 0.0 for default
		beta_min=BETA_MIN, # 3.0, set to 0.0 for default
		phase1_updates=600,
		decay_updates=400,
		dq_steps_per_update=STEPS_PER_UPDATE*NUM_ENVS,
		reward_scale=SCALE_REWARD,  # use 0.1 if you want to match DQ-PPO exactly
	)
	env = VecNormalize(
		env,
		norm_obs=True,
		norm_reward=True,
	)

	# =============================================================================
	# PPO model (official baseline)
	# =============================================================================
	seed_everything(SEED)
	print(f"PPO Training...: SEED {SEED}")
	model = PPO(
		"MlpPolicy",
		env,
		learning_rate=3e-4,
		n_steps=2048,  # PPO uses n_steps instead of buffer_size
		batch_size=BATCH_SIZE,
		gamma=0.99,
		gae_lambda=0.95,
		ent_coef=0.0,
		vf_coef=0.5,
		max_grad_norm=0.5,
		verbose=0,
		device=DEVICE,
		policy_kwargs=dict(net_arch=[HIDDEN_DIM, HIDDEN_DIM]),
		seed=SEED,
	)


	# =============================================================================
	# Train
	# =============================================================================
	log_path = f"PPO_runs_shaping/metaworld_{env_name}_{time.strftime('%Y%m%d-%H%M%S')}"
	os.makedirs(log_path, exist_ok=True)

	callback = LogCallback(log_path=log_path)

	model.learn(
		total_timesteps=TOTAL_TIMESTEPS,
		callback=callback
	)


	# =============================================================================
	# Save model
	# =============================================================================
	os.makedirs("baselines", exist_ok=True)
	model.save(f"baselines/ppo_{env_name}")

	print("Training complete.")
	env.close()


if __name__ == '__main__':
	train()