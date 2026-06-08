import os
import time
import random
import numpy as np
import gymnasium as gym
import metaworld
import torch
import csv

from stable_baselines3 import SAC
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



# =============================================================================
# Setup
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


env_name = "assembly-v3"
SEED = 1
TOTAL_TIMESTEPS = 20_000_000
print('ENV:', env_name)





################################
# Env reward shaping
################################
class SuccessBonusVecWrapper(VecEnvWrapper):
    def __init__(
        self,
        venv,
        beta_start=10.0,
        beta_min=3.0,
        phase1_updates=600,
        decay_updates=400,
        dq_steps_per_update=1024 * 32,
        reward_scale=0.1,
    ):
        super().__init__(venv)
        self.beta_start = beta_start # bonus success
        self.beta_min = beta_min # minimum bonus success
        self.phase1_updates = phase1_updates
        self.decay_updates = decay_updates # decay
        self.dq_steps_per_update = dq_steps_per_update
        self.reward_scale = reward_scale
        self.global_env_steps = 0

    def reset(self):
        self.global_env_steps = 0
        return self.venv.reset()

    def _current_beta(self):
        update = self.global_env_steps // self.dq_steps_per_update + 1
        if update < self.phase1_updates:
            return self.beta_start

        return max(
            self.beta_min,
            self.beta_start * (1.0 - (update - self.phase1_updates) / self.decay_updates),
        )

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        beta = self._current_beta()
        successes = np.zeros_like(rewards, dtype=np.float32)

        for i, info in enumerate(infos):
            successes[i] = float(info.get("success", 0.0))

        raw_rewards = rewards.copy()
        shaped_rewards = raw_rewards + beta * successes
        shaped_rewards = shaped_rewards * self.reward_scale

        for i, info in enumerate(infos):
            info["raw_reward"] = float(raw_rewards[i])
            info["success_bonus_beta"] = float(beta)
            info["success_bonus"] = float(beta * successes[i])

        self.global_env_steps += self.num_envs

        return obs, shaped_rewards, dones, infos


# =============================================================================
# Logging callback (matches your style)
# =============================================================================
class LogCallback(BaseCallback):
	def __init__(self, log_path="sac_log", window=50, save_freq=1_000_000, verbose=0):
		super().__init__(verbose)
		self.start_time = time.time()
		self.window = window
		self.episode_rewards = []
		self.episode_success = []
		self.log_path = log_path
		self.csv_path = os.path.join(log_path, "training_log.csv")
		self.save_freq = save_freq

		with open(self.csv_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["steps", "avg_reward", "success_rate", "sps"])

	def _on_training_start(self):
		n_envs = self.training_env.num_envs
		self.current_returns = np.zeros(n_envs)
		self.current_success = np.zeros(n_envs)

	def _on_step(self) -> bool:
		rewards = self.locals.get("rewards", [])
		dones = self.locals.get("dones", [])
		infos = self.locals.get("infos", [])

		self.current_returns += rewards

		for i, info in enumerate(infos):
			self.current_success[i] = max(
				self.current_success[i],
				float(info.get("success", 0.0))
			)

			if dones[i]:
				self.episode_rewards.append(self.current_returns[i])
				self.episode_success.append(self.current_success[i])

				self.current_returns[i] = 0.0
				self.current_success[i] = 0.0

		# print(self.num_timesteps)
		if self.n_calls % max(1, 10_000 // self.training_env.num_envs) == 0:
			print("Global steps:", self.num_timesteps)

		if self.n_calls % (100_000 // self.training_env.num_envs) == 0:
			recent_rewards = self.episode_rewards[-self.window:]
			avg_reward = np.mean(recent_rewards)

			recent_success = self.episode_success[-self.window:]
			success_rate = np.mean(recent_success)

			sps = int(self.num_timesteps / (time.time() - self.start_time))

			print(
				f"steps={self.num_timesteps} | "
				f"reward={avg_reward:.2f} | "
				f"success={success_rate*100:.1f}% | "
				f"SPS={sps}"
			)

			with open(self.csv_path, "a", newline="") as f:
				writer = csv.writer(f)
				writer.writerow([
					self.num_timesteps,
					avg_reward,
					success_rate,
					sps
				])

		if (
			self.num_timesteps > 0
			and self.num_timesteps % self.save_freq < self.training_env.num_envs
		):
			ckpt_path = f"{self.log_path}/sac_ckpt_{self.num_timesteps}_steps.zip"
			print(f"[Checkpoint] Saving model at {self.num_timesteps} steps -> {ckpt_path}")
			self.model.save(ckpt_path)

		return True


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
	print('Using reward shaping for SAC.')
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
	# SAC model (official baseline)
	# =============================================================================
	seed_everything(SEED)
	print(f"SAC Training...: SEED: {SEED}")
	model = SAC(
		"MlpPolicy",
		env,
		learning_rate=3e-4,
		buffer_size=1_000_000,
		batch_size=BATCH_SIZE,
		gamma=0.99,
		tau=0.005,
		ent_coef="auto",  # automatic entropy tuning (important for fairness)
		train_freq=1,
		gradient_steps=1,
		learning_starts=5_000,
		verbose=0,
		device=DEVICE,
		policy_kwargs=dict(
			net_arch=[HIDDEN_DIM, HIDDEN_DIM]
		),
		seed=SEED,
	)


	# =============================================================================
	# Train
	# =============================================================================
	log_path = f"SAC_runs_shaping/metaworld_{env_name}_{time.strftime('%Y%m%d-%H%M%S')}"
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
	model.save(f"baselines/sac_{env_name}")

	print("Training complete.")
	env.close()


if __name__ == '__main__':
	train()