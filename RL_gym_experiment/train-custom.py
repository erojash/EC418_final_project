import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time
import os
import numpy as np
from src.utils import make_env


models_dir = f"models"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = make_env(0)()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 100000
model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"PPO3")
model.save("ppo_supertuxkart")

'''
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
	model.save(f"{models_dir}/A2C")

model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_supertuxkart")

model = DQN.load("dqn_supertuxkart")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

'''