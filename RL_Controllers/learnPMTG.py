import gymnasium as gym
from stable_baselines3 import PPO
from pawsEnvPMTG import PawsEnv
import os
import torch

models_dir = "modelsPMTG/PPO_2"
logdir = "logsPMTG/PPO_2"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = PawsEnv(terrain="RANDOM_ROUGH", hieght_mean=0.03, height_std=0.008, xy_scale=0.045)
env.reset()

##### Learn from Scratch #####
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[128, 64, 16], vf=[128, 64, 16]))
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, ent_coef=0.03, batch_size=1024, verbose=1,
            tensorboard_log=logdir)

##### Transfer Learn #####
# model = PPO.load(f"modelsPMTG/PPO_1/20800.zip", env=env, ent_coef=0.03, batch_size=1024, verbose=1,
#                  tensorboard_log=logdir)

TIMESTEPS = 100
iters = 0
model.save(f"{models_dir}/{TIMESTEPS*iters}")

while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")

env.close()
