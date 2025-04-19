import gymnasium as gym
from stable_baselines3 import PPO
from pawsSemiEndToEndEnv import PawsEnv
import os
import torch

models_dir = "models/trotPPO_2"
logdir = "logs/trotPPO_2"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = PawsEnv()
env.reset()

##### Learn from Scratch #####
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 128, 32], vf=[256, 128, 32]))
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, ent_coef=0.08, batch_size=1024, verbose=1,
            tensorboard_log=logdir)

##### Transfer Learn #####
# model = PPO.load(f"models/trotPPO_1/1000000.zip", env=env, ent_coef=0.008, tensorboard_log=logdir)


TIMESTEPS = 1200
iters = 0
model.save(f"{models_dir}/{TIMESTEPS*iters}")

while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")

env.close()
