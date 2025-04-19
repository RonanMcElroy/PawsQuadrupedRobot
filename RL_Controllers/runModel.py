import gymnasium as gym
from stable_baselines3 import PPO
import os

############################################  Semi-End-to-End RL Controller ############################################
from pawsSemiEndToEndEnv import PawsEnv

env = PawsEnv(renderMode="human")
env.reset()

models_dir = "models/trotPPO_2"
model_path = f"{models_dir}/1696800.zip"

############################### Policy Modulating Trajectory Generator (PMTG) Controller ###############################
# from pawsEnvPMTG import PawsEnv
#
# env = PawsEnv(renderMode="human", terrain="RANDOM_ROUGH", hieght_mean=0.03, height_std=0.0046, xy_scale=0.045)
# env.reset()
#
# models_dir = "modelsPMTG/PPO_2"
# model_path = f"{models_dir}/10100.zip"

##################################### Comment Out One of the Above to Use the Other ####################################

model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    while terminated==False and truncated==False:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        # print(reward)

env.close()