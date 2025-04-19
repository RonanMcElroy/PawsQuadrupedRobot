from stable_baselines3.common.env_checker import check_env
from pawsSemiEndToEndEnv import PawsEnv
# from pawsEnvPMTG import PawsEnv

################ This script is used check that the created custom environment is valid and error free #################

env = PawsEnv()
# This function will check the custom environment and output warnings if needed
check_env(env)

# ### Double check ###
# env = PawsEnv(renderMode="human")
# episodes = 10
#
# for episode in range(episodes):
# 	terminated = False
# 	truncated = False
# 	obs, info = env.reset()
# 	while not truncated and not terminated:
# 		random_action = env.action_space.sample()
#
# 		obs, reward, terminated, truncated, info = env.step(random_action)
# 		# print('reward',reward)
# 		# print('observation', obs)
