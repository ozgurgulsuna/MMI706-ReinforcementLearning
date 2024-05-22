from stable_baselines3.common.env_checker import check_env
from tetrahedron_locomotion_env import TetrahedronLocomotionEnv
from env import TetrahedronLocomotionEnv2
from stable_baselines3 import SAC
import random


# initialize your enviroment
env = TetrahedronLocomotionEnv(render_mode="human")
env2 = TetrahedronLocomotionEnv2(render_mode="human")

env_list =[env , env2, env ,env2]
# it will check your custom environment and output additional warnings if needed
#check_env(env)
#check_env(env2)
# learning with tensorboard logging and saving model
model = SAC("MlpPolicy", env_list[0], verbose=1, tensorboard_log="./sac_truss_locomotion_tensorboard/")
n_episodes = 10
timesteps_per_episode = 20

for episode in range(n_episodes):
   current_env = random.choice(env_list)
   print(current_env)
   
   model.set_env(current_env)
   
   model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)
  
model.save("aa")