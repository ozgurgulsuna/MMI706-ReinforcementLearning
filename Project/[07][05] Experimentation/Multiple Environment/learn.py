from stable_baselines3.common.env_checker import check_env
from env1tet import Env1TET
from env2tet import Env2TET
from ant_v4 import AntEnv
from  tetrahedron_locomotion_env import TetrahedronLocomotionEnv
#from env import% Env1TET, Env2TET
from stable_baselines3 import SAC
import random


# initialize your enviroment
env1= Env1TET(render_mode="depth_array")
env2 = Env2TET(render_mode="depth_array")

env_list =[env1 , env2, env1 ,env2]
# it will check your custom environment and output additional warnings if needed
#check_env(env)
#check_env(env2)
# learning with tensorboard logging and saving model
model = SAC("MlpPolicy", env_list[0], verbose=1, tensorboard_log="./sac_truss_locomotion_tensorboard/")
#model.learn(total_timesteps=150000, log_interval=4)
#model.save("sac_truss_locosdsmotion_2")
n_episodes = 100
timesteps_per_episode = 1000

for episode in range(n_episodes):
   current_env = random.choice(env_list)
   print(current_env)
   
   model.set_env(current_env)
   
   model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)
  
model.save("aa")
