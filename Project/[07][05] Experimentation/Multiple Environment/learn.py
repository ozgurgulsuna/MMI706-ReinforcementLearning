import random
from stable_baselines3 import PPO  # Import the PPO class
from env1tet import Env1TET
from env2tet import Env2TET  # Adjust the import according to your environment module
from env3tet import Env3TET

# Create the environments
env1 = Env3TET(render_mode="rgb_array")
env2 = Env3TET(render_mode="rgb_array")

# List of environments
env_list = [env1, env2]

# Initialize the PPO model with the first environment in the list
model = PPO("MlpPolicy", env_list[0], verbose=1, tensorboard_log="./ppo_truss_locomotion_tensorboard_2")

# Set the number of episodes and timesteps per episode
n_episodes = 1000
timesteps_per_episode = 5000

# Train the model
for episode in range(n_episodes):
    # Randomly select an environment from the list
    current_env = random.choice(env_list)
    print(current_env)
    
    # Set the current environment
    model.set_env(current_env)
    
    # Learn for a specific number of timesteps
    model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)
    model.save("ppo_truss_locomotion_3")
  


