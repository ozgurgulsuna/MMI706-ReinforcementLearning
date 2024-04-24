from stable_baselines3.common.env_checker import check_env
from tetrahedron_locomotion_env import TetrahedronLocomotionEnv
from stable_baselines3 import SAC


# initialize your enviroment
env = TetrahedronLocomotionEnv(render_mode="human")
# it will check your custom environment and output additional warnings if needed
check_env(env)
# learning with tensorboard logging and saving model
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_truss_locomotion_tensorboard/")
model.learn(total_timesteps=150000, log_interval=4)
model.save("sac_truss_locomotion_2")
