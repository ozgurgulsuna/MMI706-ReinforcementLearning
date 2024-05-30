from stable_baselines3 import PPO  # Import the PPO class
from env2tet import Env2TET
from env3tet import Env3TET
from env1tet import Env1TET
import cv2
import imageio

env = Env3TET(render_mode="human")
model = PPO.load("ppo_truss_locomotion_3")

obs, info = env.reset()
frames = []
for _ in range(4000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    image = env.render()
    #if _ % 5 == 0:
     #   frames.append(image)
    #bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    #cv2.imshow("image", bgr)
    #cv2.waitKey(1)
    if done or truncated:
        obs, info = env.reset()

# uncomment to save result as gif
# with imageio.get_writer("media/test.gif", mode="I") as writer:
#     for idx, frame in enumerate(frames):
#         writer.append_data(frame)
