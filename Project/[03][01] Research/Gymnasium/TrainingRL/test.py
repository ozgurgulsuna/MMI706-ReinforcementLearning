from stable_baselines3 import SAC
from tetrahedron_locomotion_env import TetrahedronLocomotionEnv
import cv2
import imageio

env = TetrahedronLocomotionEnv(render_mode="rgb_array")
model = SAC.load("sac_truss_locomotion_2.zip")

obs, info = env.reset()
frames = []
for _ in range(750):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    image = env.render()
    if _ % 5 == 0:
        frames.append(image)
    bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imshow("image", bgr)
    cv2.waitKey(1)
    if done or truncated:
        obs, info = env.reset()

# uncomment to save result as gif
# with imageio.get_writer("media/test.gif", mode="I") as writer:
#     for idx, frame in enumerate(frames):
#         writer.append_data(frame)
