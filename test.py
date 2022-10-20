from matplotlib.pyplot import show
from environment.environment import PathingEnvironmentPybullet
import time

env = PathingEnvironmentPybullet("./assets/", train=True, use_joints=True, display=True, show_target=True, ignore_obstacles_for_target_box=False)

start = time.time()

for i in range(150):
    env.reset()
    env.step(env.action_space.sample())
    input("testo")

end = time.time()

duration = end-start

print(duration)
print(duration/50)