from matplotlib.pyplot import show
from environment.environment import PathingEnvironmentPybullet
import time

env = PathingEnvironmentPybullet("./assets/", train=False, larger_to_smaller=False, use_joints=True, display=True, show_target=True, ignore_obstacles_for_target_box=False)
#env.ee_pos_reward_thresh = 5e-2


start = time.time()

env.reset()
while True:
    env.manual_control()
    env.step(env.action_space.sample())
    input("testo")

end = time.time()

duration = end-start

print(duration)
print(duration/50)