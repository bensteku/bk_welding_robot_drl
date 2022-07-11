from multiprocessing.dummy import active_children
from environment.environment import WeldingEnvironmentPybullet
from agent.agent import AgentPybulletDemonstration
from time import sleep
from scipy.spatial.transform import Rotation
from util.util import quaternion_to_euler_angle

a = AgentPybulletDemonstration("./assets/objects/")
e = WeldingEnvironmentPybullet(a, "./assets/", True, robot="kr16",relative_movement=True)


a.load_object_into_env(2)

for i in range(500000):
    act = a.act()
    print(e._get_obs())
    e.step(act)