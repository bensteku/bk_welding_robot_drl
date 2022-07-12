from multiprocessing.dummy import active_children
from environment.environment import WeldingEnvironmentPybullet
from agent.agent import AgentPybulletDemonstration
from time import sleep
from scipy.spatial.transform import Rotation
from util.util import quaternion_to_rpy

a = AgentPybulletDemonstration("./assets/objects/")
e = WeldingEnvironmentPybullet(a, "./assets/", True, robot="kr16",relative_movement=True)


index = a.dataset["filenames"].index("201910204483_R1.urdf")
print(index)
print(a.dataset["filenames"][index])
#print(a.dataset["frames"][index])
a.load_object_into_env(index)
a._set_goals(index)
print(a.goals[0])
print(quaternion_to_rpy(a.goals[0]["target_rot"][0]))
e.manual_control()

for i in range(500000):

    act = a.act()
    e.step(act)

e.close()