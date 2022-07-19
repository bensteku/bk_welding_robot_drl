from multiprocessing.dummy import active_children
from environment.environment import WeldingEnvironmentPybullet
from agent.agent import AgentPybulletDemonstration, AgentPybulletOracle
from time import sleep
from scipy.spatial.transform import Rotation
from util.util import quaternion_to_rpy

a = AgentPybulletOracle("./assets/objects/")
e = WeldingEnvironmentPybullet(a, "./assets/", True, robot="kr16",relative_movement=True)


index = a.dataset["filenames"].index("201910204483_R1.urdf")
a.load_object_into_env(index)
a._set_goals(index)
#a.goals.pop(0)
#a.goals.pop(0)
#e.manual_control()
e.switch_tool(1)

for i in range(500000):

    act = a.act(e._get_obs())
    e.step(act)
    sleep(0.05)

e.close()