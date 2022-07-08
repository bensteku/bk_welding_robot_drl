# this is for playing around with the code

from environment.environment import WeldingEnvironmentPybullet
from agent.agent import AgentPybulletDemonstration
from time import sleep

e = WeldingEnvironmentPybullet("./assets/", True)
a = AgentPybulletDemonstration("./assets/objects/")
a.set_env(e)
for i in range(40):
    a.load_object_into_env(i)
    sleep(2)
    e.reset()
print(e.obj_ids)
e.close()