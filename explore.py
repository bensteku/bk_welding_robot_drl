from environment.environment import WeldingEnvironmentPybullet
from agent.agent import AgentPybulletDemonstration
from time import sleep

e = WeldingEnvironmentPybullet("./assets/", True)
a = AgentPybulletDemonstration("./assets/objects/")

a.set_env(e)
a.load_object_into_env(0)

sleep(10)
for i in range(500):
    act = a.act()
    e.step(act)