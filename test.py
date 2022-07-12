# this is for playing around with the code

from environment.environment import WeldingEnvironmentPybullet
from agent.agent import AgentPybulletDemonstration
from time import sleep

a = AgentPybulletDemonstration("./assets/objects/")
e = WeldingEnvironmentPybullet(a, "./assets/", True, robot="kr16",relative_movement=True)

for i in range(40):
    a.load_object_into_env(i)
    sleep(2)
    e.reset()
e.close()