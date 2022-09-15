from environment.environment import WeldingEnvironmentPybullet, WeldingEnvironmentPybulletConfigSpace
from agent.agent import AgentPybulletNN, AgentPybulletRRTPlanner
from time import sleep

#a = AgentPybulletOracle("./assets/objects/")
a = AgentPybulletRRTPlanner()
e = WeldingEnvironmentPybulletConfigSpace(a, "./assets/", True, robot="kr16")


obs = e._get_obs(False)
done = False

#e.switch_tool(1)
#input("dodo")

while not False:
    act = a.act(obs)
    obs, reward, done, info = e.step(act)
    if info["is_success"]:
        a.trajectory = []
    sleep(0.05)

e.close()