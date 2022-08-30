from environment.environment import WeldingEnvironmentPybullet, WeldingEnvironmentPybulletConfigSpace
from agent.agent import AgentPybulletNN, AgentPybulletRRTPlanner
from time import sleep

#a = AgentPybulletOracle("./assets/objects/")
a = AgentPybulletRRTPlanner("./assets/objects/")
e = WeldingEnvironmentPybulletConfigSpace(a, "./assets/", True, robot="kr16", relative_movement=True)


index = a.dataset["filenames"].index("201910204483_R1.urdf")
a.load_object_into_env(index)
obs = e._get_obs()
done = False

a.goals = a.goals[18:]

#e.switch_tool(1)
#input("dodo")

while not False:
    act = a.act(obs)
    #print(e._get_obs())
    #print("action")
    #print(act["translate_base"], act["translate"], act["rotate"])
    print("agent state")
    print(a.state)
    print("objective")
    print(a.objective)
    #if a.trajectory:
    #    for i in a.trajectory:
    #        e.movej(i)
    obs, reward, done, success = e.step(act)
    print("obs")
    print(obs["position"], obs["rotation"])
    if success:
        a.trajectory = []
    print("reward")
    print(reward)
    sleep(0.05)

e.close()