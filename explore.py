from environment.environment import WeldingEnvironmentPybullet, WeldingEnvironmentPybulletConfigSpace
from agent.agent import AgentPybulletNN, AgentPybulletRRTPlanner
from time import sleep

#a = AgentPybulletOracle("./assets/objects/")
a = AgentPybulletRRTPlanner("./assets/objects/")
e = WeldingEnvironmentPybulletConfigSpace(a, "./assets/", True, robot="kr16")


index = a.dataset["filenames"].index("201910204483_R1.urdf")
a.load_object_into_env(index)
obs = e._get_obs()
done = False

a.goals = a.goals[18:]

#e.switch_tool(1)
#input("dodo")

while not False:
    a.update_objectives()
    if a.path_state !=2 and e.move_base(a.objective[4]):
        obs = e._get_obs()
    act = a.act(obs)
    #print(e._get_obs())
    print("agent state")
    print(a.path_state)
    print("objective")
    print(a.objective)
    obs, reward, done, success = e.step(act)
    print("obs")
    print(obs[1], obs[2])
    if success:
        a.trajectory = []
    print("reward")
    print(reward)
    sleep(0.05)

e.close()