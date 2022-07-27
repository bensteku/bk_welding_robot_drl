from multiprocessing.dummy import active_children
from environment.environment import WeldingEnvironmentPybullet
from agent.agent import AgentPybulletDemonstration, AgentPybulletOracle
from time import sleep
from scipy.spatial.transform import Rotation
from util.util import quaternion_to_rpy

a = AgentPybulletOracle("./assets/objects/")
e = WeldingEnvironmentPybullet(a, "./assets/", True, robot="kr16", relative_movement=False)


index = a.dataset["filenames"].index("201910204483_R1.urdf")
a.load_object_into_env(index)

obs = e._get_obs()
done = False

while not done:
    act = a.act(obs)
    #print(e._get_obs())
    #print(act)
    obs, reward, done, info = e.step(act)
    print("reward")
    print(reward)
    sleep(0.075)

e.close()