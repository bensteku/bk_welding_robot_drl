from environment.environment import WeldingEnvironmentPybullet
from agent.agent import AgentPybulletNN
from time import sleep
from scipy.spatial.transform import Rotation
from util.util import quaternion_to_rpy
from util import planner

#a = AgentPybulletOracle("./assets/objects/")
a = AgentPybulletNN("./assets/objects/")
e = WeldingEnvironmentPybullet(a, "./assets/", True, robot="kr16", relative_movement=True)


index = a.dataset["filenames"].index("201910204483_R1.urdf")
a.load_object_into_env(index)
a._set_plan()
a._set_objective()
import pybullet as pyb
import numpy as np
pyb.resetBasePositionAndOrientation(e.robot, np.array([0.5, 1, e.fixed_height[e.robot_name]]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
goal = pyb.calculateInverseKinematics(
            bodyUniqueId=e.robot,
            endEffectorLinkIndex=e.end_effector_link_id[e.robot_name],
            targetPosition=a.objective[0]+np.array([0,0,0.005]) + a.objective[1][0] * 0.02 + 0.02 * a.objective[1][1],
            targetOrientation=e._quat_w_to_ee(a.objective[2]),
            lowerLimits=e.joints_lower[e.robot_name],
            upperLimits=e.joints_upper[e.robot_name],
            jointRanges=e.joints_range[e.robot_name],
            restPoses=np.float32(e.resting_pose_angles[e.robot_name]).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
currj = [pyb.getJointState(e.robot, i)[0] for i in e.joints]
import pybullet_planning as pybp
#input("waiting")
#plan = pybp.plan_joint_motion(e.robot, e.joints, goal, e.obj_ids["fixed"])
#pybp.set_joint_positions(e.robot, e.joints, currj)
#for ding in plan:
#    pybp.set_joint_positions(e.robot, e.joints, ding)
#    sleep(1.5)
#input("waiting again")
pybp.set_joint_positions(e.robot, e.joints, currj)
path = planner.bi_rrt(currj, goal, 0.20, e.robot, e.joints, e.obj_ids["fixed"][0], 500000, 1e-3)
path = planner.interpolate_path(path)
input("Waiting")
for pose in path:
    pybp.set_joint_positions(e.robot, e.joints, pose)
    sleep(0.015)
#e.obj_ids["fixed"]

"""

obs = e._get_obs()
done = False

while not done:
    act = a.act(obs)
    #print(e._get_obs())
    #print(act)
    obs, reward, done, info = e.step(act)
    print("reward")
    print(reward)
    #sleep(0.075)

e.close()
"""