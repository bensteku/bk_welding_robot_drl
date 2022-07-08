from ast import Not
from turtle import Turtle
import gym
import pybullet as pyb
import time
import numpy as np
from util.util import matrix_to_quaternion, rpy_to_quaternion

# TODO: subclass such that there are two classes, one for using pybullet, one for MOSES

class WeldingEnvironment(gym.Env):

    def __init__(self):
        
        self._random_state = None

        # variables needed by Gym env subclasses

        #   contains the position (as xyz) and rotation (as quaternion xyzw) of the end effector (i.e. the welding torch) in workspace coordinates
        min_position = np.array([0., -1., 0.1])  # provisional
        max_position = np.array([1., 1, 0.3])

        min_rotation = np.array(rpy_to_quaternion([ele * np.pi/180. for ele in [-10., -10., -100.]])) 
        max_rotation = np.array(rpy_to_quaternion([ele * np.pi/180. for ele in [10., 10., 100.]]))
        self.observation_space = gym.spaces.Dict(
            {
                'position': gym.spaces.Box(low=min_position, high=max_position, shape=(3,), dtype=np.float32),
                'rotation': gym.spaces.Box(low=min_rotation, high=max_rotation, shape=(4,), dtype=np.float32)
            }
        )
        
        #   actions consist of translating and rotating the end effector
        self.action_space = gym.spaces.Dict(
            {
                'translate': gym.spaces.Box(low=min_position, high=max_position, shape=(3,), dtype=np.float32),
                'rotate': gym.spaces.Box(low=min_rotation, high=max_rotation, shape=(4,), dtype=np.float32)
            }
        )
        # TODO: marginal movement or absolute movement?

        # agent, needs to be set after construction due to mutual dependence
        self.agent = 1#None

    #####################################################
    # methods for Gym subclass, left as virtual methods #
    #####################################################

    def seed(self, seed=None):
        self._random_state = np.random.RandomState(seed)
        return self._random_state

    def reset(self):

        raise NotImplementedError
        
    def _get_obs(self):

        raise NotImplementedError

    def step(self, action=None):

        if not self.agent:  # no agent has been set
            raise ValueError("Agent needs to be created and set for this environment via set_agent()")

        self._perform_action(action)
        reward, info = self.calc_reward() if action is not None else (0, {})
        done = self.is_done()

        return self._get_obs(), reward, done, info

    # helper method

    def perform_action(self):

        raise NotImplementedError

    def close(self):

        raise NotImplementedError

    #################
    # other methods #
    #################

    def perform_action(self):

        raise NotImplementedError

    def is_done(self):

        raise NotImplementedError

    def set_agent(self, agent):

        self.agent = agent

class WeldingEnvironmentMOSES(WeldingEnvironment):

    def __init__(self,
                asset_files_path,
                ):

        raise NotImplementedError

class WeldingEnvironmentPybullet(WeldingEnvironment):

    def __init__(self,
                asset_files_path,
                display=False,
                hz=240):

        super().__init__()

        self.asset_files_path = asset_files_path
        self.obj_ids = {'fixed': [], 'rigid': []}  # dict of objects by type of body dynamics
        self.resting_pose_angles = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi  # angles for the default pose of the UR5 robot, to be replaced with some angles of the welding robot
        self.tool = 0  # 0: TAND GERAD, 1: MRW510
        
        # pybullet connection and setup
        disp = pyb.DIRECT  # direct <-> no gui, use for training
        if display:
            disp = pyb.GUI
        client = pyb.connect(disp)

        pyb.setTimeStep(1. / hz)
        pyb.setAdditionalSearchPath(self.asset_files_path)

        # set up the scene into the initial state
        self.reset()

        # tmp section for quick testing, delete after complete implemetation
        #input("waiting for button press")
        #self.close()

    ###############
    # Gym methods #
    ###############

    def reset(self):

        self.obj_ids = { 'fixed': [], 'rigid': [] }
        pyb.resetSimulation(pyb.RESET_USE_DEFORMABLE_WORLD)
        pyb.setGravity(0, 0, -9.8)

        # disable rendering for performance, becomes especially relevant if reset is called over and over again
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)

        # load ground plane to hold objects in place
        pyb.loadURDF("workspace/plane.urdf", basePosition=[0, 0, -0.001])

        # TODO: load in welding table

        # load robot arm and set it to its default pose
        # info: the welding torch is contained within the urdf file of the robot
        # I tried to do this via a constraint connecting the end effector link with the torch loaded in as a separate urf
        # this works, but one cannot then use the pybullet inverse kinematics method for the the tip of the torch
        if self.tool:
            self.robot = pyb.loadURDF("ur5/ur5_mrw510.urdf")  # UR5 robot for the moment, TODO: replace with actual welding robot
        else:
            self.robot = pyb.loadURDF("ur5/ur5_tand_gerad.urdf")
        joints = [pyb.getJointInfo(self.robot, i) for i in range(pyb.getNumJoints(self.robot))]
        self.joints = [j[0] for j in joints if j[2] == pyb.JOINT_REVOLUTE]
        for i in range(len(self.joints)):
            pyb.resetJointState(self.robot, self.joints[i], self.resting_pose_angles[i])

        # turn on rendering again
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)

        """
        # code to manually control the robot in real time
        rollId = pyb.addUserDebugParameter("roll", -1.5, 1.5, 0)
        pitchId = pyb.addUserDebugParameter("pitch", -1.5, 1.5, 0)
        yawId = pyb.addUserDebugParameter("yaw", -1.5, 1.5, 0)
        fwdxId = pyb.addUserDebugParameter("fwd_x", -1, 1, 0)
        fwdyId = pyb.addUserDebugParameter("fwd_y", -1, 1, 0)
        fwdzId = pyb.addUserDebugParameter("fwd_z", -1, 1, 0)
        yaw = 0
        
        while yaw!=-1.5:
            roll = pyb.readUserDebugParameter(rollId)
            pitch = pyb.readUserDebugParameter(pitchId)
            yaw = pyb.readUserDebugParameter(yawId)
            x = pyb.readUserDebugParameter(fwdxId)
            y = pyb.readUserDebugParameter(fwdyId)
            z = pyb.readUserDebugParameter(fwdzId)

            self.movep(([x,y,z],pyb.getQuaternionFromEuler([roll,pitch,yaw])))

        """

        obs, _, _, _ = self.step()  # return an observation of the environment without any actions taken

        return obs

    def _get_obs(self):

        tmp = pyb.getLinkState(
                            bodyUniqueId=self.robot,
                            linkIndex=10,
                            computeForwardKinematics=True  # need to check if this is necessary, if not can be turned off for performance gain
            )
        return {
            'position': tmp[0],  # index 0 is linkWorldPosition
            'rotation': tmp[1]  # index 1 is linkWorldOrientation as quaternion
        }       

    def close(self):

        pyb.disconnect()

    #################################################
    # methods for dealing with pybullet environment #
    #################################################

    def switch_tool(self, tool, reset=False):
        """
        Switches out the welding torch, but only if the robot is very close to default configuration.
        Does nothing if desired tool is already attached.

        Args:
            tool: int, 0 for TAND GERAD, 1 for MRW510
            reset: bool, set to true if the self.reset() method is called right after, will prevent this method from reloading the robot
                   because reset() is going to do that anyway

        Returns:
            True if switching completed, False if robot not in proper position or desired tool already attached
        """
        if tool == self.tool:
            return False
        else:
            self.tool = tool

            # check if current joint state is suffieciently close to resting state
            currj = [pyb.getJointState(self.robot, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = self.resting_pose_angles - currj
            if not all(np.abs(diffj) < 1e-2):
                return False
            else:
                if not reset:  # don't reload the robot model to save performance when reset() is going to do it anyway
                    return True
                pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
                pyb.removeBody(self.robot)
                if self.tool:
                    self.robot = pyb.loadURDF("ur5/ur5_mrw510.urdf")
                else:
                    self.robot = pyb.loadURDF("ur5/ur5_tand_gerad.urdf")
                joints = [pyb.getJointInfo(self.robot, i) for i in range(pyb.getNumJoints(self.robot))]
                self.joints = [j[0] for j in joints if j[2] == pyb.JOINT_REVOLUTE]
                for i in range(len(self.joints)):
                    pyb.resetJointState(self.robot, self.joints[i], self.resting_pose_angles[i])
                pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)

                return True




    def _perform_action(self, action):

        if action is not None:
            timeout = self.movep((action["translate"], action["rotate"]))

            if timeout:
                return self._get_obs(), 0.0, True, None
        
        while not self.is_static:
            pyb.stepSimulation()

    def calc_reward(self):

        # idea: this method gets information about current welding part and the weld seam from the agent that is acting in the environment
        # then calculate reward based on this and the environment state

        return 0.0, None

    def is_done(self):

        return False
        return True

    # methods taken almost 1:1 from ravens code, need to add proper attribution later TODO
    def movej(self, targj, speed=0.01, timeout=0.5):
        """Move UR5 to target joint configuration."""

        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [pyb.getJointState(self.robot, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            pyb.setJointMotorControlArray(
                bodyIndex=self.robot,
                jointIndices=self.joints,
                controlMode=pyb.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            pyb.stepSimulation()
        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True 

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""

        targj = self.solve_ik(pose)
        return self.movej(targj, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""

        joints = pyb.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=10,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.resting_pose_angles).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""

        v = [np.linalg.norm(pyb.getBaseVelocity(i)[0])
            for i in self.obj_ids['rigid']]
        return all(np.array(v) < 5e-3)

    def add_object(self, urdf, pose, category='rigid'):
        """List of (fixed, rigid, or deformable) objects in env."""

        fixed_base = 1 if category == 'fixed' else 0
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
        obj_id = pyb.loadURDF(
            urdf,
            pose[0],  # xyz
            pose[1],  # xyzw quaternion
            useFixedBase=fixed_base)
        self.obj_ids[category].append(obj_id)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
        return obj_id

if __name__ == "__main__":
    e = WeldingEnvironmentPybullet("../assets/",True)