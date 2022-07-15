from cv2 import rotate
import gym
import pybullet as pyb
import time
import numpy as np
from util.util import matrix_to_quaternion, rpy_to_quaternion, quaternion_to_rpy, quaternion_multiply, quaternion_invert
from collections import OrderedDict

class WeldingEnvironment(gym.Env):

    def __init__(self, 
                agent,
                relative_movement=False):
        
        self._random_state = None
        self._relative_movement = relative_movement

        # variables needed by Gym env subclasses, set by method to be implemented by subclasses
        self._init_gym_vars()

        # agent, needs to be set after construction due to mutual dependence
        self.agent = agent
        self.agent._set_env(self)

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

        timeout = self._perform_action(action)
        if timeout:
            return self._get_obs(), 0, False, None
        reward, info = self.calc_reward() if action is not None else (0, {})
        done = self.is_done()

        return self._get_obs(), reward, done, info

    # helper method

    def perform_action(self):

        raise NotImplementedError

    def _init_gym_vars(self):

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
                agent,
                asset_files_path,
                relative_movement=False
                ):

        self.asset_files_path = asset_files_path
        super().__init__(agent, relative_movement)

class WeldingEnvironmentPybullet(WeldingEnvironment):

    def __init__(self,
                agent,
                asset_files_path,
                display=False,
                hz=240,
                robot="ur5",
                relative_movement=False):

        super().__init__(agent, relative_movement)

        self.asset_files_path = asset_files_path
        self.obj_ids = {'fixed': [], 'rigid': []}  # dict of objects by type of body dynamics
        self.tool = 0  # 1: TAND GERAD, 0: MRW510
        if robot in ["ur5","kr6","kr16"]:
            self.robot_name = robot
        else:
            raise ValueError("Robot model not supported")

        self._init_settings()  # method to clean up the constructor, sets a bunch of class variables with hardoced values used for many calculations
        
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
        
        base_quat = [0,0,0,1]
        #pyb.loadURDF("test.urdf", useFixedBase=True, baseOrientation=base_quat)
        pose_quat = [ -0.7071068, 0.7071068, 0.000139, -0.000139 ]
        rotate_quat = [ 0.9226898, 0, 0, 0.3855431 ]
        end_quat = quaternion_multiply(pose_quat, rotate_quat)
        end_quat = quaternion_multiply(base_quat, end_quat)
        #pyb.loadURDF("test.urdf", useFixedBase=True, baseOrientation=end_quat)
        pyb.loadURDF("test.urdf", useFixedBase=True, baseOrientation=[-0.27059805, -0.27059805,  0.65328148,  0.65328148])

        #pyb.loadURDF("test.urdf", useFixedBase=True, baseOrientation=[-0.1765669, -0.4233322, 0.8201255, 0.3420647])
        pyb.loadURDF("test.urdf", useFixedBase=True)

        # TODO: load in welding table

        # load robot arm and set it to its default pose
        # info: the welding torch is contained within the urdf file of the robot
        # I tried to do this via a constraint connecting the end effector link with the torch loaded in as a separate urf
        # this works, but one cannot then use the pybullet inverse kinematics method for the the tip of the torch
        # because it relies on using a link within the robot urdf
        if not self.tool:
            self.robot = pyb.loadURDF(self.robot_name+"/"+self.robot_name+"_mrw510.urdf", useFixedBase=True, basePosition=[0, 0, self.fixed_height[self.robot_name]], baseOrientation=pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
        else:
            self.robot = pyb.loadURDF(self.robot_name+"/"+self.robot_name+"_tand_gerad.urdf", useFixedBase=True, basePosition=[0, 0, self.fixed_height[self.robot_name]], baseOrientation=pyb.getQuaternionFromEuler([np.pi, 0., 0.]))

        joints = [pyb.getJointInfo(self.robot, i) for i in range(pyb.getNumJoints(self.robot))]
        self.joints = [j[0] for j in joints if j[2] == pyb.JOINT_REVOLUTE]
        for i in range(len(self.joints)):
            pyb.resetJointState(self.robot, self.joints[i], self.resting_pose_angles[self.robot_name][i])

        # turn on rendering again
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
        
        obs, _, _, _ = self.step()  # return an observation of the environment without any actions taken

        return obs

    def _get_obs(self):

        tmp = pyb.getLinkState(
                            bodyUniqueId=self.robot,
                            linkIndex=self.end_effector_link_id[self.robot_name],
                            computeForwardKinematics=True  # need to check if this is necessary, if not can be turned off for performance gain
            )
        tmp2 = pyb.getLinkState(
                            bodyUniqueId=self.robot,
                            linkIndex=self.base_link_id[self.robot_name],
                            computeForwardKinematics=True  # need to check if this is necessary, if not can be turned off for performance gain
            )
        return {
            'position': np.array(tmp[0]),  # index 4 is worldLinkWorldPosition,
            'position_base':np.array(tmp2[4][:2]),  # only xy position of baselink
            #'rotation': pyb.multiplyTransforms([0,0,0],np.array(tmp[5]), [0,0,0], pyb.getQuaternionFromEuler(np.array((2.35, 0, 0))))[1]  # index 5 is worldLinkWorldOrientation as quaternion
            'rotation': np.array(tmp[1])
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
            diffj = self.resting_pose_angles[self.robot_name] - currj
            if not all(np.abs(diffj) < 1e-2):
                return False
            else:
                if not reset:  # don't reload the robot model to save performance when reset() is going to do it anyway
                    return True
                pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
                pyb.removeBody(self.robot)
                if not self.tool:
                    self.robot = pyb.loadURDF(self.robot_name+"/"+self.robot_name+"_mrw510.urdf", useFixedBase=True, basePosition=[0, 0, self.fixed_height[self.robot_name]], baseOrientation=pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
                else:
                    self.robot = pyb.loadURDF(self.robot_name+"/"+self.robot_name+"_tand_gerad.urdf", useFixedBase=True, basePosition=[0, 0, self.fixed_height[self.robot_name]], baseOrientation=pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
                joints = [pyb.getJointInfo(self.robot, i) for i in range(pyb.getNumJoints(self.robot))]
                self.joints = [j[0] for j in joints if j[2] == pyb.JOINT_REVOLUTE]
                for i in range(len(self.joints)):
                    pyb.resetJointState(self.robot, self.joints[i], self.resting_pose_angles[self.robot_name][i])
                pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)

                return True




    def _perform_action(self, action):

        if action is not None:
            # if relative movement is enabled, action must be transformed into absolute movement needed for robot control...
            state = self._get_obs()
            new_state = OrderedDict()
            if self._relative_movement:
                # unfortunately, the order of dict entries matters to the gym contains() method here
                # if somehow the order dict entries in the observation space ordereddict changes, then the order of the next lines defining the entries of the new state needs to be switched as well
                new_state["position_base"] = state["position_base"] + action["translate_base"]
                new_state["position"] = state["position"] + action["translate"]
                new_state["rotation"] = state["rotation"] + action["rotate"]
                """
                print("state")
                print(state)
                print("action")
                print(action)
                print("new state")
                print(new_state)
                print("enthalten")
                print(self.observation_space.contains(new_state))
                """                
                if not self.observation_space.contains(new_state):
                    return False  # if the current state+action results in invalid state, return false and do nothing
                
                # convert rpy to quaternion for pybullet processing
                #new_state["rotation"] = pyb.multiplyTransforms([0,0,0], pyb.invertTransform([0,0,0], pyb.getQuaternionFromEuler(np.array((2.35, 0, 0))))[1], [0,0,0], pyb.getEulerFromQuaternion(new_state["rotation"]))[1]
                new_state["rotation"] = quaternion_multiply(pyb.getQuaternionFromEuler(np.array((2.35, 0, 0))), pyb.getQuaternionFromEuler(new_state["rotation"]))
            # ....otherwise use it as is
            else:
                new_state["position_base"] = action["translate_base"]
                new_state["position"] = action["translate"]
                new_state["rotation"] = rpy_to_quaternion(action["rotate"])

            # first move the base of the robot...(but only if the new location is sufficiently different from the old one)
            if np.linalg.norm(new_state["position_base"]-state["position_base"]) > 1e-4:
                pyb.resetBasePositionAndOrientation(self.robot, np.append(new_state["position_base"], self.fixed_height[self.robot_name]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
            # ...then the joints
            timeout = self.movep((new_state["position"], new_state["rotation"]))
            if timeout:
                return timeout
        
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
    def movej(self, targj, speed=0.05, timeout=0.5):
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
            endEffectorLinkIndex=self.end_effector_link_id[self.robot_name],
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=self.joints_lower[self.robot_name],
            upperLimits=self.joints_upper[self.robot_name],
            jointRanges=self.joints_range[self.robot_name],
            restPoses=np.float32(self.resting_pose_angles[self.robot_name]).tolist(),
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

    ###################
    # utility methods #
    ###################

    def _init_settings(self):
        """
        Sets a number of class variables containing constants for various calculations.
        Put into this method to clean up the constructor.
        """

        # angles for the default pose of the robot, found by trial and error
        self.resting_pose_angles = {  
            "ur5": np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi,
            "kr16": np.array([0, -0.5, 0.5, -1, 0.5, 0.5]) * np.pi,
            "kr6": np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi # copied from ur5, needs to be adjusted
        }

        # end effector link id
        self.end_effector_link_id = {
            "ur5": 10,
            "kr16": 7,  #subject to change, need to add invisible link for tool tip
            "kr6": 6  # needs confirmation
        }

        # base link id
        self.base_link_id =  {
            "ur5": None,  # tbd
            "kr16": 8,
            "kr6": 8  # tbd
        }

        # joint limits and ranges, needed for inverse kinematics
        self.joints_lower = {
            "ur5": [-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            "kr16": [-3.228858, -3.228858, -2.408553, -6.108651, -2.26891, -6.108651],
            "kr6": []
        }

        self.joints_upper = {
            "ur5": [-np.pi / 2, 0, 17, 17, 17, 17],
            "kr16": [3.22885911, 1.13446401, 3.0543261, 6.10865238, 2.26892802, 6.1086523],
            "kr6": []
        }

        self.joints_range = {
            "ur5": [np.pi, 2.3562, 34, 34, 34, 34],
            "kr16": list(np.array(self.joints_upper["kr16"])-np.array(self.joints_lower["kr16"])),
            "kr6": []
        }

        self.fixed_height = {
            "ur5": 2, #tbd
            "kr16": 1.8,
            "kr6": 2 #tbd
        }

    def _init_gym_vars(self):

        #   contains the position (as xyz) and rotation (as roll-pitch-yaw rpy in radians) of the end effector (i.e. the welding torch) in workspace
        min_position = np.array([-4., -4., 0.05])  # provisional
        max_position = np.array([4., 4, 1])
        min_rotation = np.array([-1, -1, -1, -1]) * np.pi/180
        max_rotation = min_rotation * (-1)

        self.observation_space = gym.spaces.Dict(
            {
                'position': gym.spaces.Box(low=min_position, high=max_position, shape=(3,), dtype=np.float32),
                'base_position': gym.spaces.Box(low=min_position[:2], high=max_position[:2], shape=(2,), dtype=np.float32),
                'rotation': gym.spaces.Box(low=min_rotation, high=max_rotation, shape=(4,), dtype=np.float32)
            }
        )
        
        # actions consist of translating and rotating the end effector
        # if relative_movement is true, then actions consists of additional movements
        # if it is false, then they consist of positions to be reached
        if self._relative_movement:
            min_position = np.array([-0.01, -0.01, -0.01])  # provisional
            max_position = -1 * min_position
            min_rotation = np.array([-0.001, -0.001, -0.001, -0.001]) * np.pi/180
            max_rotation = np.array([0.001, 0.001, 0.001, 0.001]) * np.pi/180

        self.action_space = gym.spaces.Dict(
            {
                'translate': gym.spaces.Box(low=min_position, high=max_position, shape=(3,), dtype=np.float32),
                'translate_base': gym.spaces.Box(low=min_position[:2], high=max_position[:2], shape=(2,), dtype=np.float32),
                'rotate': gym.spaces.Box(low=min_rotation, high=max_rotation, shape=(4,), dtype=np.float32)
            }
        )

    def manual_control(self):
        # code to manually control the robot in real time
        qx = pyb.addUserDebugParameter("qx", -1.5, 1.5, 0)
        qy = pyb.addUserDebugParameter("qy", -1.5, 1.5, 0)
        qz = pyb.addUserDebugParameter("qz", -1.5, 1.5, 0)
        qw = pyb.addUserDebugParameter("qw", -1.5, 1.5, 1)
        fwdxId = pyb.addUserDebugParameter("fwd_x", -4, 4, 0)
        fwdyId = pyb.addUserDebugParameter("fwd_y", -4, 4, 0)
        fwdzId = pyb.addUserDebugParameter("fwd_z", 0, 4, 0.5)
        fwdxIdbase = pyb.addUserDebugParameter("fwd_x_base", -4, 4, 0)
        fwdyIdbase = pyb.addUserDebugParameter("fwd_y_base", -4, 4, 0)
        x_base = 0
        y_base = 0
        oldybase = 0
        oldxbase = 0

        while True:
            if x_base != oldxbase or y_base != oldybase:
                pyb.resetBasePositionAndOrientation(self.robot, np.array([x_base, y_base, self.fixed_height[self.robot_name]]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
            qxr = pyb.readUserDebugParameter(qx)
            qyr = pyb.readUserDebugParameter(qy)
            qzr = pyb.readUserDebugParameter(qz)
            qwr = pyb.readUserDebugParameter(qw)
            x = pyb.readUserDebugParameter(fwdxId)
            y = pyb.readUserDebugParameter(fwdyId)
            z = pyb.readUserDebugParameter(fwdzId)
            oldxbase = x_base
            oldybase = y_base
            x_base = pyb.readUserDebugParameter(fwdxIdbase)
            y_base = pyb.readUserDebugParameter(fwdyIdbase)

            command_quat = [qxr,qyr,qzr,qwr]
            #pose_quat = self._get_obs()["rotation"]
            joint_quat = [ 0.7071068, -0.7071068, 0, 0 ]
            #joint_quat = [ 0.7071068, -0.7071068, 0, 0 ]
            rotate_quat = [ 0.9226898, 0, 0, 0.3855431 ]
            #pose_quat = quaternion_invert(pose_quat)
            #rotate_quat = quaternion_invert(rotate_quat)
            #joint_quat = quaternion_invert(joint_quat)

            #print("joint quat")
            #q = quaternion_multiply(joint_quat, rotate_quat)
            #q =quaternion_multiply(quaternion_invert(joint_quat),rotate_quat)
            #q =quaternion_multiply(joint_quat, quaternion_invert(rotate_quat))
            #q =quaternion_multiply(quaternion_invert(joint_quat), quaternion_invert(rotate_quat)) #ganz gut
            #print("rotate quat")
            q=quaternion_multiply(rotate_quat, joint_quat) #gut
            #q=quaternion_multiply(quaternion_invert(rotate_quat),joint_quat) #schlecht
            #q=quaternion_multiply(quaternion_invert(rotate_quat), quaternion_invert(joint_quat)) #schlecht
            #q=quaternion_multiply(rotate_quat, quaternion_invert(joint_quat)) #gut
            #q=quaternion_multiply()

            #end_command_quat = quaternion_multiply(q, command_quat)
            #end_command_quat = quaternion_multiply(command_quat, quaternion_invert(q))
            #end_command_quat = quaternion_multiply(command_quat, q)
            #end_command_quat = quaternion_multiply(command_quat, quaternion_invert(q))
            #end_command_quat = quaternion_multiply(quaternion_invert(q), command_quat)

            end_command_quat = quaternion_multiply(quaternion_invert(joint_quat),command_quat)
            #end_command_quat = quaternion_multiply(quaternion_invert(rotate_quat), end_command_quat)
            #end_command_quat = quaternion_multiply(joint_quat,command_quat)
            #end_command_quat = quaternion_multiply(rotate_quat, end_command_quat)
            end_command_quat = quaternion_multiply(quaternion_invert(rotate_quat), end_command_quat)
            #end_command_quat = command_quat
            
            #print(joint_quat)
            #print(quaternion_multiply(joint_quat,command_quat))
            #print(end_command_quat)

            self.movep(([x,y,z],end_command_quat))
            print(self._get_obs())

if __name__ == "__main__":
    e = WeldingEnvironmentPybullet("../assets/",True)

#-0.27059805, -0.27059805,  0.65328148,  0.65328148