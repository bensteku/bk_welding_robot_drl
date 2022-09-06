import gym
import pybullet as pyb
import time
import numpy as np
from util.util import matrix_to_quaternion, quaternion_diff, quaternion_normalize, rpy_to_quaternion, quaternion_to_rpy, quaternion_multiply, quaternion_invert
from collections import OrderedDict

class WeldingEnvironment(gym.Env):

    def __init__(self, 
                agent):
        
        self._random_state = None

        # variables needed by Gym env subclasses, set by method to be implemented by subclasses
        self._init_gym_vars()

        # agent
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

        timeout = self._perform_action(action)
        
        obs = self._get_obs()

        reward, success, done = self.agent.reward(obs, timeout) if action is not None else (0, False, False)
        if success:
            self.agent.next_state()

        return obs, reward, done, success

    def close(self):

        raise NotImplementedError

    #################
    # other methods #
    #################

    def _perform_action(self):

        raise NotImplementedError

    def _init_gym_vars(self):

        raise NotImplementedError

    def is_done(self):

        raise NotImplementedError

    def set_agent(self, agent):

        self.agent = agent

class WeldingEnvironmentMOSES(WeldingEnvironment):

    def __init__(self,
                agent,
                asset_files_path):

        self.asset_files_path = asset_files_path
        super().__init__(agent)

class WeldingEnvironmentPybullet(WeldingEnvironment):

    def __init__(self,
                agent,
                asset_files_path,
                display=False,
                hz=240,
                robot="ur5"):

        super().__init__(agent)

        self.asset_files_path = asset_files_path
        self.obj_ids = []  # list of object ids
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

    ###############
    # Gym methods #
    ###############

    def reset(self):

        self.obj_ids = []
        pyb.resetSimulation(pyb.RESET_USE_DEFORMABLE_WORLD)
        pyb.setGravity(0, 0, -9.8)

        # disable rendering for performance, becomes especially relevant if reset is called over and over again
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)

        # load ground plane to hold objects in place
        plane_id = pyb.loadURDF("workspace/plane.urdf", basePosition=[0, 0, -0.001])
        self.obj_ids.append(plane_id)

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

        self.agent.state = 0
        
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
        return np.hstack([np.array(tmp2[4][:2]), np.array(tmp[0]), self._quat_ee_to_w(np.array(tmp[1])), self.get_joint_state()]) 

    def close(self):

        pyb.disconnect()

    #################################################
    # methods for dealing with pybullet environment #
    #################################################

    def get_joint_state(self):
        """
        Returns a list with the angles of the current robot configuration.
        """
        return np.array([pyb.getJointState(self.robot, i)[0] for i in self.joints])

    def set_joint_state(self, config):
        """
        Debug method for setting the joints of the robot to a certain configuration. Will kill all physics, momentum, movement etc. going on.
        """
        for i in range(len(self.joints)):
            pyb.resetJointState(self.robot, self.joints[i], config[i])

    def switch_tool(self, tool):
        """
        Switches out the welding torch, but only if the robot is very close to default configuration.
        Does nothing if desired tool is already attached.

        Args:
            tool: int, 1 for TAND GERAD, 0 for MRW510

        Returns:
            True if switching completed, False if robot not in proper position or desired tool already attached
        """
        if tool == self.tool:
            return False
        else:
            self.tool = tool

            # check if current joint state is sufficiently close to resting state
            currj = [pyb.getJointState(self.robot, i)[0] for i in self.joints]
            currj = np.array(currj)

            base_pos = self._get_obs()[:2]

            self.movej(self.resting_pose_angles[self.robot_name])
            
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
            pyb.removeBody(self.robot)
            if not self.tool:
                self.robot = pyb.loadURDF(self.robot_name+"/"+self.robot_name+"_mrw510.urdf", useFixedBase=True, basePosition=[base_pos[0], base_pos[1], self.fixed_height[self.robot_name]], baseOrientation=pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
            else:
                self.robot = pyb.loadURDF(self.robot_name+"/"+self.robot_name+"_tand_gerad.urdf", useFixedBase=True, basePosition=[base_pos[0], base_pos[1], self.fixed_height[self.robot_name]], baseOrientation=pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
            joints = [pyb.getJointInfo(self.robot, i) for i in range(pyb.getNumJoints(self.robot))]
            self.joints = [j[0] for j in joints if j[2] == pyb.JOINT_REVOLUTE]
            for i in range(len(self.joints)):
                pyb.resetJointState(self.robot, self.joints[i], self.resting_pose_angles[self.robot_name][i])
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)

            self.movej(currj)

            return True

    def is_in_collision(self):
        """
        Returns whether there is a collision anywhere between the robot and the specified object
        """
        pyb.performCollisionDetection()  # perform just the collision detection part of the PyBullet engine
        col = False
        for obj in self.obj_ids:
            if len(pyb.getContactPoints(self.robot, obj)) > 0:
                col = True 
                break
        return col

    def config_is_in_collision(self, config):
        """
        Returns whether there is a collision anywhere if the robot is in the specified configuration
        """
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
        currj = [pyb.getJointState(self.robot, i)[0] for i in self.joints]
        for joint, val in zip(self.joints, config):
            pyb.resetJointState(self.robot, joint, val)    
        col = self.is_in_collision()
        for joint, val in zip(self.joints, currj):
            pyb.resetJointState(self.robot, joint, val)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
        return col


    def _perform_action(self, action):

        if action is not None:
            state = self._get_obs()
            # unfortunately, the order of dict entries matters to the gym contains() method here
            # if somehow the order of dict entries in the observation space OrderedDict changes, then the order of the next lines defining the entries of the new state needs to be switched as well
            new_base_position = state[:2] + action[:2]
            new_ee_position = state[2:5] + action[2:5]
            current_rotation_as_rpy = pyb.getEulerFromQuaternion(state[5:9])
            new_rotation_as_rpy = np.array([entry + action[:5][idx] for idx, entry in enumerate(current_rotation_as_rpy)])
            new_rotation_as_quaternion = pyb.getQuaternionFromEuler(new_rotation_as_rpy)
            new_rotation_as_quaternion_in_correct_frame = self._quat_w_to_ee(new_rotation_as_quaternion)         
            # TODO: reimplement the valid bounds check once the bounds have actually been settled on sometime in the future

            # first move the base of the robot...(but only if the new location is sufficiently different from the old one to prevent constant)
            if np.linalg.norm(new_base_position - state[:2]) > 1e-4:
                pyb.resetBasePositionAndOrientation(self.robot, np.append(new_base_position, self.fixed_height[self.robot_name]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
            # ...then the joints
            timeout = self.movep((new_ee_position, new_rotation_as_quaternion_in_correct_frame))
            if timeout:
                return timeout
        
        while not self.is_static:
            pyb.stepSimulation()

    # methods taken almost 1:1 from ravens code, need to add proper attribution later TODO
    def movej(self, targj, speed=0.05, timeout=0.025, use_dynamics=False):
        """
        Move UR5 to target joint configuration.
        """
        if use_dynamics:
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
        else:
            currj = [pyb.getJointState(self.robot, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            while any(np.abs(diffj) > 1e-2):
                # Move with constant velocity
                norm = np.linalg.norm(diffj)
                if norm > speed:
                    v = diffj / norm
                    stepj = currj + v * speed
                else:
                    stepj = currj + diffj
                self.set_joint_state(stepj)
                #pyb.stepSimulation()

                currj = stepj
                diffj = targj - currj
            return False

    def movep(self, pose, speed=0.01):
        """
        Move UR5 to target end effector pose.
        """

        targj = self.solve_ik(pose)
        return self.movej(targj, speed)

    def solve_ik(self, pose):
        """
        Calculate joint configuration with inverse kinematics.
        """

        joints = pyb.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self.end_effector_link_id[self.robot_name],
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=self.joints_lower[self.robot_name],
            upperLimits=self.joints_upper[self.robot_name],
            jointRanges=self.joints_range[self.robot_name],
            restPoses=np.float32(self.resting_pose_angles[self.robot_name]).tolist(),
            maxNumIterations=2000,
            residualThreshold=5e-3)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def solve_fk(self, config):
        """
        Gets the robot pose associated with a certain configuration.
        Involves actually setting the robot to the desired spot and then reading the ee state,
        apparently there is no more elegant way as Pybullet does not expose the necessary matrices.
        """
        currj = [pyb.getJointState(self.robot, i)[0] for i in self.joints]
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)  # turn off rendering
        for i in range(len(self.joints)):
            pyb.resetJointState(self.robot, self.joints[i], config[i])
        state = pyb.getLinkState(
                            bodyUniqueId=self.robot,
                            linkIndex=self.end_effector_link_id[self.robot_name],
                            computeForwardKinematics=True)
        for i in range(len(self.joints)):
            pyb.resetJointState(self.robot, self.joints[i], currj[i])
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)  # turn rendering back on
        return (np.array(state[0]), np.array(state[1]))

    @property
    def is_static(self):
        """
        Return true if objects are no longer moving.
        """

        v = [np.linalg.norm(pyb.getBaseVelocity(i)[0])
            for i in self.obj_ids]
        return all(np.array(v) < 5e-3)

    def add_object(self, urdf, pose):
        """
        Add objects to env.
        """
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
        obj_id = pyb.loadURDF(
            urdf,
            pose[0],  # xyz
            pose[1],  # xyzw quaternion
            useFixedBase=True)
        self.obj_ids.append(obj_id)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
        return obj_id

    def _quat_w_to_ee(self, quat):
        """
        Takes an input quaternion in the world frame, rotates it such that it offsets the rotation of the welding torch
        (meaning that a [0,0,0,1] input quaternion for the robot arm ee results in the same pose of the torch as loading it with the loadURDF method) 
        and transforms it into the end effector frame (meaning that that inputs result in the correct rotation w.r.t the world frame)
        """

        # get offset
        offset = self.ik_offset_angles[self.robot_name][self.tool]

        # rotate the user input by the offset (the torch will now be positioned like when its original mesh is loaded by the loadURDF method if input is [0, 0, 0, 1])
        middle_quat = quaternion_multiply(offset, quat)
        # however, the coordinate system is still wrongly aligned, so we will have to switch systems by multiplying through the offset
        # this will make it so that our input (command_quat) rotates around the axes of the world coordinate system instead of the world axes rotated by the offset
        offset = quaternion_invert(offset)
        res = quaternion_multiply(offset, middle_quat)
        res = quaternion_multiply(res, quaternion_invert(offset))

        return res

    def _quat_ee_to_w(self, quat):
        """
        Same as above, just from end effector frame to world frame.
        """

        offset = self.ik_offset_angles[self.robot_name][self.tool]

        tmp = quaternion_multiply(offset, quat)
        tmp = quaternion_multiply(tmp, quaternion_invert(offset))

        res = quaternion_multiply(quaternion_invert(offset), tmp)

        return res


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
            "kr16": np.array([0, -0.5, 0.75, -1, 0.5, 0.5]) * np.pi,
            "kr6": np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi # copied from ur5, needs to be adjusted
        }

        # end effector link id
        self.end_effector_link_id = {
            "ur5": 10,
            "kr16": 7,
            "kr6": 6  # tbd
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
            "kr16": 2,
            "kr6": 2 #tbd
        }

        # angles used to transform the coordinate system of the end effector for usage in inverse kinematics
        # such that the attached torch is positioned at [0, 0, 0, 1] exactly as it would be if loaded into the world separately
        # this has the effect that the ground truth rotations from the xml files can be used as inputs without changes
        # first array: MRW510, second array: TAND GERAD
        self.ik_offset_angles = {
            "ur5": [[0, 0, 0, 1], [0, 0, 0, 1]],  # tbd
            "kr16": [[-0.2726201, 0.2726201, -0.6524402, -0.6524402], [-0.0676347, 0.0676347, -0.7038647, -0.7038647]],
            "kr6": [[0, 0, 0, 1], [0, 0, 0, 1]]  # tbd
        }

    def _init_gym_vars(self):

        # contains the position (as xyz) and rotation (as quaternion) of the end effector (i.e. the welding torch) in world frame
        min_position = np.array([-0.2, -0.2, 0.001])  # provisional
        max_position = np.array([6., 6., 1.25])
        min_position_base = np.array([-0.2, -0.2]) 
        max_position_base = np.array([6., 6.])
        min_rotation = np.array([-1, -1, -1]) * np.pi * 2
        max_rotation = min_rotation * (-1)
        self.pos_speed = 0.01  # provisional
        self.base_speed = 10 * self.pos_speed

        self.observation_space = gym.spaces.Dict(
            {
                'position': gym.spaces.Box(low=min_position, high=max_position, shape=(3,), dtype=np.float32),
                'base_position': gym.spaces.Box(low=min_position_base, high=max_position_base, shape=(2,), dtype=np.float32),
                'rotation': gym.spaces.Box(low=min_rotation, high=max_rotation, shape=(3,), dtype=np.float32)
            }
        )
        
        # actions consist of marginal (base-)translations and rotations
        # indices 0-1:base, 2-4: ee, 5-7: ee rotation in rpy
        min_action = np.array([-1, -1, -1, -1, -1, -1, -1, -1 ]) #tbd
        max_action = min_action * -1

        self.action_space = gym.spaces.Box(low=min_action, high=max_action, shape=(8,), dtype=np.float32)

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

        pyb.addUserDebugLine([0,0,0],[0,0,1],[0,0,1],parentObjectUniqueId=self.robot, parentLinkIndex= self.end_effector_link_id[self.robot_name])
        pyb.addUserDebugLine([0,0,0],[0,1,0],[0,1,0],parentObjectUniqueId=self.robot, parentLinkIndex= self.end_effector_link_id[self.robot_name])
        pyb.addUserDebugLine([0,0,0],[1,0,0],[1,0,0],parentObjectUniqueId=self.robot, parentLinkIndex= self.end_effector_link_id[self.robot_name])

        while True:
            if x_base != oldxbase or y_base != oldybase:
                pyb.resetBasePositionAndOrientation(self.robot, np.array([x_base, y_base, self.fixed_height[self.robot_name]]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
            # read inputs from GUI
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

            # build quaternion from user input
            command_quat = [qxr,qyr,qzr,qwr]
            command_quat = self._quat_w_to_ee(command_quat)

            self.movep(([x,y,z],command_quat))

class WeldingEnvironmentPybulletConfigSpace(WeldingEnvironmentPybullet):

    def __init__(self,
                agent,
                asset_files_path,
                display=False,
                hz=240,
                robot="ur5"):

        super().__init__(agent, asset_files_path, display, hz, robot)

    def _perform_action(self, action):
        if action is not None:
            state = self._get_obs()
            new_base_position = state[:2] + action[:2]
            new_joints = action[2:]        

            # first move the base of the robot...(but only if the new location is sufficiently different from the old one to prevent constant)
            if np.linalg.norm(new_base_position - state[0]) > 1e-4:
                pyb.resetBasePositionAndOrientation(self.robot, np.append(new_base_position, self.fixed_height[self.robot_name]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
            # ...then the joints
            timeout = self.movej(new_joints)
            if timeout:
                return timeout
        
        while not self.is_static:
            pyb.stepSimulation()

        