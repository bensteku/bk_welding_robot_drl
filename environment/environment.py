import gym
import pybullet as pyb
import time
import numpy as np
from util.util import quaternion_multiply, quaternion_invert, suppress_stdout, matrix_to_quaternion, exp_decay_alt, quaternion_similarity, quaternion_to_matrix
from util import xml_parser
import os
from collections import OrderedDict

class WeldingEnvironmentPybullet(gym.Env):

    def __init__(self,
                agent,
                asset_files_path,
                display=False,
                hz=240,
                robot="ur5"):

        self.asset_files_path = asset_files_path
        self.obj_ids = []  # list of object ids
        self.tool = 0  # 1: TAND GERAD, 0: MRW510
        if robot in ["ur5","kr6","kr16"]:
            self.robot_name = robot
        else:
            raise ValueError("Robot model not supported")

        # method to clean up the constructor, sets a bunch of class variables with hardoced values used for many calculations, implemented by subclass
        self._init_settings()  
        # variables needed by Gym env subclasses, set by method to be implemented by subclasses
        self._init_gym_vars()

        # agent, used as an abstraction for the act method
        # could also simply be implemented as part of this class
        self.agent = agent
        self.agent._set_env(self)

        # path state variable, 0: moving ee down to weld seam, 1: welding, 2: moving ee back up
        self.path_state = 0

        # goals: overall collection of weldseams that are left to be dealt with
        # plan: expansion of intermediate steps containted within one weldseam
        # objective: the next part of the plan    
        self.goals = []
        self.objective = None
        self.plan = None
        
        # pybullet connection and setup
        disp = pyb.DIRECT  # direct <-> no gui, use for training
        if display:
            disp = pyb.GUI
        client = pyb.connect(disp)

        pyb.setTimeStep(1. / hz)
        pyb.setAdditionalSearchPath(self.asset_files_path)

        # get dataset of meshes
        self.dataset = self._register_data()

        # set index of welding mesh that is to be loaded in
        self.data_index = self.dataset["filenames"].index("201910204483_R1.urdf")  # TODO: replace by 0 and add a function for random choice from all indices after done signal was sent

        # set up the scene into the initial state
        self.reset()

    ###############
    # Gym methods #
    ###############

    def step(self, action=None):

        self._perform_action(action)
        
        obs = self._get_obs(False)

        reward, success, done = self.reward(obs) if action is not None else (0, False, False)
        if success:
            self.next_state()
        self.update_objectives()
        info = {
            "episode_counter": 0,  # TODO
            "is_success": success
        }

        #return self._normalize_obs(obs), reward, done, info
        return obs, reward, done, info

    def reset(self):

        self.obj_ids = []
        pyb.resetSimulation(pyb.RESET_USE_DEFORMABLE_WORLD)
        pyb.setGravity(0, 0, 0)

        # disable rendering for performance, becomes especially relevant if reset is called over and over again
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)

        # load in the meshes, the function here supresses the pointless pybullet warnings
        #with suppress_stdout():
        # load ground plane to hold objects in place
        plane_id = pyb.loadURDF("workspace/plane.urdf", basePosition=[0, 0, -0.001])
        self.obj_ids.append(plane_id)

        # load in the welding part and set goals
        self.load_object_into_env(self.data_index)

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

        # reset path state and get new plan and objective
        # also moves the base into position via the plan update in update_objectives()
        self.path_state = 0
        self.plan = None
        self.objective = None
        self.update_objectives()
        
        obs, _, _, _ = self.step()  # return an observation of the environment without any actions taken

        return obs

    def _get_obs(self, normalize=True):

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
        if self.objective:
            state = np.hstack((np.array(tmp2[4][:2]), np.array(tmp[0]), pyb.getEulerFromQuaternion(self._quat_ee_to_w(np.array(tmp[1]))), self.get_joint_state(), self.objective[0], self.objective[1][0], self.objective[1][1], [self.path_state])).astype(np.float64)
        else:
            state = np.hstack((np.array(tmp2[4][:2]), np.array(tmp[0]), pyb.getEulerFromQuaternion(self._quat_ee_to_w(np.array(tmp[1]))), self.get_joint_state(), [0, 0, 0], [1, 0, 0], [1, 0, 0], [self.path_state])).astype(np.float64)
        if normalize:
            state = self._normalize_obs(state)
        return state

    def _normalize_obs(self, obs):
        """
        Method to normalize the state such that the inputs for the NN are between -1 and 1
        Expects a numpy array, not a pytorch tensor
        """
        low = self.observation_space.low
        high = self.observation_space.high
        # both base position and ee position can be expressed as a percentage of their upper bound by simple division with it
        # this works as long as the lower bound stays at 0
        normalized_base_position = obs[:2] / high[:2]  # element-wise division
        normalized_ee_position = obs[2:5] / high[2:5]
        # the rpy values are projected to between -1 and 1 with max and min values being pi and -pi
        normalized_rpy = 2 * ((obs[5:8] + np.ones(3) * np.pi) / (np.ones(3) * 2 * np.pi)) - np.ones(3) # -(-)pi/(pi-(-)pi), standard lower-upper-bound formula
        # the joint values are normalized via the saved joint limits
        normalized_joints = 2 * ((obs[8:14] - self.joints_lower[self.robot_name]) / (self.joints_range[self.robot_name])) - np.ones(6)
        # the objective position needs to be normalized via the upper bound as above with the ee, the norms are already normalized
        normalized_objective_position = obs[14:17] / high[2:5]
        # finally, the agent path state can be normalized by dividing by 2, as there are only 3 states
        normalized_agent_path_state = obs[23] / 2

        return np.hstack([normalized_base_position, normalized_ee_position, normalized_rpy, normalized_joints, normalized_objective_position, obs[17:23], normalized_agent_path_state])

    def close(self):

        pyb.disconnect()

    ####################################################################
    # methods for dealing with pybullet environment and the simulation #
    ####################################################################

    def _reward(self, obs=None):
        """
        Method for calculating the rewards.
        Args:
            - obs: env observation
        Returns:
            - reward as float number
            - boolean indicating if the objectives for the current state have been fulfilled
        """
        
        # bool flags for when the objective is achieved
        pos_done, rot_done = False, False
        # base reward
        reward = 0

        if self.path_state == 2:
            # state for moving the ee upwards after one line has been welded

            # simply measure distance up to safe height
            distance = np.linalg.norm(np.array([obs[2], obs[3], self.safe_height]) - obs[2:5]) 
            if distance < self.ee_pos_reward_thresh:
                reward += 1
                pos_done = True
                rot_done = True
            else:
                reward += exp_decay_alt(distance, 1, 2*self.ee_pos_reward_thresh)
                #reward += 10 - distance * 2.5
        else:
            # if the arm is in welding mode or moving to the start position for welding give out a reward in concordance to how far away it is from the desired position and how closely
            # it matches the ground truth rotation
            # if the robot is in a problematic configuration (collision or not reachable(timeout)) give out a negative reward
            objective_with_slight_offset = self.objective[0] + self.objective[1][0] * 0.01 + self.objective[1][1] * 0.01
            distance = np.linalg.norm(objective_with_slight_offset - obs[2:5]) 
            
            if distance < self.ee_pos_reward_thresh:
                reward += 1
                pos_done = True  # objective achieved
            else:
                reward += exp_decay_alt(distance, 1, 2*self.ee_pos_reward_thresh)
                #reward += 20 - distance * 2.5

            quat_sim = quaternion_similarity(self.objective[2], obs[5:9])    
            if quat_sim > 1-self.quat_sim_thresh:
                rot_done = True
            
            reward = reward * (quat_sim**0.5)  # take root of quaternion similarity to dampen its effect a bit

        # hand out penalties
        col = self.is_in_collision()
        if col and not (pos_done and rot_done):
            reward = -1
            pos_done = False
            rot_done = False
        elif col and (pos_done and rot_done):
            col = False 

        if pos_done and rot_done:
            self.objective = None

        

        success = (pos_done and rot_done) and not col
        done = (self.objective is None and len(self.plan)==0 and len(self.goals)==0) or col

        return reward, success, done

    def reward(self, obs):
        pos_done=False
        if self.path_state == 2:
            distance = np.linalg.norm(np.array([obs[2], obs[3], self.safe_height]) - obs[2:5]) 
        else:
            objective_with_slight_offset = self.objective[0] + self.objective[1][0] * 0.01 + self.objective[1][1] * 0.01
            objective_with_slight_offset = np.array([1.96945097e+00,  1.87400000e+00,  8.26572114e-01])
            distance = np.linalg.norm(objective_with_slight_offset - obs[2:5]) 
        if distance < self.ee_pos_reward_thresh:
            reward = 1
            pos_done = True  # objective achieved
        else:
            reward = -0.1 * distance
        col = self.is_in_collision()
        if col and not (pos_done):
            reward = -5
            pos_done = False

        if pos_done:
            self.objective = None

        success = pos_done and not col
        done = (self.objective is None and len(self.plan)==0 and len(self.goals)==0) or col

        return reward, success, done

    def next_state(self):
        """
        Method that iterates the state of the agent. Called from outside if certain goal conditions are fulfilled.
        """

        # if the robot is in state 0 or 1 an objective has been completed
        if self.path_state == 0 or self.path_state == 1:
            self.objective = None
        # if the robot is in state 1 and the plan is not empty yet, that means it should remain in state 1
        # because it has more linear welding steps to complete
        if self.path_state == 1 and len(self.plan) != 0:
            return self.path_state
        else:
            # otherwise it's in another state or it's in state 1 but there's currently no more welding to be done,
            # then increment state or wrap back around if in state 2
            self.path_state = self.path_state + 1 if self.path_state < 2 else 0

        return self.path_state

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

            base_pos = self._get_obs(False)[:2]

            self.movej(self.resting_pose_angles[self.robot_name])
            
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
            pyb.removeBody(self.robot)
            #with suppress_stdout():
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
            state = self._get_obs(False)
            # unfortunately, the order of dict entries matters to the gym contains() method here
            # if somehow the order of dict entries in the observation space OrderedDict changes, then the order of the next lines defining the entries of the new state needs to be switched as well
            new_ee_position = state[2:5] + action[:3]
            current_rotation_as_rpy = pyb.getEulerFromQuaternion(state[5:9])
            new_rotation_as_rpy = np.array([entry + action[3:][idx] for idx, entry in enumerate(current_rotation_as_rpy)])
            new_rotation_as_quaternion = pyb.getQuaternionFromEuler(new_rotation_as_rpy)
            new_rotation_as_quaternion_in_correct_frame = self._quat_w_to_ee(new_rotation_as_quaternion)         
            # TODO: reimplement the valid bounds check once the bounds have actually been settled on sometime in the future

            # based on the information above, move the joints
            timeout = self.movep((new_ee_position, new_rotation_as_quaternion_in_correct_frame))
            if timeout:
                return timeout
        
        while not self.is_static:
            pyb.stepSimulation()
    
    def _move_base(self, new_base_pos, dynamic=True, delay = 0):
        """
        Method for moving the robot base to different coordinates.
        Will return True if movement happened, False if not.
        """
        state = self._get_obs(False)
        pos_diff = new_base_pos - state[:2]
        pos_dist = np.linalg.norm(pos_diff)
        if pos_dist > 1e-4:
            if not dynamic:
                pyb.resetBasePositionAndOrientation(self.robot, np.append(new_base_pos, self.fixed_height[self.robot_name]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
                return True
            else:
                steps = int(pos_dist / self.base_speed)
                step = pos_diff * (self.base_speed / pos_dist)
                for i in range(steps):
                    pyb.resetBasePositionAndOrientation(self.robot, np.append(state[:2] + (i+1)*step, self.fixed_height[self.robot_name]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
                    if delay:
                        time.sleep(delay)
                pyb.resetBasePositionAndOrientation(self.robot, np.append(new_base_pos, self.fixed_height[self.robot_name]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
                return True
        return False

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
            maxNumIterations=50,
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
        #with suppress_stdout():
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

    def _register_data(self):
        """
        Scans URDF(obj) files in asset path and creates a list, associating file name with weld seams and ground truths.
        This can can later be used to load these objects into the simulation, see load_object_into_env method.
        """
        data_path = self.asset_files_path+"objects/"
        filenames = []
        frames = []
        for file in [file_candidate for file_candidate in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, file_candidate))]:
            if ".urdf" in file:
                if (os.path.isfile(os.path.join(data_path, file.replace(".urdf",".xml"))) and 
                os.path.isfile(os.path.join(data_path, file.replace(".urdf",".obj"))) ):
                    
                    frames.append(xml_parser.parse_frame_dump(os.path.join(data_path, file.replace(".urdf",".xml"))))
                    filenames.append(file)
                else:
                    raise FileNotFoundError("URDF file "+file+" is missing its associated xml or obj!")

        return { "filenames": filenames, "frames": frames }

    def load_object_into_env(self, index):
        """
        Method for loading an object into the simulation.
        Args:
            - index: index of the desired file in the dataset list
        """

        self.current_part_id = self.add_object(os.path.join(self.asset_files_path+"objects/", self.dataset["filenames"][index]), 
                                                    pose = (self.xyz_offset, [0, 0, 0, 1]))
        self._set_goals(index)


    def _set_goals(self, index):
        """
        Uses the dataset to load in the weldseams of the file indicated by index into the goals list.
        One element in goal array <=> one weldseam in the original xml

        Args:
            index: int, signifies index in the dataset array
        """

        self.goals = []
        frames = self.dataset["frames"][index]
        for frame in frames:
            tmp = {}
            tmp["weldseams"] = [ele["position"] * self.pybullet_scale_factor + self.xyz_offset for ele in frame["weld_frames"]]
            tmp["norm"] = [ele["norm"] for ele in frame["weld_frames"]]
            tmp["target_pos"] = [ele[:3,3] * self.pybullet_scale_factor + self.xyz_offset for ele in frame["pose_frames"]]
            tmp["target_rot"] = [matrix_to_quaternion(ele[:3,:3]) for ele in frame["pose_frames"]]
            tmp["tool"] = 1 if frame["torch"][3] == "TAND_GERAD_DD" else 0           
            self.goals.append(tmp)

    def _set_plan(self):
        """
        Sets the plan by extracting all intermediate steps from the foremost element in the goals list.
        Also moves the base such that the robot arm can fullfill the plan.
        """
        if self.goals:
            self.plan = []
            goal = self.goals.pop(0)
            target_pos_bas = np.average(goal["weldseams"], axis=0)[:2]
            for idx in range(len(goal["weldseams"])):
                tpl = (goal["weldseams"][idx], goal["norm"][idx], goal["target_rot"][idx], goal["tool"], target_pos_bas)
                self.plan.append(tpl)
            self._move_base(target_pos_bas)
            return True
        else:
            return False

    def _set_objective(self):
        """
        Gets the foremost step of the plan.
        """
        if self.plan:
            self.objective = self.plan.pop(0)
            return True
        else:
            return False

    def update_objectives(self):
        """
        Just a wrapper method for convenience
        """

        if not self.plan and not self.objective and self.path_state != 2:  # only create new plan if there's also no current objective
            self._set_plan()
        if not self.objective and self.path_state != 2:
            self._set_objective()
            tool = self.objective[3]
            self.switch_tool(tool)

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

        # movement speed of the robot arm and base
        self.pos_speed = 0.01  # provisional
        self.base_speed = 10 * self.pos_speed

        # offset at which the welding assets will be placed into the world
        self.xyz_offset = np.array((0, 0, 0.01)) 

        # reward variables
        # radius of the sphere around the goal position for the robot end effector in which a full reward will be given
        self.ee_pos_reward_thresh = 5e-2  # might need adjustment
        # threshold for quaternion similarity in reward function (see util/util.py method)
        self.quat_sim_thresh = 4e-2  # this probably too

        # pybullet id of the part the agent is currently dealing with
        self.current_part_id = None

        # height at which the ee is transported when not welding
        self.safe_height = 0.5

        # scale factor for the meshes
        self.pybullet_scale_factor = 0.0005

    def _init_gym_vars(self):

        # observation space and its limits
        # contains the position (as xyz) and rotation (as quaternion) of the end effector (i.e. the welding torch) in world frame
        min_position_base = np.array([0, 0]) 
        max_position_base = np.array([5., 5.])
        min_position = np.array([0, 0, 0])  
        max_position = np.array([5., 5., 2])
        min_rotation = np.array([-1, -1, -1]) * np.pi
        max_rotation = min_rotation * (-1)
        min_joints = self.joints_lower[self.robot_name]
        max_joints = self.joints_upper[self.robot_name]
        min_objective_position = min_position
        max_objective_position = max_position
        min_norms = np.zeros(6)
        max_norms = np.ones(6)
        min_agent_path_state = 0
        max_agent_path_state = 2

        low = np.hstack([min_position_base, min_position, min_rotation, min_joints, min_objective_position, min_norms, min_agent_path_state]).astype(np.float64)
        high = np.hstack([max_position_base, max_position, max_rotation, max_joints, max_objective_position, max_norms, max_agent_path_state]).astype(np.float64)

        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(24,), dtype=np.float32)
        
        # actions consist of marginal (base-)translations and rotations
        # indices 0-1:base, 2-4: ee, 5-7: ee rotation in rpy
        min_action = np.array([-1, -1, -1, -1, -1, -1 ]).astype(np.float64) / 20 #tbd
        max_action = min_action * -1

        self.action_space = gym.spaces.Box(low=min_action, high=max_action, shape=(6,), dtype=np.float64)

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
                robot="ur5",
                additive=False):

        super().__init__(agent, asset_files_path, display, hz, robot)
        self.additive = additive

    def _perform_action(self, action):
        if action is not None:
            new_joints = action       

            # move the joints
            if self.additive:
                timeout = self.movej(self.get_joint_state() + new_joints)
            else:
                timeout = self.movej(new_joints)
            if timeout:
                return timeout
        
        while not self.is_static:
            pyb.stepSimulation()
    
    def step(self, action=None):

        self._perform_action(action)
        
        obs = self._get_obs(False)

        reward, success, done = self.reward(obs) if action is not None else (0, False, False)
        if success:
            self.next_state()
        self.update_objectives()
        info = {
            "episode_counter": 0,  # TODO
            "is_success": success
        }

        return obs, reward, done, info

    def solve_ik(self, pose):
        """
        Calculate joint configuration with inverse kinematics.
        """

        joints = pyb.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self.end_effector_link_id[self.robot_name],
            targetPosition=pose[0],
            targetOrientation=pose[1],
            #lowerLimits=self.joints_lower[self.robot_name],
            #upperLimits=self.joints_upper[self.robot_name],
            #jointRanges=self.joints_range[self.robot_name],
            #restPoses=np.float32(self.resting_pose_angles[self.robot_name]).tolist(),
            maxNumIterations=1000000,
            residualThreshold=1e-4)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

        
class WeldingEnvironmentPybulletLidar(WeldingEnvironmentPybullet):

    def __init__(self,
                agent,
                asset_files_path,
                display=False,
                hz=240,
                robot="ur5"):
        super().__init__(agent, asset_files_path, display, hz, robot)
        
        # overwrite the observation space to extend it with lidar data
        min_position_base = np.array([0, 0]) 
        max_position_base = np.array([5., 5.])
        min_position = np.array([0, 0, 0])  
        max_position = np.array([5., 5., 2])
        min_rotation = np.array([-1, -1, -1]) * np.pi
        max_rotation = min_rotation * (-1)
        min_joints = self.joints_lower[self.robot_name]
        max_joints = self.joints_upper[self.robot_name]
        min_objective_position = min_position
        max_objective_position = max_position
        min_norms = np.zeros(6)
        max_norms = np.ones(6)
        min_agent_path_state = 0
        max_agent_path_state = 2
        min_lidar = np.zeros(10)
        max_lidar = np.ones(10)

        low = np.hstack([min_position_base, min_position, min_rotation, min_joints, min_objective_position, min_norms, min_agent_path_state, min_lidar]).astype(np.float64)
        high = np.hstack([max_position_base, max_position, max_rotation, max_joints, max_objective_position, max_norms, max_agent_path_state, max_lidar]).astype(np.float64)

        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(34,), dtype=np.float32)

    def _set_lidar_cylinder(self, ray_min=0.02, ray_max=0.4, ray_num_ver=6, ray_num_hor=12, render=False):
        ray_froms = []
        ray_tops = []
        inf = pyb.getLinkState(self.robot, self.end_effector_link_id[self.robot_name])
        frame = quaternion_to_matrix(inf[5])
        frame[0:3,3] = inf[4]
        ray_froms.append(np.matmul(np.asarray(frame),np.array([0.0,0.0,0.01,1]).T)[0:3].tolist())
        ray_tops.append(np.matmul(np.asarray(frame),np.array([0.0,0.0,ray_max,1]).T)[0:3].tolist())


        for angle in range(230, 270, 20):
            for i in range(ray_num_hor):
                z = -ray_max * np.sin(angle*np.pi/180)
                l = ray_max * np.cos(angle*np.pi/180)
                x_end = l*np.cos(2*np.pi*float(i)/ray_num_hor)
                y_end = l*np.sin(2*np.pi*float(i)/ray_num_hor)
                start = np.matmul(np.asarray(frame),np.array([0.0,0.0,0.01,1]).T)[0:3].tolist()
                end = np.matmul(np.asarray(frame),np.array([x_end,y_end,z,1]).T)[0:3].tolist()
                ray_froms.append(start)
                ray_tops.append(end)
        
        # set the angle of rays
        interval = -0.005
        
        for i in range(8):
            ai = i*np.pi/4
            for angle in range(ray_num_ver):    
                z_start = (angle)*interval-0.1
                x_start = ray_min*np.cos(ai)
                y_start = ray_min*np.sin(ai)
                start = np.matmul(np.asarray(frame),np.array([x_start,y_start,z_start,1]).T)[0:3].tolist()
                z_end = (angle)*interval-0.1
                x_end = ray_max*np.cos(ai)
                y_end = ray_max*np.sin(ai)
                end = np.matmul(np.asarray(frame),np.array([x_end,y_end,z_end,1]).T)[0:3].tolist()
                ray_froms.append(start)
                ray_tops.append(end)
        
        for angle in range(250, 270, 20):
            for i in range(ray_num_hor):
                z = -0.2+ray_max * np.sin(angle*np.pi/180)
                l = ray_max * np.cos(angle*np.pi/180)
                x_end = l*np.cos(np.pi*float(i)/ray_num_hor-np.pi/2)
                y_end = l*np.sin(np.pi*float(i)/ray_num_hor-np.pi/2)
                
                start = np.matmul(np.asarray(frame),np.array([x_start,y_start,z_start-0.1,1]).T)[0:3].tolist()
                end = np.matmul(np.asarray(frame),np.array([x_end,y_end,z,1]).T)[0:3].tolist()
                ray_froms.append(start)
                ray_tops.append(end)
        results = pyb.rayTestBatch(ray_froms, ray_tops)
       
        if render:
            hitRayColor = [0, 1, 0]
            missRayColor = [1, 0, 0]

            pyb.removeAllUserDebugItems()

            for index, result in enumerate(results):
                if result[0] == -1:
                    pyb.addUserDebugLine(ray_froms[index], ray_tops[index], missRayColor)
                else:
                    pyb.addUserDebugLine(ray_froms[index], ray_tops[index], hitRayColor)
        return results

    def _get_lidar_probe(self):
        lidar_results = self._set_lidar_cylinder()
        obs_rays = np.zeros(shape=(85,),dtype=np.float32)
        indicator = np.zeros((10,), dtype=np.float32)
        for i, ray in enumerate(lidar_results):
            obs_rays[i] = ray[2]
        rays_sum = []
        rays_sum.append(obs_rays[0:25])
        rays_sum.append(obs_rays[25:31])
        rays_sum.append(obs_rays[31:37])
        rays_sum.append(obs_rays[37:43])
        rays_sum.append(obs_rays[43:49])
        rays_sum.append(obs_rays[49:55])
        rays_sum.append(obs_rays[55:61])
        rays_sum.append(obs_rays[61:67])
        rays_sum.append(obs_rays[67:73])
        rays_sum.append(obs_rays[73:])
        for i in range(10):
            if rays_sum[i].min()>=0.99:
                indicator[i] = 0
            if 0.5<rays_sum[i].min()<0.99:
                indicator[i] = 1
            if rays_sum[i].min()<=0.5:
                indicator[i] = 2
        return indicator

    def _get_obs(self, normalize=False):

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
        lidar_results = self._get_lidar_probe()
        if self.objective:
            state = np.hstack((np.array(tmp2[4][:2]), np.array(tmp[0]), pyb.getEulerFromQuaternion(self._quat_ee_to_w(np.array(tmp[1]))), self.get_joint_state(), self.objective[0], self.objective[1][0], self.objective[1][1], [self.path_state], lidar_results)).astype(np.float64)
        else:
            state = np.hstack((np.array(tmp2[4][:2]), np.array(tmp[0]), pyb.getEulerFromQuaternion(self._quat_ee_to_w(np.array(tmp[1]))), self.get_joint_state(), [0, 0, 0], [1, 0, 0], [1, 0, 0], [self.path_state], lidar_results)).astype(np.float64)
        if normalize:
            state = self._normalize_obs(state)
        return state

    def _normalize_obs(self, obs):
        """
        Method to normalize the state such that the inputs for the NN are between -1 and 1
        Expects a numpy array, not a pytorch tensor
        """
        low = self.observation_space.low
        high = self.observation_space.high
        # both base position and ee position can be expressed as a percentage of their upper bound by simple division with it
        # this works as long as the lower bound stays at 0
        normalized_base_position = obs[:2] / high[:2]  # element-wise division
        normalized_ee_position = obs[2:5] / high[2:5]
        # the rpy values are projected to between -1 and 1 with max and min values being pi and -pi
        normalized_rpy = 2 * ((obs[5:8] + np.ones(3) * np.pi) / (np.ones(3) * 2 * np.pi)) - np.ones(3) # -(-)pi/(pi-(-)pi), standard lower-upper-bound formula
        # the joint values are normalized via the saved joint limits
        normalized_joints = 2 * ((obs[8:14] - self.joints_lower[self.robot_name]) / (self.joints_range[self.robot_name])) - np.ones(6)
        # the objective position needs to be normalized via the upper bound as above with the ee, the norms are already normalized
        normalized_objective_position = obs[14:17] / high[2:5]
        # finally, the agent path state can be normalized by dividing by 2, as there are only 3 states
        normalized_agent_path_state = obs[23] / 2
        # same goes for the lidar results
        normalized_lidar_results = obs[24:] / 2

        return np.hstack([normalized_base_position, normalized_ee_position, normalized_rpy, normalized_joints, normalized_objective_position, obs[17:23], normalized_agent_path_state, normalized_lidar_results])

class WeldingEnvironmentPybulletLidar2(WeldingEnvironmentPybulletLidar):

    def __init__(self,
                agent,
                asset_files_path,
                display=False,
                hz=240,
                robot="ur5"):
        super().__init__(agent, asset_files_path, display, hz, robot)
        
        # overwrite the observation space to extend it with lidar data
        min_position_base = np.array([0, 0]) 
        max_position_base = np.array([5., 5.])
        min_position = np.array([0, 0, 0])  
        max_position = np.array([5., 5., 2])
        min_rotation = np.array([-1, -1, -1]) * np.pi
        max_rotation = min_rotation * (-1)
        min_joints = self.joints_lower[self.robot_name]
        max_joints = self.joints_upper[self.robot_name]
        min_objective_position = min_position
        max_objective_position = max_position
        min_norms = np.zeros(6)
        max_norms = np.ones(6)
        min_agent_path_state = 0
        max_agent_path_state = 2
        min_lidar = np.zeros(10)
        max_lidar = np.ones(10)

        low = np.hstack([min_position_base, min_position, min_rotation, min_joints, min_objective_position]).astype(np.float64)
        high = np.hstack([max_position_base, max_position, max_rotation, max_joints, max_objective_position]).astype(np.float64)

        #self.observation_space = gym.spaces.Box(low=low, high=high, shape=(34,), dtype=np.float32)

        obs_spaces = {
            'position': gym.spaces.Box(low=low, high=high, shape=(17,), dtype=np.float32),
            'indicator': gym.spaces.Box(low=0, high=2, shape=(10,), dtype=np.int8),
            'path_state': gym.spaces.Box(low=0, high=2, shape=(1,), dtype=np.int8)
        } 
        self.observation_space=gym.spaces.Dict(obs_spaces)

    def _get_obs(self, normalize=False):

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
        lidar_results = self._get_lidar_probe()
        if self.objective:
            state = np.hstack((np.array(tmp2[4][:2]), np.array(tmp[0]), pyb.getEulerFromQuaternion(self._quat_ee_to_w(np.array(tmp[1]))), self.get_joint_state(), self.objective[0])).astype(np.float64)
        else:
            state = np.hstack((np.array(tmp2[4][:2]), np.array(tmp[0]), pyb.getEulerFromQuaternion(self._quat_ee_to_w(np.array(tmp[1]))), self.get_joint_state(), [0, 0, 0])).astype(np.float64)
    
        return {
            'position': state,
            'indicator': lidar_results,
            'path_state': self.path_state
        }

    def _move_base(self, new_base_pos, dynamic=True, delay = 0):
        """
        Method for moving the robot base to different coordinates.
        Will return True if movement happened, False if not.
        """
        state = self._get_obs(False)
        pos_diff = new_base_pos - state["position"][:2]
        pos_dist = np.linalg.norm(pos_diff)
        if pos_dist > 1e-4:
            if not dynamic:
                pyb.resetBasePositionAndOrientation(self.robot, np.append(new_base_pos, self.fixed_height[self.robot_name]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
                return True
            else:
                steps = int(pos_dist / self.base_speed)
                step = pos_diff * (self.base_speed / pos_dist)
                for i in range(steps):
                    pyb.resetBasePositionAndOrientation(self.robot, np.append(state["position"][:2] + (i+1)*step, self.fixed_height[self.robot_name]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
                    if delay:
                        time.sleep(delay)
                pyb.resetBasePositionAndOrientation(self.robot, np.append(new_base_pos, self.fixed_height[self.robot_name]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
                return True
        return False

    def _perform_action(self, action):

        if action is not None:
            state = self._get_obs(False)
            # unfortunately, the order of dict entries matters to the gym contains() method here
            # if somehow the order of dict entries in the observation space OrderedDict changes, then the order of the next lines defining the entries of the new state needs to be switched as well
            new_ee_position = state["position"][2:5] + action[:3]
            current_rotation_as_rpy = pyb.getEulerFromQuaternion(state["position"][5:9])
            new_rotation_as_rpy = np.array([entry + action[3:][idx] for idx, entry in enumerate(current_rotation_as_rpy)])
            new_rotation_as_quaternion = pyb.getQuaternionFromEuler(new_rotation_as_rpy)
            new_rotation_as_quaternion_in_correct_frame = self._quat_w_to_ee(new_rotation_as_quaternion)         
            # TODO: reimplement the valid bounds check once the bounds have actually been settled on sometime in the future

            # based on the information above, move the joints
            timeout = self.movep((new_ee_position, new_rotation_as_quaternion_in_correct_frame))
            if timeout:
                return timeout
        
        while not self.is_static:
            pyb.stepSimulation()

    def reward(self, obs):
        pos_done=False
        if self.path_state == 2:
            distance = np.linalg.norm(np.array([obs[2], obs[3], self.safe_height]) - obs["position"][2:5]) 
        else:
            objective_with_slight_offset = self.objective[0] + self.objective[1][0] * 0.01 + self.objective[1][1] * 0.01
            objective_with_slight_offset = np.array([1.96945097e+00,  1.87400000e+00,  8.26572114e-01])
            distance = np.linalg.norm(objective_with_slight_offset - obs["position"][2:5]) 
        if distance < self.ee_pos_reward_thresh:
            reward = 10
            pos_done = True  # objective achieved
        else:
            reward = -0.01 * distance
        col = self.is_in_collision()
        if col and not (pos_done):
            reward = -10
            pos_done = False

        if pos_done:
            self.objective = None

        success = pos_done and not col
        done = (self.objective is None and len(self.plan)==0 and len(self.goals)==0) or col

        return reward, success, done