import gym
import pybullet as pyb
import numpy as np
from util import xml_parser, util
import os
import pickle

class PathingEnvironmentPybullet(gym.Env):

    def __init__(self,
                asset_files_path,
                train=False,
                use_joints=False,
                display=False,
                show_target=False,
                ignore_obstacles_for_target_box=False,
                id=0):

        # id, default is 0, but can be another number, used for running several envs in parallel
        self.id = id

        # eval or train mode
        self.train = train

        # path for asset files
        self.asset_files_path = asset_files_path

        # bool flag for showing the target in the pybullet simulation
        self.show_target = show_target

        # bool falg for whether actions are xyz and rpy movements or joint movements
        self.use_joints = use_joints
        
        # list of pybullet object ids currently in the env
        self.obj_ids = []

        # tool mounted to the robot, 0: MRW510, 1: TAND GERAD
        self.tool = 0

        # attribute for target box, see its method
        self.ignore_obstacles_for_target_box = ignore_obstacles_for_target_box

        # pybullet robot constants
        self.resting_pose_angles = np.array([0, -0.5, 0.75, -1, 0.5, 0.5]) * np.pi  # resting pose angles for the kr16
        self.end_effector_link_id = 7  # link id for the end effector
        self.base_link_id = 8  # link id for the base
        self.joints_lower_limits = np.array([-3.228858, -3.228858, -2.408553, -6.108651, -2.26891, -6.108651])
        self.joints_upper_limits = np.array([3.22885911, 1.13446401, 3.0543261, 6.10865238, 2.26892802, 6.1086523])
        self.joints_range = self.joints_upper_limits - self.joints_lower_limits
        self.ceiling_mount_height = 2  # height at which the robot is mounted on the ceiling

        # angle conversion constants
        # these are used for converting the pybullet coordinate system of the end effector into the coordinate system used
        # by the MOSES ground truth, entry one is for MRW510, entry two is for TAND GERAD
        # they were derived by combining the rotation of the end effector link with the rotation of the torch mesh
        # both can be found in their respective urdf files
        self.ground_truth_conversion_angles = [np.array([-0.2726201, 0.2726201, -0.6524402, -0.6524402]), 
                                               np.array([-0.0676347, 0.0676347, -0.7038647, -0.7038647])]

        # if use_joints is set to false:
        # maximum translational movement per step
        self.pos_speed = 0.00001
        # maximum rotational movement per step
        self.rot_speed = 0.00001 
        # if use_joints is set to true:
        # maximum joint movement per step
        self.joint_speed = 0.01

        # offset at which welding meshes will be placed into the world
        self.xyz_offset = np.array([0, 0, 0])  # all zero for now, might change later on
        # scale factor for the meshes in the pybullet environment
        self.pybullet_mesh_scale_factor = 0.0005
        # xy offset for the robot base
        # the move base method moves its lower left corner, but we want to move its center, thus the offset
        self.base_offset = np.array([-0.125, -0.125])  # found by trial and error

        # threshold for the end effector position for maximum reward
        # == a sphere around the target where maximum reward is applied
        # is modified in the course of training
        if self.train:
            self.ee_pos_reward_thresh = 6e-1
            self.ee_pos_reward_thresh_min = 4e-3
            self.ee_pos_reward_thresh_max = 6e-1
            self.ee_pos_reward_thresh_increments = 1e-2
            self.ee_pos_reward_threshold_change_after_episodes = 50
            self.success_buffer = []
        else:
            self.ee_pos_reward_thresh = 1e-2

        # variables storing information about the current state, saves performance from having to 
        # access the pybullet interface all the time
        self.pos = None
        self.rot = None
        self.rot_internal = None
        self.pos_last = None
        self.target = None
        self.target_norms = None
        self.lidar_probe = None
        self.joints = None

        # process the dataset of meshes and urdfs
        self.dataset = self._register_data()

        # steps taken in current epsiode and total
        self.steps_current_episode = 0
        self.steps_total = 0
        # maximum steps per episode
        if self.train:
            self.max_episode_steps = 1024
        else:
            self.max_episode_steps = 512  # reduce the overhead for model evaluation
        # episode counter
        self.episodes = 0
        self.episode_reward = 0
        self.episode_distance = 0
        # success counter, to calculate success rate
        self.successes = 0

        # gym action space
        # first three entries: ee position, last three entries: ee rotation given as rpy in pybullet world frame
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # gym observation space
        # spatial: 1-6: current joint angles, 7-9: vector difference between target and current position, 10-13: current rotation as quaternion, 14: distance between target and position, all normalized to be between -1 and 1 for deep learning
        # lidar: 10 ints that signify occupancy around the ee
        self.observation_space = gym.spaces.Dict(
            {
              'spatial': gym.spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32),
              'lidar': gym.spaces.Box(low=-1, high=1, shape=(17,), dtype=np.int8)  
            }
        )
        # workspace bounds
        self.x_max = 8
        self.y_max = 8
        self.z_max = 1.8
        # constants to make normalizing more efficient
        max_distance = np.ceil(np.sqrt(self.x_max**2 + self.y_max**2 + self.z_max**2))  # this is the maximum possible distance given the workspace
        vec_distance_max = np.array([self.x_max, self.y_max, self.z_max])
        vec_distance_min = -1 * vec_distance_max
        self.normalizing_constant_a = np.zeros(14)
        self.normalizing_constant_a[:6] = 2 / self.joints_range
        self.normalizing_constant_a[6:9] = 2 / (vec_distance_max - vec_distance_min)
        self.normalizing_constant_a[9:13] = 0  # unit quaternions are already normalized
        self.normalizing_constant_a[13] = 2 / max_distance 
        self.normalizing_constant_b = np.zeros(14)
        self.normalizing_constant_b[:6] = np.ones(6) - np.multiply(self.normalizing_constant_a[:6], self.joints_upper_limits)
        self.normalizing_constant_b[6:9] = np.ones(3) - np.multiply(self.normalizing_constant_a[6:9], vec_distance_max)
        self.normalizing_constant_b[9:13] = 0
        self.normalizing_constant_b[13] = 1 - self.normalizing_constant_a[13] * max_distance

        # pybullet connection and setup
        disp = pyb.DIRECT  # direct <-> no gui, use for training
        if display:
            disp = pyb.GUI
        pyb.connect(disp)
        pyb.setAdditionalSearchPath(self.asset_files_path)

    ###############
    # Gym methods #
    ###############

    def reset(self):

        # set the step and episode counters correctly
        self.steps_current_episode = 0
        self.episodes += 1
        self.episode_reward = 0
        self.episode_distance = 0

        # clear out stored objects and reset the simulation
        self.obj_ids = []
        pyb.resetSimulation()

        # stop pybullet rendering for performance
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)

        # rebuild environment
        # load in the ground plane
        self._add_object("workspace/plane.urdf", [0, 0, -0.01])

        # load in the mesh of the welding part
        # info: the method will load the mesh at given index within the self.dataset variable
        file_index = np.random.choice(range(len(self.dataset["filenames"]))) 
        file_index = self.dataset["filenames"].index("201910204483_R1.urdf")
        #self._add_object("objects/"+self.dataset["filenames"][file_index], self.xyz_offset)
        # set the target and base target
        target_index = np.random.choice(range(len(self.dataset["frames"][file_index])))  # pick a random target from the welding part's xml
        #target_index = 0
        self._set_target(file_index, target_index)
        self._set_target_box(self.ee_pos_reward_thresh)
        if self.show_target:
            self._show_target()
        #tmp
        self.tool=0

        # load in the robot, the correct tool was set above while setting the target
        if self.tool:
            self.robot = self._add_object("kr16/kr16_tand_gerad.urdf", np.append(self.target_base, self.ceiling_mount_height), pyb.getQuaternionFromEuler([np.pi, 0, 0]))
        else:
            self.robot = self._add_object("kr16/kr16_mrw510.urdf", np.append(self.target_base, self.ceiling_mount_height), pyb.getQuaternionFromEuler([np.pi, 0, 0]))      

        # get the joint ids of the robot and set the joints to their resting position
        joints = [pyb.getJointInfo(self.robot, i) for i in range(pyb.getNumJoints(self.robot))]
        self.joint_ids = [j[0] for j in joints if j[2] == pyb.JOINT_REVOLUTE]
        self._set_joint_state(self.resting_pose_angles)

        # get state information
        ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        self.pos = np.array(ee_link_state[4])
        self.pos_last = self.pos
        self.rot = np.array(ee_link_state[5])
        self.rot_internal = np.array(ee_link_state[1])
        self.joints = self.resting_pose_angles
        self.lidar_probe = self._get_lidar_indicator()

        if self.train and self.episodes % self.ee_pos_reward_threshold_change_after_episodes == 0:
            success_rate = np.average(self.success_buffer) if len(self.success_buffer) != 0 else 0
            if success_rate < 0.8 and self.ee_pos_reward_thresh < self.ee_pos_reward_thresh_max:
                self.ee_pos_reward_thresh += self.ee_pos_reward_thresh_increments/4
            elif success_rate > 0.8 and self.ee_pos_reward_thresh > self.ee_pos_reward_thresh_min:
                self.ee_pos_reward_thresh -= self.ee_pos_reward_thresh_increments/2
            if self.ee_pos_reward_thresh < self.ee_pos_reward_thresh_min:
                self.ee_pos_reward_thresh = self.ee_pos_reward_thresh_min
            if self.ee_pos_reward_thresh > self.ee_pos_reward_thresh_max:
                self.ee_pos_reward_thresh = self.ee_pos_reward_thresh_max
            print("Current threshold for maximum reward: " + str(self.ee_pos_reward_thresh))

        # turn on rendering again
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)

        return self._get_obs()

    def _get_obs(self):
        spatial = np.zeros(14)
        spatial[:6] = np.multiply(self.normalizing_constant_a[:6], self.joints) + self.normalizing_constant_b[:6]
        spatial[6:9] = np.multiply(self.normalizing_constant_a[6:9], (self.target - self.pos)) + self.normalizing_constant_b[6:9]
        spatial[9:13] = self.rot
        spatial[13] = self.normalizing_constant_a[13] * np.linalg.norm(self.target - self.pos) + self.normalizing_constant_b[13]

        return {
            'spatial': spatial,
            'lidar': self.lidar_probe
        }

    def step(self, action):

        # get state info
        ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        self.pos = np.array(ee_link_state[4])
        self.pos_last = self.pos
        self.rot = np.array(ee_link_state[5])

        if not self.use_joints:
            # transform action
            pos_delta = action[:3] * self.pos_speed
            rpy_delta = action[3:] * self.rot_speed

            # add the action to the current state
            rot_rpy = util.quaternion_to_rpy(self.rot)
            pos_desired = self.pos + pos_delta
            rpy_desired = rot_rpy + rpy_delta
            quat_desired = util.rpy_to_quaternion(rpy_desired)

            # move the robot to the new positions and get the associated joint config
            self.joints = self._movep(pos_desired, quat_desired)
        else:
            # transform action
            joint_delta = action * self.joint_speed

            # add action to current state
            joints_desired = self.joints + joint_delta

            # execute movement by setting the desired joint state
            self.joints = self._movej(joints_desired)

        # get new state info
        ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        self.pos = np.array(ee_link_state[4])
        self.rot = np.array(ee_link_state[5])
        self.rot_internal = np.array(ee_link_state[1])
        self.lidar_probe = self._get_lidar_indicator()

        # increment steps
        self.steps_current_episode += 1
        self.steps_total += 1

        # get reward and info
        reward, done, info = self._reward()

        return self._get_obs(), reward, done, info


    def _reward(self):
        
        collided = self._collision()

        distance_cur = np.linalg.norm(self.target - self.pos)
        distance_last = np.linalg.norm(self.target - self.pos_last)

        # check if out of bounds
        if self._is_out_of_bounds():
            done = True
            is_success = False
            reward = -30
        elif not collided:
            #if distance_cur < self.ee_pos_reward_thresh:
            if self._is_in_target_box():
                reward = 10
                done = True
                is_success = True
                self.successes += 1
            else:
                reward = -0.05 * distance_cur
                done = False
                is_success = False
                if distance_cur > distance_last:
                    # add a very small penalty if the distance increased in comparison to the last step
                    reward -= 0.005 * distance_last
        else:
            reward = -15
            done = True
            is_success = False

        if self.steps_current_episode > self.max_episode_steps:
            done = True

        if self.train:
            if done:
                self.success_buffer.append(is_success)
                if len(self.success_buffer) > self.ee_pos_reward_threshold_change_after_episodes:
                    self.success_buffer.pop(0)
        
        self.episode_reward += reward
        self.episode_distance += distance_cur

        info = {
            'step': self.steps_current_episode,
            'steps_total': self.steps_total,
            'episodes': self.episodes,
            'done': done,
            'is_success': is_success,
            'success_rate': self.successes/self.episodes if not self.train else np.average(self.success_buffer),
            'collided': collided,
            'reward': reward,
            'episode_reward': self.episode_reward,
            'distance': distance_cur,
            'episode_distance': self.episode_distance,
            'distance_threshold': self.ee_pos_reward_thresh
        }

        if done:
            info_string = ""
            for key in info:
                info_string += key + ": " + str(round(info[key], 2)) + ", "
            print(info_string)

        return reward, done, info

    ####################
    # Pybullet methods #
    ####################

    def _add_object(self, urdf, position=[0, 0, 0], rotation=[0, 0, 0, 1]):
        """
        Add objects to env and registers it in the collection of all objects.
        """
        obj_id = pyb.loadURDF(
            urdf,
            position,  # xyz
            rotation,  # xyzw quaternion
            useFixedBase=True)
        self.obj_ids.append(obj_id)
        return obj_id

    def _get_joint_state(self):
        """
        Returns an array with the angles of the current robot configuration.
        """
        return np.array([pyb.getJointState(self.robot, i)[0] for i in self.joint_ids])
    
    def _set_joint_state(self, config):
        """
        Method for setting the joints of the robot to a certain configuration. Will kill all physics, momentum, movement etc. going on.
        """
        for i in range(len(self.joint_ids)):
            pyb.resetJointState(self.robot, self.joint_ids[i], config[i])

    def _collision(self):
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

    def _move_base(self, position):
        """
        Simply sets the robot base to the desired position.
        """
        pyb.resetBasePositionAndOrientation(self.robot, np.append(position, self.ceiling_mount_height), pyb.getQuaternionFromEuler([np.pi, 0, 0]))

    # methods taken almost 1:1 from ravens code, need to add proper attribution later TODO
    def _movej(self, targj, speed=0.01):
        """
        Move UR5 to target joint configuration.
        Returns the reached joint config.
        """
        if self.train:  # if we're training intermediate steps are not necessary
            self._set_joint_state(targj)
            return targj
        else:
            currj = self._get_joint_state()
            diffj = targj - currj
            while any(np.abs(diffj) > 1e-5):
                # Move with constant velocity
                norm = np.linalg.norm(diffj)
                if norm > speed:
                    v = diffj / norm
                    stepj = currj + v * speed
                else:
                    stepj = currj + diffj
                self._set_joint_state(stepj)

                currj = stepj
                diffj = targj - currj
            return currj

    def _movep(self, position, rotation, speed=0.01):
        """
        Move UR5 to target end effector pose. Returns the reached joint config.
        """

        targj = self._solve_ik(position, rotation)
        return self._movej(targj, speed)

    def _solve_ik(self, position, rotation):
        """
        Calculate joint configuration with inverse kinematics.
        """

        joints = pyb.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self.end_effector_link_id,
            targetPosition=position,
            targetOrientation=rotation,
            lowerLimits=self.joints_lower_limits.tolist(),
            upperLimits=self.joints_upper_limits.tolist(),
            jointRanges=self.joints_range.tolist(),
            restPoses=np.float32(self.resting_pose_angles).tolist(),
            maxNumIterations=2000,
            residualThreshold=5e-3)
        joints = np.float32(joints)
        #joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def _cast_lidar_rays(self, ray_min=0.02, ray_max=0.05, ray_num_side=6, ray_num_forward=12, render=False):
        """
        Casts rays from various positions on the torch and receives collision information from that. Currently only adjusted for the MRW tool.
        """
        ray_froms = []
        ray_tops = []
        # get the frame of the torch tip
        frame_torch_tip = util.quaternion_to_matrix(util.quaternion_multiply(self._quat_ee_to_w(self.rot_internal), np.array([ 0, 1, 0, 0 ])))
        frame_torch_tip[0:3,3] = self.pos
        # get the frame of the torch grip
        frame_torch_grip = util.quaternion_to_matrix(self.rot)
        frame_torch_grip[0:3,3] = self.pos

        # cast a ray from the tip of the torch straight down
        ray_froms.append(np.matmul(np.asarray(frame_torch_tip),np.array([0.0,0.0,0.01,1]).T)[0:3].tolist())
        ray_tops.append(np.matmul(np.asarray(frame_torch_tip),np.array([0.0,0.0,ray_max,1]).T)[0:3].tolist())


        # cast rays from the torch tip in a cone in forward direction
        for angle in range(230, 270, 20):
            for i in range(ray_num_forward):
                z = -ray_max * np.sin(angle*np.pi/180)
                l = ray_max * np.cos(angle*np.pi/180)
                x_end = l*np.cos(2*np.pi*float(i)/ray_num_forward)
                y_end = l*np.sin(2*np.pi*float(i)/ray_num_forward)
                start = np.matmul(np.asarray(frame_torch_tip),np.array([0.0,0.0,0.01,1]).T)[0:3].tolist()
                end = np.matmul(np.asarray(frame_torch_tip),np.array([x_end,y_end,z,1]).T)[0:3].tolist()
                ray_froms.append(start)
                ray_tops.append(end)
        
        # set the angle of rays
        interval = -0.005
        
        # cast rays from the tip horizontally around the torch tip 
        for i in range(8):
            ai = i*np.pi/4
            for angle in range(ray_num_side):    
                z_start = (angle)*interval-0.01
                x_start = ray_min*np.cos(ai)
                y_start = ray_min*np.sin(ai)
                start = np.matmul(np.asarray(frame_torch_tip),np.array([x_start,y_start,z_start,1]).T)[0:3].tolist()
                z_end = (angle)*interval-0.01
                x_end = ray_max*np.cos(ai)
                y_end = ray_max*np.sin(ai)
                end = np.matmul(np.asarray(frame_torch_tip),np.array([x_end,y_end,z_end,1]).T)[0:3].tolist()
                ray_froms.append(start)
                ray_tops.append(end)

        # cast rays from the horizontally around the grip
        for i in range(8):
            ai = i*np.pi/4
            for angle in range(ray_num_side):    
                z_start = (angle)*interval - 0.05
                x_start = ray_min*np.cos(ai) - 0.00
                y_start = ray_min*np.sin(ai) - 0.03
                start = np.matmul(np.asarray(frame_torch_grip),np.array([x_start,y_start,z_start,1]).T)[0:3].tolist()
                z_end = (angle)*interval - 0.05
                x_end = ray_max*np.cos(ai) - 0.00
                y_end = ray_max*np.sin(ai) - 0.03
                end = np.matmul(np.asarray(frame_torch_grip),np.array([x_end,y_end,z_end,1]).T)[0:3].tolist()
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

    def _get_lidar_indicator(self):
        ray_num_side=6
        ray_num_forward=12
        lidar_results = np.array(self._cast_lidar_rays(ray_num_side=ray_num_side, ray_num_forward=ray_num_forward), dtype=object)[:,2]  # only use the distance information
        indicator = np.zeros((17,), dtype=np.float32)
        # side note: the array slices here are very complicated, but they basically just count up the rays in the order they are in the lidar_results object
        # as pybullet will output them in the lider_cylinder method. Instead of hardcoding the slices they are kept as variables such that the amount of rays can be changed at will
        # tip front indicator
        lidar_min = lidar_results[0:(2 * ray_num_forward + 1)].min()  # 1 ray going straight forward + 2 cones of ray_num_forward rays around it
        indicator[0] = 1 if lidar_min >= 0.99 else (0 if 0.5 < lidar_min < 0.99 else -1)
        # tip sides indicators
        for i in range(8):
            lidar_min = lidar_results[(2 * ray_num_forward + 1 + i * ray_num_side):(2 * ray_num_forward + 1 + (i + 1) * ray_num_side)].min()
            indicator[1+i] = 1 if lidar_min >= 0.99 else (0 if 0.5 < lidar_min < 0.99 else -1)
        # grip sides indicators
        for i in range(8):
            lidar_min = lidar_results[(2 * ray_num_forward + 1 + 8 * ray_num_side + i * ray_num_side):(2 * ray_num_forward + 1 + 8 * ray_num_side + (i + 1) * ray_num_side)].min()
            indicator[1+8+i] = 1 if lidar_min >= 0.99 else (0 if 0.5 < lidar_min < 0.99 else -1)
        return indicator

    def _is_in_target_box(self):
        """
        Returns a bool that indicates whether the current position is within the target box or not.
        """
        # first transform the current position into the target frame
        # this makes checking wether it's in the box the much easier
        position_in_target_frame = util.rotate_vec(self.target_transform, (self.pos - self.target))

        # now check for all three directions
        if position_in_target_frame[0] < 0 or position_in_target_frame[0] > self.target_x_max:  # less than 0 would be in the mesh or behind it
            return False
        if position_in_target_frame[1] < 0 or position_in_target_frame[1] > self.target_y_max:  # dito
            return False
        if position_in_target_frame[2] < self.target_m_z_max or position_in_target_frame[2] > self.target_p_z_max:  # for the constructed z-axis both directions are fine
            return False
        return True 

    def _set_target_box(self, dist, extend_z_axis=True):
        """
        Method for determining the size of the bounds of target box (where full reward is given).
        Uses Pybullet raycasting to determine the bounds and position of the box.
        This is necessary to make sure that the box doesn't cut into a mesh and gives full reward to a spot where the torch has no direct path to the actual target itself.
        The extend_z_axis parameter will make it such that the box is 1.5x longer in (world) z direction, such that the robot has an easier time finding the target from above (or below even). 
        This will not be applied to the third axis in target frame as it would in many cases lead to the box crossing through mesh walls.
        """
        # get the norms from the target
        # and also calculate the vectors perpendicular to the two norms
        norms = [self.target_norms[0], self.target_norms[1], np.cross(self.target_norms[0], self.target_norms[1]), -np.cross(self.target_norms[0], self.target_norms[1])]   

        # calculate which of the three norms is most aligned with world z axis, this is used later on
        z_axis = np.array([0, 0, 1])
        z_axis_norm = 0
        z_axis_score = -2
        for i, norm in enumerate(norms):
            cos_sim = util.cosine_similarity(z_axis, norm)
            if cos_sim > z_axis_score:
                z_axis_score = cos_sim
                z_axis_norm = i

        # info: the way the norms are given in the source files makes it such that the crossproduct of both will always be parallel to a wall while the norms point up and away
        # that's why we also need the negative crossproduct, to cast a ray into the other direction as well
        # for the other two norms we're only interested in positive direction, because negative direction will immediately hit the mesh where the welding seam is sitting on
        
        # the three vectors can be taken to represent a rotation, get the matrix for it
        transform = np.zeros((3,3))
        transform[0:3,0] = norms[0]
        transform[0:3,1] = norms[1]
        transform[0:3,2] = norms[2]

        # now we can get the associated quaternion
        # which tells us how to transform coordinates in world frame to the frame implied by the target norms
        # this is useful for easily checking if a position is within the target box or not, especially if it's rotated w.r.t the world frame
        self.target_transform = util.matrix_to_quaternion(transform)

        # now use the norms to cast rays to determine the maximum possible extent of the target box if so desired
        scale_factor = 1.5
        if not self.ignore_obstacles_for_target_box:
            ray_starts = [(self.target).tolist() for i in range(4)]
            ray_ends = [(self.target + norms[i] * dist).tolist() for i in range(4)]
            ray_results = np.array(pyb.rayTestBatch(ray_starts, ray_ends), dtype=object)[:,2]  # only extract the hitfraction

            # sometimes the target position is such that some rays will immediately hit a wall, giving all results as zero
            # in that case, move the starting point of the rays slightly further inward along the norms
            norm_between = norms[0] + norms[1]
            norm_between = norm_between / np.linalg.norm(norm_between)
            j = 1
            while np.any(ray_results == 0):
                ray_starts = [(self.target + j*0.01*norm_between).tolist() for i in range(4)]
                ray_results = np.array(pyb.rayTestBatch(ray_starts, ray_ends), dtype=object)[:,2]
                j += 1

            self.target_x_max = dist * ray_results[0] if z_axis_norm !=0 else dist * scale_factor #* ray_results[0]
            self.target_y_max = dist * ray_results[1] if z_axis_norm !=1 else dist * scale_factor #* ray_results[1]
            self.target_p_z_max = dist * ray_results[2]
            self.target_m_z_max = -dist * ray_results[3]
        else:
            self.target_x_max = dist if z_axis_norm !=0 else dist * scale_factor
            self.target_y_max = dist if z_axis_norm !=1 else dist * scale_factor
            self.target_p_z_max = dist/2 if z_axis_norm !=2 else dist * scale_factor
            self.target_m_z_max = -dist/2 if z_axis_norm !=2 else dist * scale_factor




    ###################
    # utility methods #
    ###################

    def _is_out_of_bounds(self):
        """
        Returns boolean if the robot ee position is out of bounds
        """
        if self.pos[0] < 0 or self.pos[0] > self.x_max:
            return True
        if self.pos[1] < 0 or self.pos[1] > self.y_max:    
            return True
        if self.pos[2] < 0 or self.pos[2] > self.z_max:
            return True
        return False

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

    def _set_target(self, file_index, target_index):
        """
        Fills the target variable with the weldseam at target_index from the xml associated with file_index.
        Also moves the base accordingly.
        """
        frame = self.dataset["frames"][file_index][target_index]
        self.tool = 1 if frame["torch"][3] == "TAND_GERAD_DD" else 0
        positions = [ele["position"] * self.pybullet_mesh_scale_factor + self.xyz_offset for ele in frame["weld_frames"]]
        norms = [ele["norm"] for ele in frame["weld_frames"]]
        self.target = np.array(positions[0])
        self.target_norms = norms[0]
        self.target_base = np.average(positions, axis=0)[:2] + self.base_offset

    def _show_target(self):
        """
        Creates a visual indicator around the target in the rendered Pybullet simulation
        """
        #visual_id = pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=self.ee_pos_reward_thresh, rgbaColor=[0, 1, 0, 1])
        #point_id = pyb.createMultiBody(
        #            baseMass=0,
        #            baseVisualShapeIndex=visual_id,
        #            basePosition=self.target,
        #            )
        
        # calculate the center point such that the rectangle will be displayed in the correct way
        # this is necessary because the pybullet interface for boxes works with halflenghts emanating from the center
        center_in_target_frame = np.array([self.target_x_max/2.0, 
                                        self.target_y_max/2.0,
                                        (self.target_m_z_max + self.target_p_z_max)/2.0])  # the target in target frame is of course 0,0,0, so no need to add it here
        #center_in_world_frame = util.rotate_vec(util.quaternion_invert(self.target_transform), center_in_target_frame) + self.target
        center_in_world_frame = util.rotate_vec(self.target_transform, center_in_target_frame) + self.target
        halfExtents = [self.target_x_max/2.0, self.target_y_max/2.0, (np.abs(self.target_m_z_max) + self.target_p_z_max)/2.0]
        third_dim = np.cross(self.target_norms[0], self.target_norms[1])
        box = pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0, 1, 0, 1])
        pyb.createMultiBody(baseVisualShapeIndex=box, basePosition=center_in_world_frame, baseOrientation=self.target_transform)

        pyb.addUserDebugLine(self.target, self.target+third_dim, [0,0,1])
        pyb.addUserDebugLine(self.target, np.array(self.target)+np.array(self.target_norms[0]),[1,0,0])
        pyb.addUserDebugLine(self.target, np.array(self.target)+np.array(self.target_norms[1]),[0,1,0])

    def _quat_w_to_ee(self, quat):
        """
        Takes an input quaternion in the world frame, rotates it such that it offsets the rotation of the welding torch
        (meaning that a [0,0,0,1] input quaternion for the robot arm ee results in the same pose of the torch as loading it with the loadURDF method would) 
        and transforms it into the end effector frame (meaning that that inputs result in the correct rotation w.r.t the world frame)
        """

        # get offset
        offset = self.ground_truth_conversion_angles[self.tool]

        # rotate the user input by the offset (the torch will now be positioned like when its original mesh is loaded by the loadURDF method if input is [0, 0, 0, 1])
        middle_quat = util.quaternion_multiply(offset, quat)
        # however, the coordinate system is still wrongly aligned, so we will have to switch systems by multiplying through the offset
        # this will make it so that our input (quat) rotates around the axes of the world coordinate system instead of the world axes rotated by the offset
        offset = util.quaternion_invert(offset)
        res = util.quaternion_multiply(offset, middle_quat)
        res = util.quaternion_multiply(res, util.quaternion_invert(offset))

        return res

    def _quat_ee_to_w(self, quat):
        """
        Same as above, just from end effector frame to world frame.
        """

        offset = self.ground_truth_conversion_angles[self.tool]

        tmp = util.quaternion_multiply(offset, quat)
        tmp = util.quaternion_multiply(tmp, util.quaternion_invert(offset))

        res = util.quaternion_multiply(util.quaternion_invert(offset), tmp)

        return res

    def _save_env_state(self, save_path="./model/saved_envs/"):
        """
        Writes the misc information about the env into a pckl file such that training progress can restored easily later on.
        The file will have the name of the id given at initialization.
        """
        save_dict = {
            'id': self.id,
            'steps': self.steps_total,
            'dist_threshold': self.ee_pos_reward_thresh,
            'episodes': self.episodes,
            'successes': self.successes,
            'success_buffer': self.success_buffer,
        }
        with open(save_path + str(self.id)+ ".pckl", "wb") as outfile:
            pickle.dump(save_dict, outfile, pickle.HIGHEST_PROTOCOL)

    def _load_env_state(self, load_path="./model/saved_envs/"):
        """
        Loads misc information useful for continuing training from a npy in load path. Will automatically use the file named after the env's id.
        """
        try:
            with open(load_path + str(self.id) + ".pckl", "rb") as infile:
                load_dict = pickle.load(infile)        
            self.id = load_dict["id"]
            self.steps_total = load_dict["steps"]
            self.ee_pos_reward_thresh = load_dict["dist_threshold"]
            self.episodes = load_dict["episodes"]
            self.successes = load_dict["successes"]
            self.success_buffer = load_dict["success_buffer"]
        except FileNotFoundError:
            pass

        