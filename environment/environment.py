from random import choice
import gym
import pybullet as pyb
import numpy as np
from util import xml_parser, util
import os
import pickle
from time import sleep, time

class PathingEnvironmentPybullet(gym.Env):

    def __init__(self,
                env_config):

        # int: id, default is 0, but can be another number, used for distinguishing envs running in parallel
        self.id = env_config["id"]

        # bool: eval or train mode
        self.train = env_config["train"]

        # path for asset files
        self.asset_files_path = env_config["asset_files_path"]

        # bool: show the target in the pybullet simulation
        self.show_target = env_config["show_target"]

        # bool: actions as xyz and rpy movements or joint movements
        self.use_joints = env_config["use_joints"]

        # bool: give rewards for reaching a goal rotation
        self.use_rotations = env_config["use_rotations"]

        # bool: process lidar results to indicator
        self.use_raw_lidar = env_config["use_raw_lidar"]

        # bool: normalize observations and rewards
        self.normalize = env_config["normalize"]

        # attribute for target box, see its method
        self.ignore_obstacles_for_target_box = env_config["ignore_obstacles_for_target_box"]

        # gamma, just used for record keeping such that the episode reward total will be the same as the one given by stable baselines
        self.gamma = env_config["gamma"]

        # bool: have pybullet display a GUI for the simulation
        self.display = env_config["display"]

        # float: minimal clearance in front of objects, failure to uphold this ends episode
        self.minimal_clearance = 0.01

        # logging level, 0: no logging, 1: logging, 2: logging and writing to textfile
        self.logging = env_config["logging"]
        if self.logging > 0:
            self.log = []

        # bool: shape of the target zone, used for overwritting by subclasses that might use a sphere as target zone
        self.target_zone_sphere = False
        
        # list of pybullet object ids currently in the env
        self.obj_ids = []

        # tool mounted to the robot, 0: MRW510, 1: TAND GERAD
        self.tool = 0

        # pybullet robot constants
        self.resting_pose_angles = np.array([0, -0.5, 0.75, -1, 0.5, 0.5]) * np.pi  # resting pose angles for the kr16
        self.end_effector_link_id = 7  # link id for the end effector
        self.base_link_id = 8  # link id for the base
        self.joints_lower_limits = np.array([-3.228858, -3.228858, -2.408553, -6.108651, -2.26891, -6.108651])
        self.joints_upper_limits = np.array([3.22885911, 1.13446401, 3.0543261, 6.10865238, 2.26892802, 6.1086523])
        self.joints_range = self.joints_upper_limits - self.joints_lower_limits
        self.ceiling_mount_height = 2  # height at which the robot is mounted on the ceiling
        self.rpy_upper_limits = np.array([1.3, 2, np.pi])
        self.rpy_lower_limits = np.array([-2.3, -2, -np.pi])
        # pybullet object ids
        self.robot = None
        self.welding_mesh = None
        self.mesh_file_index = None

        # angle conversion constants
        # these are used for converting the pybullet coordinate system of the end effector into the coordinate system used
        # by the MOSES ground truth, entry one is for MRW510, entry two is for TAND GERAD
        # they were derived by combining the inherent rotation of the end effector link in the URDF with the rotation of the torch mesh on top of that
        # both can be found in their respective urdf files
        self.ground_truth_conversion_angles = [np.array([-0.2726201, 0.2726201, -0.6524402, -0.6524402]), 
                                               np.array([-0.0676347, 0.0676347, -0.7038647, -0.7038647])]

        # if use_joints is set to false:
        # maximum translational movement per step
        self.pos_speed = 0.005
        # maximum rotational movement per step
        self.rot_speed = 0.02
        # if use_joints is set to true:
        # maximum joint movement per step
        self.joint_speed = 0.005

        # offset at which welding meshes will be placed into the world
        self.xyz_offset = np.array([0, 0, 0])  # all zero for now, might change later on
        # scale factor for the meshes in the pybullet environment
        self.pybullet_mesh_scale_factor = 0.0005
        # xy offset for the robot base
        # the move base method moves its lower left corner, but we want to move its center, thus the offset
        self.base_offset = np.array([-0.125, -0.125])  # found by trial and error

        # thresholds for the end effector for maximum reward
        # and values for modifiyng it during training
        if self.train:
            # position
            self.ee_pos_reward_thresh = 6e-1
            self.ee_spawn_thresh = 1e-1
            self.ee_spawn_thresh_max = 1.5
            self.ee_spawn_thresh_min = 1e-3
            self.ee_pos_reward_thresh_min = 1e-2
            self.ee_pos_reward_thresh_max = 6e-1
            self.ee_pos_reward_thresh_max_increment = 1e-2
            self.ee_pos_reward_thresh_min_increment = 1e-3
            # rotation
            self.ee_rot_reward_thresh = 4e-1
            self.ee_rot_reward_thresh_min = 5e-2
            self.ee_rot_reward_thresh_max = 4e-1
            self.ee_rot_reward_thresh_max_increment = 1e-2
            self.ee_rot_reward_thresh_min_increment = 1e-3
            # for statistics during training
            self.stats_buffer_size = 25
            self.success_buffer = []
            self.success_rot_buffer = []
            self.success_pos_buffer = []
            self.collision_buffer = []
            self.clearance_buffer = []
            self.timeout_buffer = []
            self.out_of_bounds_buffer = []
        else:
            self.ee_pos_reward_thresh = 1e-2
            self.ee_pos_reward_thresh_min = 1e-2
            self.ee_rot_reward_thresh = 5e-2
            self.ee_rot_reward_thresh_min = 5e-2

        # variables storing information about the current state, saves performance from having to 
        # access the pybullet interface all the time
        self.pos = None
        self.pos_vel = None
        self.rot = None
        self.rot_internal = None
        self.pos_last = None
        self.target = None
        self.target_norms = None
        self.target_rot = None
        self.lidar_indicator = None
        self.lidar_indicator_raw = None
        self.joints = None
        self.joints_vel = None
        self.max_dist = None
        self.min_dist = None
        self.distance = None
        self.time = None

        # lidar settings
        self.ray_num_forward = 12  # number of rays emanating from the torch tip forwards
        self.ray_num_side_circle = 8  # number of directions rays are sent in around the torch sides
        self.ray_num_side = 6  # number of rays per above direction
        self.ray_start = 0.02  # offset from the mesh center from where the ray starts, necessary to avoid the ray hitting the torch mesh itself, don't edit
        self.ray_end = 0.3  # distance away from the mesh center where the ray ends
        self.lidar_offset = 0.0093  # "true" offset between the measurements from this lidar setup and and pybullet getClosestPoints (due to the thickness of the mesh), meaning that the real distance between an object and the torch is at least this offset larger

        # process the dataset of meshes and urdfs
        self.dataset = self._register_data()

        # steps taken in current epsiode and total
        self.steps_current_episode = 0
        self.steps_total = 0
        # maximum steps per episode
        if self.train:
            self.max_episode_steps = 750
        else:
            self.max_episode_steps = 500  # reduce the overhead for model evaluation, succesful episodes should never take more than ~250 steps anyway
        # episode counter
        self.episodes = 0
        self.episode_discounted_reward = 0
        self.episode_distance = 0
        # success counter, to calculate success rate
        self.successes = 0
        # collision counter, to calculate collision rate
        self.collisions = 0
        # clearance failure counter
        self.clearance_failures = 0
        # bool flag if current episode is a success, gets used in the next episode for adjusting the target zone difficulty
        self.is_success = False
        # counter for reseting the mesh in the reset method
        # this means that for x episodes, the mesh in the env will stay the same and only the targets will change
        # this saves compute performance with no drawback on training quality
        self.reset_mesh_after_episodes = 100

        # gym action space
        # vector of length 6, gets interpreted as either xyz-rpy change or joint angle change
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # gym observation space
        joint_dims = 6 if self.use_joints else 0
        base_target_vector_pos_dims = 2
        target_vector_pos_dims = 3
        orientation_dims = 3
        distance_pos_dims = 1
        target_orientation_dims = 3 if self.use_rotations else 0
        distance_rot_dims = 1 if self.use_rotations else 0
        spatial_dims = joint_dims + base_target_vector_pos_dims + target_vector_pos_dims + orientation_dims + distance_pos_dims + target_orientation_dims + distance_rot_dims
        lidar_rays_default = 2 * 8 + 1
        lidar_dims = lidar_rays_default
        self.observation_space = gym.spaces.Dict(
            {
            'spatial': gym.spaces.Box(low=-1 if self.normalize else -8, high=1 if self.normalize else 8, shape=(spatial_dims,), dtype=np.float32),
            'lidar': gym.spaces.Box(low=0 if self.use_raw_lidar else -1, high=1, shape=(lidar_dims,), dtype=np.float32)  
            }
        )

        # reward bounds
        self.max_reward = 10  # reward given for a full success
        self.min_reward = -10  # reward given for collision
        # workspace bounds
        self.x_max = 8
        self.y_max = 8
        self.z_max = 1.8
        self.x_min = 0
        self.y_min = 0
        self.z_min = 0
        # constants to make normalizing more efficient
        max_distance = np.ceil(np.sqrt((self.x_max - self.x_min)**2 + (self.y_max - self.y_min)**2 + (self.z_max - self.z_min)**2))  # this is the maximum possible distance given the workspace
        vec_distance_max = np.array([self.x_max, self.y_max, self.z_max])
        vec_distance_min = -1 * vec_distance_max
        self.normalizing_constant_obs_a = np.zeros(19)
        self.normalizing_constant_obs_a[:6] = 2 / self.joints_range
        self.normalizing_constant_obs_a[6:8] = 2 / (vec_distance_max - vec_distance_min)[2:]
        self.normalizing_constant_obs_a[8:11] = 2 / (vec_distance_max - vec_distance_min)
        self.normalizing_constant_obs_a[11:14] = 2 / (self.rpy_upper_limits - self.rpy_lower_limits)
        self.normalizing_constant_obs_a[14] = 1 / max_distance
        self.normalizing_constant_obs_a[15:18] = 2 / (self.rpy_upper_limits - self.rpy_lower_limits)
        self.normalizing_constant_obs_a[18] = 0  # doesnt need normalizing
        self.normalizing_constant_reward_a = 2 / (self.max_reward - self.min_reward)

        self.normalizing_constant_obs_b = np.zeros(19)
        self.normalizing_constant_obs_b[:6] = np.ones(6) - np.multiply(self.normalizing_constant_obs_a[:6], self.joints_upper_limits)
        self.normalizing_constant_obs_b[6:8] = np.ones(2) - np.multiply(self.normalizing_constant_obs_a[6:8], vec_distance_max[2:])
        self.normalizing_constant_obs_b[8:11] = np.ones(3) - np.multiply(self.normalizing_constant_obs_a[8:11], vec_distance_max)
        self.normalizing_constant_obs_b[11:14] = np.ones(3) - np.multiply(self.normalizing_constant_obs_a[11:14], self.rpy_upper_limits)
        self.normalizing_constant_obs_b[14] = 1 - self.normalizing_constant_obs_a[14] * max_distance
        self.normalizing_constant_obs_b[15:18] = np.ones(3) - np.multiply(self.normalizing_constant_obs_a[15:18], self.rpy_upper_limits)
        self.normalizing_constant_obs_b[18] = 0  # doesnt need normalizing
        self.normalizing_constant_reward_b = 1 - self.normalizing_constant_reward_a * self.max_reward

        self.reload = False

        # pybullet connection and setup
        disp = pyb.DIRECT  # direct <-> no gui, use for training
        if self.display:
            disp = pyb.GUI
        pyb.connect(disp)
        pyb.setAdditionalSearchPath(self.asset_files_path)

        self.last_reward = 0

    ###############
    # Gym methods #
    ###############

    def reset(self):

        self.time = time()

        complete_reset = self.episodes == 0 or self.episodes % self.reset_mesh_after_episodes == 0 or self.reload
        self.reload = False

        # clear out stored objects and reset the simulation if necessary
        if complete_reset:
            self.obj_ids = []
            pyb.resetSimulation()

        # set the step and episode counters correctly
        self.steps_current_episode = 0
        self.episodes += 1
        self.episode_discounted_reward = 0
        self.episode_distance = 0

        # stop pybullet rendering for performance
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)

        if complete_reset:
            # rebuild environment
            # load in the ground plane
            self._add_object("workspace/plane.urdf", [0, 0, -0.01])

            # load in the mesh of the welding part
            # info: the method will load the mesh at given index within the self.dataset variable
            #self.mesh_file_index = np.random.choice(range(len(self.dataset["filenames"]))) 
            self.mesh_file_index = self.dataset["filenames"].index("201910204483_R1.urdf")
            self.welding_mesh = self._add_object("objects/"+self.dataset["filenames"][self.mesh_file_index], self.xyz_offset)
        # set the target and base target (this is done in the method calls below)
        while True:
            try:
                target_index = np.random.choice(range(len(self.dataset["frames"][self.mesh_file_index])))  # pick a random target from the welding part's xml
                break
            except ValueError:
                self.mesh_file_index = np.random.choice(range(len(self.dataset["filenames"])))  # sometimes there are no targets in a given entry of the xml and the code will throw an error, this is to prevent that and choose another xml
        target_index = 0
        tool_old = self.tool  # remember the old tool for checking in a moment
        self._set_target(self.mesh_file_index, target_index)
        if not self.target_zone_sphere:
            self._set_target_box(self.ee_pos_reward_thresh)
        if self.show_target:
            self._show_target()
        #tmp, set to one tool for now, full implementation for both tools later on
        self.tool=0

        # load in the robot, the correct tool was set above in the _set_target method call
        if self.tool == tool_old and not complete_reset:
            # if the tool has stayed the same, we can avoid an expensive load in of the robot mesh from the drive and simply change its current position
            pyb.resetBasePositionAndOrientation(self.robot, np.append(self.target_base, self.ceiling_mount_height), pyb.getQuaternionFromEuler([np.pi, 0, 0]))
        else:
            # if the tool has changed (or we're running the env for the first time or completely reseting), loading in the robot mesh is necessary
            if self.tool:
                self.robot = self._add_object("kr16/kr16_tand_gerad.urdf", np.append(self.target_base, self.ceiling_mount_height), pyb.getQuaternionFromEuler([np.pi, 0, 0]))
            else:
                self.robot = self._add_object("kr16/kr16_mrw510.urdf", np.append(self.target_base, self.ceiling_mount_height), pyb.getQuaternionFromEuler([np.pi, 0, 0]))      

        # get the joint ids of the robot and set the joints to their resting position
        joints = [pyb.getJointInfo(self.robot, i) for i in range(pyb.getNumJoints(self.robot))]
        self.joint_ids = [j[0] for j in joints if j[2] == pyb.JOINT_REVOLUTE]
        self._set_joint_state(self.resting_pose_angles)
        self.joints = self.resting_pose_angles
        self.joints_vel = np.zeros(6)

        # in training set the arm to a random position sometimes to force a variety of starting positions
        if self.train:
            if np.random.random() < -1:
                start = True
                while start or self._collision():
                    start = False
                    random_xyz_offset = np.random.random(3) * 2 - 1
                    random_xyz_offset[2] = abs(random_xyz_offset[2])
                    random_xyz_offset = random_xyz_offset * (max(self.ee_pos_reward_thresh * 1.5, np.random.random()) / np.linalg.norm(random_xyz_offset))
                    random_rpy = np.random.uniform(low=self.rpy_lower_limits, high=self.rpy_upper_limits, size=3)
                    self.joints = self._movep(self.target +  random_xyz_offset, self._quat_w_to_ee(util.rpy_to_quaternion(random_rpy)))
                    start = np.linalg.norm(np.array(pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)[4]) - (random_xyz_offset + self.target)) > self.ee_pos_reward_thresh * 1.5

        # get state information
        ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        self.pos = np.array(ee_link_state[4])
        self.pos_last = self.pos
        self.pos_vel = np.zeros(3)
        self.rot = np.array(ee_link_state[5])
        self.rot_internal = self._quat_ee_to_w(np.array(ee_link_state[1]))
        self.lidar_indicator, self.lidar_indicator_raw = self._get_lidar_indicator()
        self.max_dist = np.linalg.norm(self.pos - self.target)
        self.min_dist = self.max_dist
        self.distance = self.max_dist

        # set the success thresholds
        if self.train:
            if len(self.success_buffer) != 0:
                success_rate_rot = np.average(self.success_rot_buffer) 
                success_rate_pos = np.average(self.success_pos_buffer)
            else:
                success_rate_rot = 0
                success_rate_pos = 0
            if success_rate_pos > 0.8:
                if self.ee_pos_reward_thresh > self.ee_pos_reward_thresh_min and self.is_success:
                    self.ee_pos_reward_thresh -= util.linear_interpolation(self.ee_pos_reward_thresh, self.ee_pos_reward_thresh_min, self.ee_pos_reward_thresh_max, self.ee_pos_reward_thresh_min_increment, self.ee_pos_reward_thresh_max_increment)
            if success_rate_rot > 0.8:
                if self.ee_rot_reward_thresh > self.ee_rot_reward_thresh_min and self.is_success:
                    self.ee_rot_reward_thresh -= util.linear_interpolation(self.ee_rot_reward_thresh, self.ee_rot_reward_thresh_min, self.ee_rot_reward_thresh_max, self.ee_rot_reward_thresh_min_increment, self.ee_rot_reward_thresh_max_increment) 
            if self.ee_pos_reward_thresh < self.ee_pos_reward_thresh_min:
                self.ee_pos_reward_thresh = self.ee_pos_reward_thresh_min
            if self.ee_pos_reward_thresh > self.ee_pos_reward_thresh_max:
                self.ee_pos_reward_thresh = self.ee_pos_reward_thresh_max
            if self.ee_rot_reward_thresh < self.ee_rot_reward_thresh_min:
                self.ee_rot_reward_thresh = self.ee_rot_reward_thresh_min
            if self.ee_rot_reward_thresh > self.ee_rot_reward_thresh_max:
                self.ee_rot_reward_thresh = self.ee_rot_reward_thresh_max
        
        self.is_success = False
        if self.logging > 0:
            self.log = []
            log = []
            log.append(self.time)
            log += self.pos.tolist()
            log += self.rot_internal.tolist()
            log.append(self.distance)
            log += self.pos_vel.tolist()
            log += self.joints_vel.tolist()
            self.log.append(log)


        # turn on rendering again
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)

        return self._get_obs()

    def _get_obs(self):
        spatial = np.zeros(19)
        if self.normalize:   
            spatial[:6] = np.multiply(self.normalizing_constant_obs_a[:6], self.joints) + self.normalizing_constant_obs_b[:6]
            spatial[6:8] = np.multiply(self.normalizing_constant_obs_a[6:8], self.target_base - self.pos[:2]) + self.normalizing_constant_obs_b[6:8]
            spatial[8:11] = np.multiply(self.normalizing_constant_obs_a[8:11], (self.target - self.pos)) + self.normalizing_constant_obs_b[8:11]
            spatial[11:14] = np.multiply(self.normalizing_constant_obs_a[11:14], util.quaternion_to_rpy(self.rot_internal)) +  self.normalizing_constant_obs_b[11:14]
            spatial[14] = self.normalizing_constant_obs_a[14] * self.distance + self.normalizing_constant_obs_b[14]
            spatial[15:18] = np.multiply(self.normalizing_constant_obs_a[15:18], util.quaternion_to_rpy(self.target_rot)) + self.normalizing_constant_obs_b[15:18]
            spatial[18] = util.quaternion_similarity(self.rot_internal, self.target_rot)
        else:
            spatial[:6] = self.joints
            spatial[6:8] = self.target_base - self.pos[:2]
            spatial[8:11] = self.target - self.pos
            spatial[11:14] = util.quaternion_to_rpy(self.rot_internal)
            spatial[14] = self.distance
            spatial[15:18] = util.quaternion_to_rpy(self.target_rot)
            spatial[18] = util.quaternion_similarity(self.rot_internal, self.target_rot)
        if not self.use_joints:
            spatial = spatial[6:]
        if not self.use_rotations:
            spatial = spatial[:-4]

        return {
            'spatial': spatial,
            'lidar': self.lidar_indicator if not self.use_raw_lidar else self.lidar_indicator_raw
        }

    def step(self, action):

        # save old position
        #ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        #self.pos = np.array(ee_link_state[4])
        self.pos_last = self.pos
        #self.rot = np.array(ee_link_state[5])

        # update timestamp
        time_last = self.time
        self.time = time()

        if not self.use_joints:
            # transform action
            pos_delta = action[:3] * self.pos_speed
            rpy_delta = action[3:] * self.rot_speed

            # add the action to the current state
            rot_rpy = util.quaternion_to_rpy(self.rot_internal)
            pos_desired = self.pos + pos_delta
            rpy_desired = rot_rpy + rpy_delta
            
            # check if desired rpy violates the limits
            upper_limit_mask = rpy_desired > self.rpy_upper_limits
            lower_limit_mask = rpy_desired < self.rpy_lower_limits
            rpy_desired[upper_limit_mask] = self.rpy_upper_limits[upper_limit_mask]
            rpy_desired[lower_limit_mask] = self.rpy_lower_limits[lower_limit_mask]

            # convert to quaternion, necesarry for pybullet
            quat_desired = self._quat_w_to_ee(util.rpy_to_quaternion(rpy_desired))

            # move the robot to the new positions and get the associated joint config
            joints_old = self.joints
            self.joints = self._movep(pos_desired, quat_desired)
        else:
            # transform action
            joint_delta = action * self.joint_speed

            # add action to current state
            joints_desired = self.joints + joint_delta

            # check if desired joints would violate any joint range constraints
            upper_limit_mask = joints_desired > self.joints_upper_limits
            lower_limit_mask = joints_desired < self.joints_lower_limits
            # set those joints to their respective max/min
            joints_desired[upper_limit_mask] = self.joints_upper_limits[upper_limit_mask]
            joints_desired[lower_limit_mask] = self.joints_lower_limits[lower_limit_mask]

            # execute movement by setting the desired joint state
            joints_old = self.joints
            self.joints = self._movej(joints_desired)

        # get new state info
        ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        self.pos = np.array(ee_link_state[4])
        self.rot = np.array(ee_link_state[5])
        self.rot_internal = self._quat_ee_to_w(np.array(ee_link_state[1]))
        self.lidar_indicator, self.lidar_indicator_raw = self._get_lidar_indicator()
        self.pos_vel = (self.pos - self.pos_last) / (self.time - time_last)
        self.joints_vel = (self.joints - joints_old) / (self.time - time_last)

        # increment steps
        self.steps_current_episode += 1
        self.steps_total += 1

        # logging
        if self.logging > 0:
            log = []
            log.append(self.time)
            log += self.pos.tolist()
            log += self.rot_internal.tolist()
            log.append(self.distance)
            log += self.pos_vel.tolist()
            log += self.joints_vel.tolist()
            self.log.append(log)

        # get reward and info
        reward, done, info = self._reward()

        return self._get_obs(), reward, done, info

    def _reward(self):
        
        # get closestpoints
        # quadratischef unktion: c 15000 n 35
        # steps vor collision angucken

        collided = self._collision()
        clearance_failed = bool(np.min(self.lidar_indicator_raw) < self.minimal_clearance)
        out_of_bounds = False
        is_in_target = False
        is_in_desired_orientation = False

        distance_cur = np.linalg.norm(self.target - self.pos)
        distance_last = np.linalg.norm(self.target - self.pos_last)

        #quaternion_similarity = np.abs(np.dot(self.rot_internal, self.target_rot))
        quaternion_similarity = util.quaternion_similarity(self.rot_internal, self.target_rot)

        # update the max_dist variable
        if distance_cur > self.max_dist:
            self.max_dist = distance_cur

        # check if out of bounds
        if self._is_out_of_bounds(distance_cur):
            done = True
            is_success = False
            out_of_bounds = True
            reward = self.min_reward
        elif not (collided or clearance_failed):
            is_in_target = self._is_in_target_box() if not self.target_zone_sphere else (distance_cur <= self.ee_pos_reward_thresh)
            is_in_desired_orientation = quaternion_similarity > (1 - self.ee_rot_reward_thresh) if self.use_rotations else True
            if is_in_target and is_in_desired_orientation:
                reward = self.max_reward
                done = True
                is_success = True
                self.successes += 1
            else: 
                distance_reward = -0.01 * distance_cur if not is_in_target else 0
                rotation_reward = -0.01 * (1 - quaternion_similarity) * (0 if distance_cur > 3 * self.ee_pos_reward_thresh else 1)  # only score if sufficiently close to the target zone
                obstacle_avoidance_reward = 0

                reward = 0
                reward += distance_reward
                reward += obstacle_avoidance_reward
                if self.use_rotations:
                    reward += rotation_reward

                done = False
                is_success = False

        else:
            reward = self.min_reward
            done = True
            is_success = False
            self.collisions += (1 if collided else 0)
            self.clearance_failures += (1 if clearance_failed else 0)

        # normalize reward if needed
        if self.normalize:
            reward = self.normalizing_constant_reward_a * reward + self.normalizing_constant_reward_b

        timeout = False
        if self.steps_current_episode > self.max_episode_steps:
            done = True
            timeout = True

        if self.train:
            # collect various stats
            if done:
                self.success_buffer.append(is_success)
                if len(self.success_buffer) > self.stats_buffer_size:
                    self.success_buffer.pop(0)
                self.success_rot_buffer.append(is_in_desired_orientation)
                if len(self.success_rot_buffer) > self.stats_buffer_size:
                    self.success_rot_buffer.pop(0)
                self.success_pos_buffer.append(is_in_target)
                if len(self.success_pos_buffer) > self.stats_buffer_size:
                    self.success_pos_buffer.pop(0)
                self.collision_buffer.append(collided)
                if len(self.collision_buffer) > self.stats_buffer_size:
                    self.collision_buffer.pop(0)
                self.clearance_buffer.append(clearance_failed)
                if len(self.clearance_buffer) > self.stats_buffer_size:
                    self.clearance_buffer.pop(0)
                self.timeout_buffer.append(timeout)
                if len(self.timeout_buffer) > self.stats_buffer_size:
                    self.timeout_buffer.pop(0)
                self.out_of_bounds_buffer.append(out_of_bounds)
                if len(self.out_of_bounds_buffer) > self.stats_buffer_size:
                    self.out_of_bounds_buffer.pop(0)
        
        self.episode_discounted_reward += (self.gamma**self.steps_current_episode) * reward
        self.episode_distance += distance_cur
        self.is_success = is_success

        info = {
            'step': self.steps_current_episode,
            'steps_total': self.steps_total,
            'episodes': self.episodes,
            'done': done,
            'is_success': is_success,
            'success_rate': self.successes/self.episodes if not self.train else np.average(self.success_buffer),
            'success_rate_pos': 0 if not self.train else np.average(self.success_pos_buffer),
            'success_rate_rot': 0 if not self.train else np.average(self.success_rot_buffer),
            'collided': collided,
            'collision_rate': self.collisions/self.episodes if not self.train else np.average(self.collision_buffer),
            'clearance_failed': clearance_failed,
            'clearance_failure_rate': self.clearance_failures/self.episodes if not self.train else np.average(self.clearance_buffer),
            'timeout': timeout,
            'timeout_rate': 0 if not self.train else np.average(self.timeout_buffer),
            'out_of_bounds': out_of_bounds,
            'out_of_bounds_rate': 0 if not self.train else np.average(self.out_of_bounds_buffer),
            'reward': reward,
            'episode_reward': self.episode_discounted_reward,
            'distance': distance_cur,
            'rotation_similarity': quaternion_similarity,
            'episode_distance': self.episode_distance,
            'distance_threshold': self.ee_pos_reward_thresh,
            'rotation_threshold': self.ee_rot_reward_thresh
        }

        if done:
            info_string = ""
            for key in info:
                info_string += key + ": " + str(round(info[key], 3)) + ", "
            print(info_string)
            if self.logging == 2:
                np.savetxt('./model/logs/'+str(self.episodes)+'.txt', np.asarray(self.log))

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
            if obj != self.robot:
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

    def _cast_lidar_rays(self, ray_min=0.02, ray_max=0.3, ray_num_side=6, ray_num_forward=12, ray_num_circle=8, render=False):
        """
        Casts rays from various positions on the torch and receives collision information from that. Currently only adjusted for the MRW tool.
        """
        ray_froms = []
        ray_tops = []
        # get the frame of the torch tip
        frame_torch_tip = util.quaternion_to_matrix(util.quaternion_multiply(self.rot_internal, np.array([ 0, 1, 0, 0 ])))
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
        for i in range(ray_num_circle):
            ai = i*np.pi/(ray_num_circle/2)
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
        for i in range(ray_num_circle):
            ai = i*np.pi/(ray_num_circle/2)
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
                if result[0] == -1 or result[0] == self.robot:
                    pyb.addUserDebugLine(ray_froms[index], ray_tops[index], missRayColor)
                else:
                    pyb.addUserDebugLine(ray_froms[index], ray_tops[index], hitRayColor)
        return results

    def _get_lidar_indicator(self, buckets=20):

        lidar_results_raw = np.array(self._cast_lidar_rays(ray_min=self.ray_start, ray_max=self.ray_end, ray_num_side=self.ray_num_side, ray_num_forward=self.ray_num_forward, ray_num_circle=self.ray_num_side_circle), dtype=object)
        lidar_results = lidar_results_raw[:,2]  # only use the distance information
        indicator_raw = np.zeros((2 * self.ray_num_side_circle + 1,), dtype=np.float32)
        indicator = np.zeros((2 * self.ray_num_side_circle + 1,), dtype=np.float32)

        # values that are used to convert the raw pybullet range data to a discrete indicator
        raw_bucket_size = 1/buckets  # 1 is the range of the pybullet lidar data (from 0 to 1)
        indicator_label_diff = 2/buckets  # 2 is the range of the indicator data (from -1 to 1)
        # small function to convert between the data
        raw_to_indicator = lambda x : 1 if x >= 0.99 else round((np.max([(np.ceil(x/raw_bucket_size)-1),0]) * indicator_label_diff - 1),5)
        # short explanation: takes a number between 0 and 1, assigns it a bucket in the range, and returns the corresponding bucket in the range of -1 and 1
        # the round is thrown in there to prevent weird numeric appendages that came up in testing, e.g. 0.200000000004, -0.199999999999 or the like

        # side note: the array slices here are very complicated, but they basically just count up the rays in the order they are in the lidar_results object
        # as pybullet will output them in the lider_cylinder method. Instead of hardcoding the slices they are kept as variables such that the amount of rays can be changed at will
        # tip front indicator
        lidar_min = lidar_results[0:(2 * self.ray_num_forward + 1)].min()  # 1 ray going straight forward + 2 cones of ray_num_forward rays around it
        #indicator[0] = 1 if lidar_min >= 0.99 else (0.5 if 0.75 <= lidar_min < 0.99 else (0 if 0.5 <= lidar_min < 0.75 else (-0.5 if 0.25 <= lidar_min < 0.5 else -1)))
        indicator[0] = raw_to_indicator(lidar_min)
        indicator_raw[0] = lidar_min * (self.ray_end - self.ray_start) + self.lidar_offset
        # tip sides indicators
        for i in range(self.ray_num_side_circle):
            lidar_min = lidar_results[(2 * self.ray_num_forward + 1 + i * self.ray_num_side):(2 * self.ray_num_forward + 1 + (i + 1) * self.ray_num_side)].min()
            #indicator[1+i] = 1 if lidar_min >= 0.99 else (0.5 if 0.75 <= lidar_min < 0.99 else (0 if 0.5 <= lidar_min < 0.75 else (-0.5 if 0.25 <= lidar_min < 0.5 else -1)))
            indicator[1+i] = raw_to_indicator(lidar_min)
            indicator_raw[1+i] = lidar_min * (self.ray_end - self.ray_start) + self.lidar_offset
        # grip sides indicators
        for i in range(self.ray_num_side_circle):
            lidar_min = lidar_results[(2 * self.ray_num_forward + 1 + self.ray_num_side_circle * self.ray_num_side + i * self.ray_num_side):(2 * self.ray_num_forward + 1 + self.ray_num_side_circle * self.ray_num_side + (i + 1) * self.ray_num_side)].min()
            #indicator[1+8+i] = 1 if lidar_min >= 0.99 else (0.5 if 0.75 <= lidar_min < 0.99 else (0 if 0.5 <= lidar_min < 0.75 else (-0.5 if 0.25 <= lidar_min < 0.5 else -1)))
            indicator[1+self.ray_num_side_circle+i] = raw_to_indicator(lidar_min)
            indicator_raw[1+self.ray_num_side_circle+i] = lidar_min * (self.ray_end - self.ray_start) + self.lidar_offset
        return indicator, indicator_raw 

    def _is_in_target_box(self):
        """
        Returns a bool that indicates whether the current position is within the target box or not.
        """
        # first transform the current position into the target frame
        # this makes checking wether it's in the box the much easier
        position_in_target_frame = util.rotate_vec(util.quaternion_invert(self.target_transform), (self.pos - self.target))

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

    def manual_control(self):
        # code to manually control the robot in real time
        rpy = util.quaternion_to_rpy(self.rot_internal)
        roll = pyb.addUserDebugParameter("r", -4.0, 4.0, rpy[0])
        pitch = pyb.addUserDebugParameter("p", -4.0, 4.0, rpy[1])
        yaw = pyb.addUserDebugParameter("y", -4.0, 4.0, rpy[2])
        fwdxId = pyb.addUserDebugParameter("fwd_x", -4, 8, self.pos[0])
        fwdyId = pyb.addUserDebugParameter("fwd_y", -4, 8, self.pos[1])
        fwdzId = pyb.addUserDebugParameter("fwd_z", -1, 3, self.pos[2])
        fwdxIdbase = pyb.addUserDebugParameter("fwd_x_base", -4, 8, 0)
        fwdyIdbase = pyb.addUserDebugParameter("fwd_y_base", -4, 8, 0)
        x_base = 0
        y_base = 0
        oldybase = 0
        oldxbase = 0

        pyb.addUserDebugLine([0,0,0],[0,0,1],[0,0,1],parentObjectUniqueId=self.robot, parentLinkIndex= self.end_effector_link_id)
        pyb.addUserDebugLine([0,0,0],[0,1,0],[0,1,0],parentObjectUniqueId=self.robot, parentLinkIndex= self.end_effector_link_id)
        pyb.addUserDebugLine([0,0,0],[1,0,0],[1,0,0],parentObjectUniqueId=self.robot, parentLinkIndex= self.end_effector_link_id)

        lineID = 0

        while True:
            if lineID:
                pyb.removeUserDebugItem(lineID)

            if x_base != oldxbase or y_base != oldybase:
                pyb.resetBasePositionAndOrientation(self.robot, np.array([x_base, y_base, self.ceiling_mount_height]), pyb.getQuaternionFromEuler([np.pi, 0., 0.]))
            # read inputs from GUI
            qrr = pyb.readUserDebugParameter(roll)
            qpr = pyb.readUserDebugParameter(pitch)
            qyr = pyb.readUserDebugParameter(yaw)
            x = pyb.readUserDebugParameter(fwdxId)
            y = pyb.readUserDebugParameter(fwdyId)
            z = pyb.readUserDebugParameter(fwdzId)
            oldxbase = x_base
            oldybase = y_base
            x_base = pyb.readUserDebugParameter(fwdxIdbase)
            y_base = pyb.readUserDebugParameter(fwdyIdbase)

            # build quaternion from user input
            command_quat = util.rpy_to_quaternion([qrr,qpr,qyr])
            command_quat = self._quat_w_to_ee(command_quat)

            self._movep([x,y,z],command_quat)
            # get new state info
            ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
            self.pos = np.array(ee_link_state[4])
            self.rot = np.array(ee_link_state[5])
            self.rot_internal = self._quat_ee_to_w(np.array(ee_link_state[1]))
            self.lidar_indicator, self.lidar_indicator_raw = self._get_lidar_indicator()
            self.max_dist = np.linalg.norm(self.pos - self.target)
            self.min_dist = self.max_dist

            ee_link_state = pyb.getLinkState(self.robot, self.base_link_id, computeForwardKinematics=True)

            #print(self.lidar_indicator_raw)
            #print(pyb.getClosestPoints(self.robot, self.obj_ids[2], 20, self.end_effector_link_id))
            print(self.pos)
            print(self.pos - np.array(ee_link_state[4]))

            lineID = pyb.addUserDebugLine([x,y,z], self.pos.tolist(), [0,0,0])

            self._reward()



    ###################
    # utility methods #
    ###################

    def _is_out_of_bounds(self, distance):
        """
        Returns boolean if the robot ee position is out of bounds
        """
        if distance < 0.5:
            return False  # allow going out of bounds slightly as long as it's close to the target
        if self.pos[0] < self.x_min or self.pos[0] > self.x_max:
            return True
        if self.pos[1] < self.y_min or self.pos[1] > self.y_max:    
            return True
        if self.pos[2] < self.z_min or self.pos[2] > self.z_max:
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
        # test: get a random offset of at most length 0.5 to move the base away such that the task becomes more varied
        random_offset = np.random.uniform(low=-1, high=1, size=2)
        random_offset = random_offset / np.linalg.norm(random_offset)
        random_offset = random_offset * np.random.uniform(low=0, high=0.25)
        self.target_base = np.average(positions, axis=0)[:2] + self.base_offset + random_offset

        # set also a target rotation if needed:
        if self.use_rotations:
            rotations = [util.matrix_to_quaternion(ele[:3,:3]) for ele in frame["pose_frames"]]
            self.target_rot = rotations[0]

    def _show_target(self):
        """
        Creates a visual indicator around the target in the rendered Pybullet simulation
        """
        if self.target_zone_sphere:
            visual_id = pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=self.ee_pos_reward_thresh, rgbaColor=[0, 1, 0, 1])
            point_id = pyb.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=visual_id,
                        basePosition=self.target,
                        )
        else:
            # calculate the center point such that the rectangle will be displayed in the correct way
            # this is necessary because the pybullet interface for boxes works with halflenghts emanating from the center
            center_in_target_frame = np.array([self.target_x_max/2.0, 
                                            self.target_y_max/2.0,
                                            (self.target_m_z_max + self.target_p_z_max)/2.0])  # the target in target frame is of course 0,0,0, so no need to add it here
            #center_in_world_frame = util.rotate_vec(util.quaternion_invert(self.target_transform), center_in_target_frame) + self.target
            center_in_world_frame = util.rotate_vec(self.target_transform, center_in_target_frame) + self.target
            halfExtents = np.array([self.target_x_max/2.0, self.target_y_max/2.0, (np.abs(self.target_m_z_max) + self.target_p_z_max)/2.0])
            third_dim = np.cross(self.target_norms[0], self.target_norms[1])
            box = pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0, 1, 0, 1])
            pyb.createMultiBody(baseVisualShapeIndex=box, basePosition=center_in_world_frame, baseOrientation=self.target_transform)

            pyb.addUserDebugLine(self.target, self.target+third_dim, [0,0,1])
            pyb.addUserDebugLine(self.target, np.array(self.target)+np.array(self.target_norms[0]),[1,0,0])
            pyb.addUserDebugLine(self.target, np.array(self.target)+np.array(self.target_norms[1]),[0,1,0])

            if self.display:
                rpy = util.quaternion_to_rpy(self.target_transform)
                pyb.resetDebugVisualizerCamera(1, (rpy[2] + np.pi/4)*180/np.pi, -35, self.target)

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
            'rot_threshold': self.ee_rot_reward_thresh,
            'episodes': self.episodes,
            'successes': self.successes,
            'collisions': self.collisions,
            'clearance_failures': self.clearance_failures,
            'success_buffer': self.success_buffer,
            'success_rot_buffer': self.success_rot_buffer,
            'success_pos_buffer': self.success_pos_buffer,
            'collision_buffer': self.collision_buffer,
            'clearance_buffer': self.clearance_buffer,
            'timeout_buffer': self.timeout_buffer,
            'out_of_bounds_buffer': self.out_of_bounds_buffer
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
            self.reload = True
            self.id = load_dict["id"]
            self.steps_total = load_dict["steps"]
            self.ee_pos_reward_thresh = load_dict["dist_threshold"]
            self.ee_rot_reward_thresh = load_dict["rot_threshold"]
            self.episodes = load_dict["episodes"]
            self.successes = load_dict["successes"]
            self.success_buffer = load_dict["success_buffer"]
            self.success_rot_buffer = load_dict["success_rot_buffer"]
            self.success_pos_buffer = load_dict["success_pos_buffer"]
            self.collisions = load_dict["collisions"]
            self.collision_buffer = load_dict["collision_buffer"]
            self.clearance_failures = load_dict["clearance_failures"]
            self.clearance_buffer = load_dict["clearance_buffer"]
            self.timeout_buffer = load_dict["timeout_buffer"]
            self.out_of_bounds_buffer = load_dict["out_of_bounds_buffer"]
        except FileNotFoundError:
            pass

class PathingEnvironmentPybulletWithCamera(PathingEnvironmentPybullet):

    def __init__(self, env_config):
        super().__init__(env_config)

        camera_height = 32
        camera_width = 32
        self.projection_matrix = pyb.computeProjectionMatrixFOV(60, 1, 0.001, 100)
        self.view_matrix = None
        self.take_image_every_x_steps = 10
        self.observation_space = gym.spaces.Dict(
            {
            'spatial': gym.spaces.Box(low=-1 if self.normalize else -8, high=1 if self.normalize else 8, shape=(13 if self.use_joints else 7,), dtype=np.float32),
            'lidar': gym.spaces.Box(low=0 if self.use_raw_lidar else -1, high=1, shape=(17,), dtype=np.float32),
            'camera': gym.spaces.Box(low=0, high=5, shape=(camera_height, camera_width))  
            }
        )

    def reset(self):
        
        complete_reset = self.episodes == 0 or self.episodes % self.reset_mesh_after_episodes == 0 or self.reload
        self.reload = False

        # clear out stored objects and reset the simulation if necessary
        if complete_reset:
            self.obj_ids = []
            pyb.resetSimulation()

        # set the step and episode counters correctly
        self.steps_current_episode = 0
        self.episodes += 1
        self.episode_reward = 0
        self.episode_distance = 0

        # stop pybullet rendering for performance
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)

        if complete_reset:
            # rebuild environment
            # load in the ground plane
            self._add_object("workspace/plane.urdf", [0, 0, -0.01])

            # load in the mesh of the welding part
            # info: the method will load the mesh at given index within the self.dataset variable
            #self.mesh_file_index = np.random.choice(range(len(self.dataset["filenames"]))) 
            self.mesh_file_index = self.dataset["filenames"].index("201910204483_R1.urdf")
            self.welding_mesh = self._add_object("objects/"+self.dataset["filenames"][self.mesh_file_index], self.xyz_offset)
        # set the target and base target (this is done in the method calls below)
        while True:
            try:
                target_index = np.random.choice(range(len(self.dataset["frames"][self.mesh_file_index])))  # pick a random target from the welding part's xml
                break
            except ValueError:
                self.mesh_file_index = np.random.choice(range(len(self.dataset["filenames"])))  # sometimes there are no targets in a given entry of the xml and the code will throw an error, this is to prevent that and choose another xml
        target_index = 0
        tool_old = self.tool  # remember the old tool for checking in a moment
        self._set_target(self.mesh_file_index, target_index)
        self._set_target_box(self.ee_pos_reward_thresh)
        if self.show_target:
            self._show_target()
        #tmp, set to one tool for now, full implementation for both tools later on
        self.tool=0

        # load in the robot, the correct tool was set above in the _set_target method call
        if self.tool == tool_old and not complete_reset:
            # if the tool has stayed the same, we can avoid an expensive load in of the robot mesh from the drive and simply change its current position
            pyb.resetBasePositionAndOrientation(self.robot, np.append(self.target_base, self.ceiling_mount_height), pyb.getQuaternionFromEuler([np.pi, 0, 0]))
        else:
            # if the tool has changed (or we're running the env for the first time or completely reseting), loading in the robot mesh is necessary
            if self.tool:
                self.robot = self._add_object("kr16/kr16_tand_gerad.urdf", np.append(self.target_base, self.ceiling_mount_height), pyb.getQuaternionFromEuler([np.pi, 0, 0]))
            else:
                self.robot = self._add_object("kr16/kr16_mrw510.urdf", np.append(self.target_base, self.ceiling_mount_height), pyb.getQuaternionFromEuler([np.pi, 0, 0]))      

        # get the joint ids of the robot and set the joints to their resting position
        joints = [pyb.getJointInfo(self.robot, i) for i in range(pyb.getNumJoints(self.robot))]
        self.joint_ids = [j[0] for j in joints if j[2] == pyb.JOINT_REVOLUTE]
        self._set_joint_state(self.resting_pose_angles)
        self.joints = self.resting_pose_angles
        # in training set the arm to a random position sometimes to force a variety of starting positions
        if self.train:
            if np.random.random() < -1:
                start = True
                while start or self._collision():
                    start = False
                    random_xyz_offset = np.random.random(3) * 2 - 1
                    random_xyz_offset[2] = abs(random_xyz_offset[2])
                    random_xyz_offset = random_xyz_offset * (max(self.ee_pos_reward_thresh * 1.5, np.random.random()) / np.linalg.norm(random_xyz_offset))
                    random_rpy = np.random.uniform(low=self.rpy_lower_limits, high=self.rpy_upper_limits, size=3)
                    self.joints = self._movep(self.target +  random_xyz_offset, self._quat_w_to_ee(util.rpy_to_quaternion(random_rpy)))
                    start = np.linalg.norm(np.array(pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)[4]) - (random_xyz_offset + self.target)) > self.ee_pos_reward_thresh * 1.5

        # get state information
        ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        self.pos = np.array(ee_link_state[4])
        self.pos_last = self.pos
        self.rot = np.array(ee_link_state[5])
        self.rot_internal = self._quat_ee_to_w(np.array(ee_link_state[1]))
        self.lidar_indicator, self.lidar_indicator_raw = self._get_lidar_indicator()
        self.max_dist = np.linalg.norm(self.pos - self.target)
        self.min_dist = self.max_dist
        self.distance = self.max_dist

        rot_internal = np.array(ee_link_state[1])
        self.view_matrix = pyb.computeViewMatrix(self.pos, self.pos-0.01*util.rotate_vec(rot_internal, np.array([0,0,1])), util.rotate_vec(rot_internal ,np.array([1,0,0])))
        self.camera_image = self._get_camera_image()

        if self.train:
            if len(self.success_buffer) != 0:
                success_rate = np.average(self.success_buffer) 
            else:
                success_rate = 0
            if success_rate < 0.8 and self.ee_pos_reward_thresh < self.ee_pos_reward_thresh_max and not self.is_success:
                self.ee_pos_reward_thresh += util.linear_interpolation(self.ee_pos_reward_thresh, self.ee_pos_reward_thresh_min, self.ee_pos_reward_thresh_max, self.ee_pos_reward_thresh_min_increment, self.ee_pos_reward_thresh_max_increment) / 50
            elif success_rate > 0.8 and self.ee_pos_reward_thresh > self.ee_pos_reward_thresh_min and self.is_success:
                self.ee_pos_reward_thresh -= util.linear_interpolation(self.ee_pos_reward_thresh, self.ee_pos_reward_thresh_min, self.ee_pos_reward_thresh_max, self.ee_pos_reward_thresh_min_increment, self.ee_pos_reward_thresh_max_increment) 
            if self.ee_pos_reward_thresh < self.ee_pos_reward_thresh_min:
                self.ee_pos_reward_thresh = self.ee_pos_reward_thresh_min
            if self.ee_pos_reward_thresh > self.ee_pos_reward_thresh_max:
                self.ee_pos_reward_thresh = self.ee_pos_reward_thresh_max
        
        self.is_success = False

        # turn on rendering again
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)

        return self._get_obs()
    
    def step(self, action):
        
        _, reward, done, info = super().step(action)

        if ((self.steps_total -  1) % self.take_image_every_x_steps) == 0:
            ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
            rot_internal = np.array(ee_link_state[1])
            self.view_matrix = pyb.computeViewMatrix(self.pos, self.pos-0.01*util.rotate_vec(rot_internal, np.array([0,0,1])), util.rotate_vec(rot_internal ,np.array([1,0,0])))
            self.camera_image = self._get_camera_image()

        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        spatial = np.zeros(13)
        if self.normalize:   
            spatial[:6] = np.multiply(self.normalizing_constant_obs_a[:6], self.joints) + self.normalizing_constant_obs_b[:6]
            spatial[6:9] = np.multiply(self.normalizing_constant_obs_a[6:9], (self.target - self.pos)) + self.normalizing_constant_obs_b[6:9]
            spatial[9:12] = np.multiply(self.normalizing_constant_obs_a[9:12], util.quaternion_to_rpy(self.rot_internal)) +  self.normalizing_constant_obs_b[9:12]
            spatial[12] = self.normalizing_constant_obs_a[12] * self.distance + self.normalizing_constant_obs_b[12]
        else:
            spatial[:6] = self.joints
            spatial[6:9] = self.target - self.pos
            spatial[9:12] = util.quaternion_to_rpy(self.rot_internal)
            spatial[12] = self.distance

        return {
            'spatial': spatial if self.use_joints else spatial[6:],
            'lidar': self.lidar_indicator if not self.use_raw_lidar else self.lidar_indicator_raw,
            'camera': self.camera_image
        }
    
    def _get_camera_image(self):
        img = (pyb.getCameraImage(32, 32, self.view_matrix, self.projection_matrix))[3]
        img = np.array(img)
        img = 0.001 * 100 / (100 - (100 - 0.001) * img)
        img = np.reshape(img, (32, 32))
        return img

class PathingEnvironmentPybulletTestingObstacles(PathingEnvironmentPybullet):

    def __init__(self, env_config):
        super().__init__(env_config)

        self.target_zone_sphere = True

        self.x_max = 0.4
        self.x_min = -0.4
        self.y_max = 0.7
        self.y_min = 0.3
        self.z_max = 0.4
        self.z_min = 0.2
        # constants to make normalizing more efficient
        max_distance = np.ceil(np.sqrt(self.x_max**2 + self.y_max**2 + self.z_max**2))  # this is the maximum possible distance given the workspace
        vec_distance_max = np.array([self.x_max, self.y_max, self.z_max])
        vec_distance_min = -1 * vec_distance_max
        self.normalizing_constant_obs_a = np.zeros(19)
        self.normalizing_constant_obs_a[:6] = 2 / self.joints_range
        self.normalizing_constant_obs_a[6:8] = 2 / (vec_distance_max - vec_distance_min)[2:]
        self.normalizing_constant_obs_a[8:11] = 2 / (vec_distance_max - vec_distance_min)
        self.normalizing_constant_obs_a[11:14] = 2 / (self.rpy_upper_limits - self.rpy_lower_limits)
        self.normalizing_constant_obs_a[14] = 1 / max_distance
        self.normalizing_constant_obs_a[15:18] = 2 / (self.rpy_upper_limits - self.rpy_lower_limits)
        self.normalizing_constant_obs_a[18] = 0  # doesnt need normalizing
        self.normalizing_constant_reward_a = 2 / (self.max_reward - self.min_reward)

        self.normalizing_constant_obs_b = np.zeros(19)
        self.normalizing_constant_obs_b[:6] = np.ones(6) - np.multiply(self.normalizing_constant_obs_a[:6], self.joints_upper_limits)
        self.normalizing_constant_obs_b[6:8] = np.ones(2) - np.multiply(self.normalizing_constant_obs_a[6:8], vec_distance_max[2:])
        self.normalizing_constant_obs_b[8:11] = np.ones(3) - np.multiply(self.normalizing_constant_obs_a[8:11], vec_distance_max)
        self.normalizing_constant_obs_b[11:14] = np.ones(3) - np.multiply(self.normalizing_constant_obs_a[11:14], self.rpy_upper_limits)
        self.normalizing_constant_obs_b[14] = 1 - self.normalizing_constant_obs_a[14] * max_distance
        self.normalizing_constant_obs_b[15:18] = np.ones(3) - np.multiply(self.normalizing_constant_obs_a[15:18], self.rpy_upper_limits)
        self.normalizing_constant_obs_b[18] = 0  # doesnt need normalizing
        self.normalizing_constant_reward_b = 1 - self.normalizing_constant_reward_a * self.max_reward

        self.inits = np.zeros((500,3))
        self.rots = np.zeros((500,4))
        self.idx = 0

        if self.train:
            if np.random.random() < 0.75:
                self.test_case = 0
            else:
                self.test_case = 1
        else:
            self.test_case = 0
        self.moving_plate = 0

        if self.train:
            self.max_episode_steps = 750

    def reset(self):

        self.time = time()

        self.reload = False
        rand = np.float32(np.random.rand(3,))
        init_x = (self.x_max + self.x_min)/2+0.5*(rand[0]-0.5)*(self.x_max - self.x_min)
        init_y = (self.y_max + self.y_min)/2+0.5*(rand[1]-0.5)*(self.y_max - self.y_min)
        init_z = (self.z_max + self.z_min)/2+0.5*(rand[2]-0.5)*(self.z_max - self.z_min)
        self.init_home = np.array([init_x, init_y, init_z])

        # clear out stored objects and reset the simulation if necessary
        self.obj_ids = []
        pyb.resetSimulation()

        # set the step and episode counters correctly
        self.steps_current_episode = 0
        self.episodes += 1
        self.episode_reward = 0
        self.episode_distance = 0

        # stop pybullet rendering for performance
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)

        # rebuild environment
        # load in the ground plane
        self._add_object("workspace/plane.urdf", [0, 0, -0.01])

        #tmp, set to one tool for now, full implementation for both tools later on
        self.tool=0

        val = False
        while not val:         
            rand = np.float32(np.random.rand(3,))
            target_x = self.x_min + rand[0]*(self.x_max - self.x_min)
            target_y = self.y_min + rand[1]*(self.y_max - self.y_min)
            target_z = self.z_min + rand[2]*(self.z_max - self.z_min)
            self.target_position = [target_x, target_y, target_z]
            if np.linalg.norm(np.array(self.init_home)-np.array(self.target_position),None)>0.4:
                val = True
        self.target_norms = [np.array([1,0,0]), np.array([0,1,0])]
        self.target = np.array(self.target_position)

        # load in the robot, the correct tool was set above in the _set_target method call
        self.target_base = ((self.target - self.init_home)/2 + self.init_home)[:2]
        self.target_base = np.array([0.0,-0.12])
        self.robot = self._add_object("kr16/kr16_mrw510.urdf", np.append(self.target_base, self.ceiling_mount_height), pyb.getQuaternionFromEuler([np.pi, 0, 0]))      

        # get the joint ids of the robot and set the joints to their resting position
        joints = [pyb.getJointInfo(self.robot, i) for i in range(pyb.getNumJoints(self.robot))]
        self.joint_ids = [j[0] for j in joints if j[2] == pyb.JOINT_REVOLUTE]
        self._set_joint_state(self.resting_pose_angles)
        self.joints = self.resting_pose_angles
        self.joints_vel = np.zeros(6)

        if self.test_case == 0:
            self._build_random_obstacles()
        elif self.test_case == 1:
            self._build_test_case_1()
        elif self.test_case == 2:
            self._build_test_case_2()
        elif self.test_case == 3:
            self._build_test_case_3()

        if self.test_case == 0:
            self.joints = self._movep(self.init_home, [0,0,0,1])
        else:
            self.joints = self._movep(self.init_home, self.init_orn)
        # set the target and base target (this is done in the method calls below)
        # print (target_position)
        if not self.target_zone_sphere:
            self._set_target_box(self.ee_pos_reward_thresh)
        if self.show_target:
            self._show_target()

        # in training set the arm to a random position sometimes to force a variety of starting positions
        if self.train:
            if np.random.random() < -1:
                start = True
                while start or self._collision():
                    start = False
                    random_xyz_offset = np.random.random(3) * 2 - 1
                    random_xyz_offset[2] = abs(random_xyz_offset[2])
                    random_xyz_offset = random_xyz_offset * (max(self.ee_pos_reward_thresh * 1.5, np.random.random()) / np.linalg.norm(random_xyz_offset))
                    random_rpy = np.random.uniform(low=self.rpy_lower_limits, high=self.rpy_upper_limits, size=3)
                    self.joints = self._movep(self.target +  random_xyz_offset, self._quat_w_to_ee(util.rpy_to_quaternion(random_rpy)))
                    start = np.linalg.norm(np.array(pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)[4]) - (random_xyz_offset + self.target)) > self.ee_pos_reward_thresh * 1.5

        # get state information
        ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        self.pos = np.array(ee_link_state[4])
        self.pos_vel = np.zeros(3)
        self.pos_last = self.pos
        self.rot = np.array(ee_link_state[5])
        self.rot_internal = self._quat_ee_to_w(np.array(ee_link_state[1]))
        self.lidar_indicator, self.lidar_indicator_raw = self._get_lidar_indicator()
        self.max_dist = np.linalg.norm(self.pos - self.target)
        self.min_dist = self.max_dist
        self.distance = self.max_dist

        # set the success thresholds
        if self.train:
            if len(self.success_buffer) != 0:
                success_rate_rot = np.average(self.success_rot_buffer) 
                success_rate_pos = np.average(self.success_pos_buffer)
            else:
                success_rate_rot = 0
                success_rate_pos = 0
            if success_rate_pos > 0.8:
                if self.ee_pos_reward_thresh > self.ee_pos_reward_thresh_min and self.is_success:
                    self.ee_pos_reward_thresh -= util.linear_interpolation(self.ee_pos_reward_thresh, self.ee_pos_reward_thresh_min, self.ee_pos_reward_thresh_max, self.ee_pos_reward_thresh_min_increment, self.ee_pos_reward_thresh_max_increment)
            if success_rate_rot > 0.8:
                if self.ee_rot_reward_thresh > self.ee_rot_reward_thresh_min and self.is_success:
                    self.ee_rot_reward_thresh -= util.linear_interpolation(self.ee_rot_reward_thresh, self.ee_rot_reward_thresh_min, self.ee_rot_reward_thresh_max, self.ee_rot_reward_thresh_min_increment, self.ee_rot_reward_thresh_max_increment) 
            if self.ee_pos_reward_thresh < self.ee_pos_reward_thresh_min:
                self.ee_pos_reward_thresh = self.ee_pos_reward_thresh_min
            if self.ee_pos_reward_thresh > self.ee_pos_reward_thresh_max:
                self.ee_pos_reward_thresh = self.ee_pos_reward_thresh_max
            if self.ee_rot_reward_thresh < self.ee_rot_reward_thresh_min:
                self.ee_rot_reward_thresh = self.ee_rot_reward_thresh_min
            if self.ee_rot_reward_thresh > self.ee_rot_reward_thresh_max:
                self.ee_rot_reward_thresh = self.ee_rot_reward_thresh_max
        
        self.is_success = False
        if self.logging > 0:
            self.log = []
            log = []
            log.append(self.time)
            log += self.pos.tolist()
            log += self.rot_internal.tolist()
            log.append(self.distance)
            log += self.pos_vel.tolist()
            log += self.joints_vel.tolist()
            self.log.append(log)

        # turn on rendering again
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)

        return self._get_obs()

    def step(self,action):
        obs, reward, done, info = super().step(action)
        if self.moving_plate:
            pass

        return obs, reward, done, info

    def _build_random_obstacles(self):
        
        # generate random quaternion
        joint_state_old = self._get_joint_state()
        val = False
        maxiter = 2000
        while not val:
            joint_state_maybe = np.array(pyb.calculateInverseKinematics(
                bodyUniqueId=self.robot,
                endEffectorLinkIndex=self.end_effector_link_id,
                targetPosition=self.target_position,
                lowerLimits=self.joints_lower_limits.tolist(),
                upperLimits=self.joints_upper_limits.tolist(),
                jointRanges=self.joints_range.tolist(),
                restPoses=np.float32(self.resting_pose_angles).tolist(),
                maxNumIterations=maxiter,
                residualThreshold=5e-3))
            self._set_joint_state(joint_state_maybe)
            state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
            pos_maybe = np.array(state[4])
            #rot = np.array(ee_link_state[5])
            self.target_rot = self._quat_ee_to_w(np.array(state[1]))
            #print("---")
            #print(util.quaternion_similarity(rot_maybe, self.target_rot))
            #print(np.linalg.norm(pos_maybe - self.target))
            if np.linalg.norm(pos_maybe - self.target) < 5e-2:
                val = True
                self._set_joint_state(joint_state_old)
            maxiter += 500

        #self.target_rot = choice([np.array([0,0,0,1]), np.array([ 0, 0, 0.8509035, 0.525322 ]), np.array([ 0, 0, 0.8939967, -0.4480736 ]), np.array([ 0, 0, 0.0883687, -0.9960878 ])])
        

        inc = 0
        while np.random.random()>0.1 + inc:
            i = choice([0,1,2])
            position = 0.5*(np.array(self.init_home)+np.array(self.target_position))+0.05*np.random.uniform(low=-1, high=1, size=(3,))
            if i==0:
                obst_id = pyb.createMultiBody(
                                baseMass=0,
                                baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.05,0.05,0.001], rgbaColor=[0.5,0.5,0.5,1]),
                                baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.05,0.05,0.001]),
                                basePosition=position
                            )
                self.obj_ids.append(obst_id)
            if i==1:
                obst_id = pyb.createMultiBody(
                                baseMass=0,
                                baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.001,0.08,0.06], rgbaColor=[0.5,0.5,0.5,1]),
                                baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.001,0.08,0.06]),
                                basePosition=position
                            )
                self.obj_ids.append(obst_id) 
            if i==2:
                obst_id = pyb.createMultiBody(
                                baseMass=0,
                                baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.08,0.001,0.06], rgbaColor=[0.5,0.5,0.5,1]),
                                baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.08,0.001,0.06]),
                                basePosition=position
                            )
                self.obj_ids.append(obst_id)
            inc += 0.15

        position = 0.5*(np.array(self.init_home)+np.array(self.target_position))+0.05*np.random.uniform(low=-1, high=1, size=(3,))
        self.moving_plate = pyb.createMultiBody(
                                baseMass=0,
                                baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.05,0.05,0.002], rgbaColor=[0.5,0.5,0.5,1]),
                                baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.05,0.05,0.002]),
                                basePosition=position
                            )
        self.obj_ids.append(self.moving_plate)

    def _build_test_case_1(self):
        random_int = np.random.random_integers(0,4)
        if random_int == 0:
            self.init_home = [0.15,0.4,0.3]
            self.init_orn = self._quat_w_to_ee(util.rpy_to_quaternion([0,0,0]))
            self.target_position = [-0.15,0.4,0.3]
            halfExtents = [0.002,0.1,0.05]
            basePosition = [0.0,0.4,0.3]
        elif random_int == 1:
            self.init_home = [-0.15,0.4,0.3]
            self.init_orn = self._quat_w_to_ee(util.rpy_to_quaternion([0,0,0]))
            self.target_position = [0.15,0.4,0.3]
            halfExtents = [0.002,0.1,0.05]
            basePosition = [0.0,0.4,0.3]
        elif random_int == 2:
            self.init_home = [0.0,0.35,0.3]
            self.init_orn = self._quat_w_to_ee(util.rpy_to_quaternion([0,0,0]))
            self.target_position = [0.0,0.65,0.3]
            halfExtents = [0.1,0.002,0.05]
            basePosition = [0.0,0.5,0.3]
        elif random_int == 3:
            self.init_home = [0.0,0.65,0.3]
            self.init_orn = self._quat_w_to_ee(util.rpy_to_quaternion([0,0,0]))
            self.target_position = [0.0,0.35,0.3]
            halfExtents = [0.1,0.002,0.05]
            basePosition = [0.0,0.5,0.3]
        else:
            self.init_home = [0.0,0.4,0.38]
            self.init_orn = self._quat_w_to_ee(util.rpy_to_quaternion([0,0,0]))
            self.target_position = [0.0,0.4,0.22]
            halfExtents = [0.1,0.1,0.002]
            basePosition = [0.0,0.4,0.3]
        
        self.target = np.array(self.target_position)
        obst_1 = pyb.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.5,0.5,0.5,1]),
            baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=halfExtents),
            basePosition=basePosition,
            baseOrientation=choice([0.707, 0, 0, 0.707])
        )
        self.obj_ids.append(obst_1)
        pyb.resetBasePositionAndOrientation(self.robot, [0.0,-0.12, self.ceiling_mount_height], pyb.getQuaternionFromEuler([np.pi, 0, 0]))
        self.target_base = np.array([0.0,-0.12])
        self.target_rot = [0, 0, 0, 1]

    def _build_test_case_3(self):
        self.init_home = [0.25,0.4,0.3]
        self.init_orn = self._quat_w_to_ee(util.rpy_to_quaternion([0,0,0]))
        self.target_position = [0,0.4,0.25]
        self.target = np.array(self.target_position)
        obst_1 = pyb.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.002,0.1,0.05], rgbaColor=[0.5,0.5,0.5,1]),
            baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.002,0.1,0.05]),
            basePosition=[-0.1,0.4,0.3],
            baseOrientation=choice([0.707, 0, 0, 0.707])
        )
        self.obj_ids.append(obst_1)
        obst_2 = pyb.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.002,0.1,0.05], rgbaColor=[0.5,0.5,0.5,1]),
            baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.002,0.1,0.05]),
            basePosition=[0.1,0.4,0.3],
            baseOrientation=choice([0.707, 0, 0, 0.707])
        )
        self.obj_ids.append(obst_2)
        pyb.resetBasePositionAndOrientation(self.robot, [0.0,-0.12, self.ceiling_mount_height], pyb.getQuaternionFromEuler([np.pi, 0, 0]))
        self.target_rot = [0, 0, 0, 1]