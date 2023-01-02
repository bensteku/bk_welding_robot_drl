from random import choice
import gym
import pybullet as pyb
import numpy as np
from util import xml_parser, util
import os
import pickle
from time import sleep, time

class PathingEnvironmentPybullet(gym.Env):

    def __init__(self, env_config):
        """
        Creates an env instance using the settings in the env_config.
        All of the work is offloaded to class methods to keep this a bit cleaner.
        """

        # task specific settings that need to be set at the start
        self._init_task_settings_pre(env_config["task"])

        # read the env config object and set general, non-task specific variables
        self._init_env_config(env_config)

        # start pybullet and initialize related variables
        self._init_pybullet_settings()

        # miscellaneous settings like training scheduling
        self._init_misc_settings()

        # set up the gym observation and action space as needed
        self._init_action_and_observation_space()

        # task specific settings that need to be set after everything else
        self._init_task_settings_post(env_config["task"])

        # determines wether the reset performs a full or just a soft reset
        self.cold_start = True  # env has to be built from scratch as this is the programm start

        # set the epoch for execution such that time stamps are relative rather than absolute
        self.epoch = time()

    def _init_task_settings_pre(self, task_settings):
        """
        Method that sets subclass specific values prior to any other default, non-implementation specific values being set.
        This could be things like position and orientation of the robot base, resting angles, max velocities etc.
        To be implenmented by subclass.
        """
        # set workspace bounds
        self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max = task_settings["workspace_bounds"]
 
    def _init_task_settings_post(self, task_settings):
        """
        Method that sets subclass specific values after all other default, non-implementation specific values have been set.
        This could be things like position and orientation of the robot base, resting angles, max velocities etc.
        To be implenmented by subclass.
        """
        raise NotImplementedError

    def _init_env_config(self, env_config):
        """
        Sets all the values contained in the env_config, aside from the custom subclass-dependent settings.
        """

        # int: id, default is 0, but can be another number, used for distinguishing envs running in parallel
        self.id = env_config["id"]

        # bool: eval or train mode
        self.train = env_config["train"]

        # string: path for asset files
        self.asset_files_path = env_config["asset_files_path"]

        # dict: task as key and task settings as value
        self.task = env_config["task"]

        # bool: actions as xyz and rpy movements or joint movements
        self.joint_actions = env_config["joint_actions"]  # TODO: rename to "joint_actions"

        # dict: contains all the sensors used for the model as keys(strings) and their configuration as values(dicts)
        self.sensors = env_config["sensor_config"]
        for sensor in self.sensors:
            if sensor not in ["lidar"]:#, "ee_rgb", "ee_depth", "fixed_rgb", "fixed_depth", "moving_rgb", "moving_depth"]:  # TODO
                raise ValueError("Sensor "+sensor+" not recognized/implemented!")

        # bool: normalize observations and rewards
        self.normalize = env_config["normalize"]
        
        # dict: contains all goals for the model as keys, values contain dicts that further set how the reward shall be processed
        self.goals = env_config["goals"]
        min_reward = np.inf
        max_reward = -np.inf
        for goal in self.goals:
            min_reward = min(min_reward, goal["reward"])
            max_reward = max(max_reward, goal["reward"])
            if goal not in ["collision", "clearance", "rotation"]:
                raise ValueError("Goal "+goal+" not recognized/implemented!")
        if "clearance" in self.goals and not "lidar" in self.sensors:
            raise ValueError("Clearance as a goal is only possible with lidar sensor input!")
        if self.normalize and abs(min_reward) != max_reward:
            raise ValueError("Normalizing is activated and min/max reward are not symmetrical! This would lead to improper mapping between -1 and 1!")
        # set up the normalizing constant for normalizing rewards
        self.reward_normalizing_a = 2 / (max_reward - min_reward)
        self.reward_normalizing_b = 1 - self.reward_normalizing_a * max_reward

        # float: gamma, just used for record keeping such that the discounted episode total reward is roughly the same as the one calculated by the automatic StableBaselines logging
        self.gamma = env_config["gamma"]

        # bool: have pybullet display a GUI for the simulation or not
        self.display = env_config["display"]

        # logging level, 0: no logging, 1: logging, 2: logging and writing to textfile
        self.logging = env_config["logging"]
        if self.logging > 0:
            self.log = []

    def _init_pybullet_settings(self):
        """
        Initializes several variables related to handling PyBullet.
        """
        # list of pybullet object ids currently in the env aside from the robot
        self.obstacles = []

        # pybullet robot id
        self.robot_id = None  # TODO: rename to robot_id

        # ids of the joints of the robot
        self.joint_ids = None
        # link ids of the base and end effector of the robot
        self.robot_ee_id = None
        self.robot_base_id = None

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
        self.lidar_distances = None 
        self.joints = None
        self.joints_vel = None
        self.max_dist = None
        self.min_dist = None
        self.distance = None
        self.ee_rgb_img = None
        self.ee_depth_img = None
        self.fixed_rgb_img = None
        self.fixed_depth_img = None
        self.moving_rgb_img = None
        self.moving_depth_img = None
        self.time = None

        # pybullet connection and setup
        disp = pyb.DIRECT  # direct <-> no gui, use for training
        if self.display:
            disp = pyb.GUI
        pyb.connect(disp)
        pyb.setAdditionalSearchPath(self.asset_files_path)

    def _init_misc_settings(self):
        """
        Initializes several settings related to stuff not covered by the other methods.
        This includes things like training related variables.
        """
        # statistics and housekeeping
        self.episodes = 0
        self.discounted_reward_current_episode = 0
        self.distance_current_episode = 0
        self.steps_current_episode = 0
        self.steps_all_episodes = 0

        self.collisions = 0
        self.clearance_failures = 0
        self.is_success = False
        
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

    def _init_action_and_observation_space(self):
        """
        Initializes the action and observation space following the input from the config.
        Also sets normalizing constants that get used to make normalizing the observations faster.
        """
        # construct observation space with fitting amount of elements and sizes based on the sensors and goals used
        
        # ee position is always used
        obs_space_dict = dict()
        
        ee_to_base_vector_dims = 3
        ee_to_target_vector_dims = 3
        ee_to_target_distance_dims = 1
        position_dims = ee_to_base_vector_dims + ee_to_target_vector_dims + ee_to_target_distance_dims
        obs_space_dict["position"] = gym.spaces.Box(low=-1 if self.normalize else np.array([self.x_min, self.y_min, self.z_min]), high=1 if self.normalize else np.array([self.x_max, self.y_max, self.z_max]), shape=(position_dims,), dtype=np.float32)
        
        # position normalizing
        max_distance = np.ceil(np.sqrt((self.x_max - self.x_min)**2 + (self.y_max - self.y_min)**2 + (self.z_max - self.z_min)**2))  # this is the maximum possible distance given the workspace
        ee_to_base_vec_max = np.array([self.robot_range, self.robot_range, self.robot_range])
        ee_to_base_vec_min = -ee_to_base_vec_max
        ee_to_target_vec_max = np.array([self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min])  # this is the largest possible vector difference between two given the allowed workspace
        ee_to_target_vec_min = np.array([self.x_min - self.x_max, self.y_min - self.y_max, self.z_min - self.z_max])
        
        self.position_normalizing_a = np.zeros(position_dims)
        self.position_normalizing_a[:3] = 2 / (ee_to_base_vec_max - ee_to_base_vec_min)
        self.position_normalizing_a[3:6] = 2 / (ee_to_target_vec_max - ee_to_target_vec_min)
        self.position_normalizing_a[6] = 1 / max_distance

        self.position_normalizing_b = np.zeros(position_dims)
        self.position_normalizing_b[:3] = np.ones(3) - np.multiply(self.position_normalizing_a[:3], ee_to_base_vec_max)
        self.position_normalizing_b[3:6] = np.ones(3) - np.multiply(self.position_normalizing_a[3:6], ee_to_target_vec_max)
        self.position_normalizing_b[6] = 1 - self.position_normalizing_a[6] * max_distance


        # ee rotation is also always used, but if the rotation goal is activated more information is added
        ee_rotation_dims = 3  # 3 for rpy
        target_rotation_dims = 3 if "rotation" in self.goals else 0  # 3 for rpy
        ee_to_target_rotation_distance_dims = 1 if "rotation" in self.goals else 0  # distance measure between the two rotations
        rotation_dims = ee_rotation_dims + target_rotation_dims + ee_to_target_rotation_distance_dims
        obs_space_dict["rotation"] = gym.spaces.Box(low=-1 if self.normalize else -np.pi, high=1 if self.normalize else np.pi, shape=(rotation_dims,), dtype=np.float32) 

        # rotation normalizing
        max_angles = np.array([np.pi, np.pi, np.pi])
        min_angles = -max_angles

        self.rotation_normalizing_a = np.zeros(rotation_dims)
        self.rotation_normalizing_a[:3] = 2 / (max_angles - min_angles)
        
        self.rotation_normalizing_b = np.zeros(rotation_dims)
        self.rotation_normalizing_b[:3] = np.ones(3) - np.multiply(self.rotation_normalizing_a[:3], max_angles)
        
        if "rotation" in self.goals:
            self.rotation_normalizing_a[3:6] = 2 / (max_angles - min_angles)
            self.rotation_normalizing_a[6] = 0  # the rotation distance is always normalized

            self.rotation_normalizing_b[3:6] = np.ones(3) - np.multiply(self.rotation_normalizing_a[3:6], max_angles)
            self.rotation_normalizing_b[6] = 0  # same as above

        # add current joint positions to observation space if needed
        if self.joint_actions:
            joints_dims = 6
            obs_space_dict["joints"] = gym.spaces.Box(low=-1 if self.normalize else self.joints_lower_limits, high=1 if self.normalize else self.joints_upper_limits, shape=(joints_dims,), dtype=np.float32)
            
            self.joints_normalizing_a = 2 / self.joints_range
            
            self.joints_normalizing_b = np.ones(joints_dims) - np.multiply(self.joints_normalizing_a, self.joints_upper_limits)

        # add optional sensors to the observation space
        if "lidar" in self.sensors:
            lidar_dims = 17  # TODO: make number custom dependent on config
            obs_space_dict["lidar"]  = gym.spaces.Box(low=0 if not self.sensors["lidar"]["indicator"] else -1, high=1, shape=(lidar_dims,), dtype=np.float32) 
            # no normalizing for lidar as the sensor data should be normalized anyway already
        
        # TODO: normalizing for all the image sensor data, TBD
        if "ee_rgb" in self.sensors:
            ee_rgb_img_x = self.sensors["ee_rgb"]["width"]
            ee_rgb_img_y = self.sensors["ee_rgb"]["height"]
            obs_space_dict["ee_rgb"] = gym.spaces.Box(low=0, high=1 if self.normalize else 255, shape=(ee_rgb_img_y, ee_rgb_img_x), dtype=np.float32)  # TODO: check if low and high are correct
        
        if "ee_depth" in self.sensors:
            ee_depth_img_x = self.sensors["ee_depth"]["width"]
            ee_depth_img_y = self.sensors["ee_depth"]["height"]
            obs_space_dict["ee_depth"] = gym.spaces.Box(low=0, high=1 if self.normalize else 255, shape=(ee_depth_img_y, ee_depth_img_x), dtype=np.float32)

        if "fixed_rgb" in self.sensors:
            fixed_rgb_img_x = self.sensors["fixed_rgb"]["width"]
            fixed_rgb_img_y = self.sensors["fixed_rgb"]["height"]
            obs_space_dict["fixed_rgb"] = gym.spaces.Box(low=0, high=1 if self.normalize else 255, shape=(fixed_rgb_img_y, fixed_rgb_img_x), dtype=np.float32)

        if "fixed_depth" in self.sensors:
            fixed_depth_img_x = self.sensors["fixed_depth"]["width"]
            fixed_depth_img_y = self.sensors["fixed_depth"]["height"]
            obs_space_dict["fixed_depth"] = gym.spaces.Box(low=0, high=1 if self.normalize else 255, shape=(fixed_depth_img_y, fixed_depth_img_x), dtype=np.float32)

        if "moving_rgb" in self.sensors:
            moving_rgb_img_x = self.sensors["moving_rgb"]["width"]
            moving_rgb_img_y = self.sensors["moving_rgb"]["height"]
            obs_space_dict["moving_rgb"] = gym.spaces.Box(low=0, high=1 if self.normalize else 255, shape=(moving_rgb_img_y, moving_rgb_img_x), dtype=np.float32)

        if "moving_depth" in self.sensors:
            moving_depth_img_x = self.sensors["moving_depth"]["width"]
            moving_depth_img_y = self.sensors["moving_depth"]["height"]
            obs_space_dict["moving_depth"] = gym.spaces.Box(low=0, high=1 if self.normalize else 255, shape=(moving_depth_img_y, moving_depth_img_x), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(obs_space_dict)

        # build action space
        xyzrpy_or_joints_dims = 6  # regardless of using the action as joints or xyz rpy inputs, we need 6 elements
        camera_movement_dims = 2 if "moving_rgb" in self.sensors else 0 + 2 if "moving_depth" in self.sensors else 0
        action_dims = xyzrpy_or_joints_dims + camera_movement_dims
        
        # the action space is always normalized, its elements get interpreted in the code further down
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dims,), dtype=np.float32)

    def reset(self):
        """
        Generalized reset method.
        """

        self.time = time() - self.epoch
        
        # determine wether a complete reset is necessary or not
        complete_reset = self.cold_start or self.episodes % self.task.complete_reset_rate == 0
        self.cold_start = False

        # set relevant counters
        self.steps_current_episode = 0
        self.episodes += 1
        self.discounted_reward_current_episode = 0
        self.distance_current_episode = 0

        # reset the environment
        if complete_reset:
            # rebuild the entire environment from scratch
            self.robot_id, self.obstacles, self.target = self.task.reset_hard()
        else:
            # just reset robot position and get new target
            self.target = self.task.reset_soft()

        # get state information
        ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        self.pos = np.array(ee_link_state[4])
        self.pos_last = self.pos
        self.pos_vel = np.zeros(3)
        self.joints_vel = np.zeros(6)
        self.rot = np.array(ee_link_state[5])
        if self.task.use_internal_rot:
            self.rot_internal = self._quat_ee_to_w(np.array(ee_link_state[1]))
        if "lidar" in self.sensor:
            self.lidar_indicator, self.lidar_distances = self._get_lidar_data()
        if "ee_rgb" in self.sensors or "ee_depth" in self.sensors:
            self.ee_rgb_img, self.ee_depth_img = self._get_ee_image()  # both images get taken anyway, so no possible performance gain by separating them
        if "fixed_rgb" in self.sensors or "fixed_depth" in self.sensors:
            self.fixed_rgb_img, self.fixed_depth_img = self._get_fixed_image()
        if "moving_rgb" in self.sensors or "moving_depth" in self.sensors:
            self.moving_rgb_img, self.moving_depth_img = self._get_moving_image()
        self.distance = np.linalg.norm(self.pos - self.target)

        # TODO: add camera setup for the various image sensors in one of the setup methods above

        # adjust the success thresholds
        # TODO: rewrite the success threshold such that only those tresholds appear/get updated that are actually needed
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

        # reset success
        self.is_success = False

        # logging
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

        return self._get_obs()

    def _get_obs(self):

