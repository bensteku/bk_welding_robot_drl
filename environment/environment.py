import gym
import pybullet as pyb
import numpy as np
from util import xml_parser, util
import os
import pickle
from time import sleep

class PathingEnvironmentPybullet(gym.Env):

    def __init__(self,
                env_config):

        # id, default is 0, but can be another number, used for distinguishing envs running in parallel
        self.id = env_config["id"]

        # eval or train mode
        self.train = env_config["train"]

        # path for asset files
        self.asset_files_path = env_config["asset_files_path"]

        # bool flag for showing the target in the pybullet simulation
        self.show_target = env_config["show_target"]

        # bool falg for whether actions are xyz and rpy movements or joint movements
        self.use_joints = env_config["use_joints"]

        # bool flag whether to also reward current rotation
        self.use_set_poses = env_config["use_set_poses"]

        # bool flag wether to process lidar results or not
        self.use_raw_lidar = env_config["use_raw_lidar"]

        # bool flag whether observations are normalized or not
        self.normalize = env_config["normalize"]

        # attribute for target box, see its method
        self.ignore_obstacles_for_target_box = env_config["ignore_obstacles_for_target_box"]

        # attribute for pybullet display
        self.display = env_config["display"]
        
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
        # they were derived by combining the rotation of the end effector link with the rotation of the torch mesh
        # both can be found in their respective urdf files
        self.ground_truth_conversion_angles = [np.array([-0.2726201, 0.2726201, -0.6524402, -0.6524402]), 
                                               np.array([-0.0676347, 0.0676347, -0.7038647, -0.7038647])]

        # if use_joints is set to false:
        # maximum translational movement per step
        self.pos_speed = 0.001
        # maximum rotational movement per step
        self.rot_speed = 0.02
        # if use_joints is set to true:
        # maximum joint movement per step
        self.joint_speed = 0.001

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
            self.ee_spawn_thresh = 1e-1
            self.ee_spawn_thresh_max = 1.5
            self.ee_spawn_thresh_min = 1e-3
            self.ee_pos_reward_thresh_min = 1e-2
            self.ee_pos_reward_thresh_max = 6e-1
            self.ee_pos_reward_thresh_max_increment = 1e-2
            self.ee_pos_reward_thresh_min_increment = 1e-3
            self.stats_buffer_size = 50
            self.success_buffer = []
            self.collision_buffer = []
            self.timeout_buffer = []
            self.out_of_bounds_buffer = []
        else:
            self.ee_pos_reward_thresh = 1e-2
            self.ee_pos_reward_thresh_min = 1e-2

        # variables storing information about the current state, saves performance from having to 
        # access the pybullet interface all the time
        self.pos = None
        self.rot = None
        self.rot_internal = None
        self.pos_last = None
        self.target = None
        self.target_norms = None
        self.target_rot = None
        self.lidar_indicator = None
        self.lidar_indicator_raw = None
        self.lidar_probe = None
        self.joints = None
        self.max_dist = None
        self.min_dist = None
        self.distance = None

        # process the dataset of meshes and urdfs
        self.dataset = self._register_data()

        # steps taken in current epsiode and total
        self.steps_current_episode = 0
        self.steps_total = 0
        # maximum steps per episode
        if self.train:
            self.max_episode_steps = 1000 if self.use_joints else 1500
        else:
            self.max_episode_steps = 250 if self.use_joints else 500  # reduce the overhead for model evaluation
        # episode counter
        self.episodes = 0
        self.episode_reward = 0
        self.episode_distance = 0
        # success counter, to calculate success rate
        self.successes = 0
        # collision counter, to calculate collision rate
        self.collisions = 0
        # bool flag if current episode is a success, gets used in the next episode for adjusting the target zone difficulty
        self.is_success = False
        # counter for reseting the mesh in the reset method
        # this means that for x episodes, the mesh in the env will stay the same and only the targets will change
        # this saves performance with no drawback
        self.reset_mesh_after_episodes = 100

        # gym action space
        # first three entries: ee position, last three entries: ee rotation given as rpy in pybullet world frame
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # gym observation space
        # spatial: 1-6: current joint angles, 7-9: vector difference between target and current position, 10-12: current rotation as quaternion, 13: distance between target and position, all optionally normalized to be between -1 and 1 for deep learning
        # joint angles are left out if training the action space is used for xyz + rpy
        # lidar: 17 floats that signify distances obstacles lie at around the welding torch in different directions
        self.observation_space = gym.spaces.Dict(
            {
            'spatial': gym.spaces.Box(low=-1 if self.normalize else -8, high=1 if self.normalize else 8, shape=(13 if self.use_joints else 7,), dtype=np.float32),
            'lidar': gym.spaces.Box(low=0 if self.use_raw_lidar else -1, high=1, shape=(17,), dtype=np.float32)  
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
        self.normalizing_constant_a[9:12] = 2 / (self.rpy_upper_limits - self.rpy_lower_limits)
        self.normalizing_constant_a[12] = 1 / max_distance 
        self.normalizing_constant_b = np.zeros(14)
        self.normalizing_constant_b[:6] = np.ones(6) - np.multiply(self.normalizing_constant_a[:6], self.joints_upper_limits)
        self.normalizing_constant_b[6:9] = np.ones(3) - np.multiply(self.normalizing_constant_a[6:9], vec_distance_max)
        self.normalizing_constant_b[9:12] = np.ones(3) - np.multiply(self.normalizing_constant_a[9:12], self.rpy_upper_limits)
        self.normalizing_constant_b[12] = 1 - self.normalizing_constant_a[12] * max_distance

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

        if self.train:
            if len(self.success_buffer) != 0:
                success_rate = np.average(self.success_buffer) 
            else:
                success_rate = 0
            if success_rate < 0.8 and self.ee_pos_reward_thresh < self.ee_pos_reward_thresh_max and not self.is_success:
                self.ee_pos_reward_thresh += util.linear_interpolation(self.ee_pos_reward_thresh, self.ee_pos_reward_thresh_min, self.ee_pos_reward_thresh_max, self.ee_pos_reward_thresh_min_increment, self.ee_pos_reward_thresh_max_increment) / 15
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

    def _get_obs(self):
        spatial = np.zeros(13)
        if self.normalize:   
            spatial[:6] = np.multiply(self.normalizing_constant_a[:6], self.joints) + self.normalizing_constant_b[:6]
            spatial[6:9] = np.multiply(self.normalizing_constant_a[6:9], (self.target - self.pos)) + self.normalizing_constant_b[6:9]
            spatial[9:12] = np.multiply(self.normalizing_constant_a[9:12], util.quaternion_to_rpy(self.rot_internal)) +  self.normalizing_constant_b[9:12]
            spatial[12] = self.normalizing_constant_a[12] * self.distance + self.normalizing_constant_b[12]
        else:
            spatial[:6] = self.joints
            spatial[6:9] = self.target - self.pos
            spatial[9:12] = util.quaternion_to_rpy(self.rot_internal)
            spatial[12] = self.distance

        return {
            'spatial': spatial if self.use_joints else spatial[6:],
            'lidar': self.lidar_indicator if not self.use_raw_lidar else self.lidar_indicator_raw
        }

    def step(self, action):

        # save old position
        #ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        #self.pos = np.array(ee_link_state[4])
        self.pos_last = self.pos
        #self.rot = np.array(ee_link_state[5])

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
            self.joints = self._movej(joints_desired)

        # get new state info
        ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        self.pos = np.array(ee_link_state[4])
        self.rot = np.array(ee_link_state[5])
        self.rot_internal = self._quat_ee_to_w(np.array(ee_link_state[1]))
        self.lidar_indicator, self.lidar_indicator_raw = self._get_lidar_indicator()

        # increment steps
        self.steps_current_episode += 1
        self.steps_total += 1

        # get reward and info
        reward, done, info = self._reward()

        return self._get_obs(), reward, done, info


    def _reward2(self):

        done = False
        is_success = False

        collided = self._collision()
        

        distance_cur = np.linalg.norm(self.target - self.pos)
        distance_last = np.linalg.norm(self.target - self.pos_last) 

        self.distance = distance_cur

        # update the max_dist variable
        if distance_cur > self.max_dist:
            self.max_dist = distance_cur

        collision_reward = -5 if collided else 0

        distance_delta = self.min_dist - distance_cur
        if self._is_in_target_box():
            distance_reward = 5
            done = True
            is_success = True
            self.successes += 1
        elif self._is_out_of_bounds(distance_cur):
            done = True
            is_success = False
            distance_reward = -5
        elif distance_delta < 0:
            distance_reward = 0.005 * distance_delta
            done = False
            is_success = False
        else:
            distance_reward = 0.01 * distance_delta
            done = False
            is_success = False

        if collided:
            done = True
            is_success = False

        time_reward = -0.005

        obstacle_reward = 0
        for ray in self.lidar_probe:
            obstacle_reward += -0.05 * 1/(3.98*ray)

        reward = collision_reward + distance_reward + time_reward + obstacle_reward
    
        # update the min_dist variable
        if distance_cur < self.min_dist:
            self.min_dist = distance_cur    

        if self.steps_current_episode > self.max_episode_steps:
            done = True

        if self.train:
            if done:
                self.success_buffer.append(is_success)
                if len(self.success_buffer) > self.stats_buffer_size:
                    self.success_buffer.pop(0)
        
        self.episode_reward += reward
        self.episode_distance += distance_cur
        self.is_success = is_success

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
                info_string += key + ": " + str(round(info[key], 3)) + ", "
            print(info_string)

        return reward, done, info

    def _reward(self):
        
        # get closestpoints
        # quadratischef unktion: c 15000 n 35
        # steps vor collision angucken

        collided = self._collision()
        out_of_bounds = False

        distance_cur = np.linalg.norm(self.target - self.pos)
        distance_last = np.linalg.norm(self.target - self.pos_last)

        # update the max_dist variable
        if distance_cur > self.max_dist:
            self.max_dist = distance_cur

        # only score on position
        if not self.use_set_poses:
            # check if out of bounds
            if self._is_out_of_bounds(distance_cur):
                done = True
                is_success = False
                out_of_bounds = True
                reward = -0.75
                reward = -5
            elif not collided:
                #if distance_cur < self.ee_pos_reward_thresh:
                if self._is_in_target_box():
                    reward = 1
                    reward = 10
                    done = True
                    is_success = True
                    self.successes += 1
                else:
                    c1 = -1
                    # the following two calculations adjust the two parameters for the obstacle penalty such that -0.05 reward is given when the closest obstacle is at a distance of 2/3 of the smallest reward threshold and
                    # -0.5 reward is given when it's 1/3 of the minimum threshold away
                    #n = np.log(-0.5/-0.05) / np.log((1+(2/3)*self.ee_pos_reward_thresh_min)/(1+(1/3)*self.ee_pos_reward_thresh_min))
                    #c2 = -0.05 / ((1/(1+(2/3)*self.ee_pos_reward_thresh_min))**n)
                    #n = 10
                    #fade_out = 3
                    #reward = -0.001 * (np.sqrt(distance_cur)**0.8)  # the square root with the exponent should give a good gradient: steep near the goal, flat but still negative far away
                    #reward_1 = 0.5 * distance_cur * distance_cur
                    c1 = -1500
                    n = 35
                    c2 = -1
                    #reward_1 = c1 * 2 * np.sqrt(distance_cur)
                    reward_1 = c1 * distance_cur**1.7
                    done = False
                    is_success = False
                    if distance_cur > distance_last:
                        # add a very small penalty if the distance increased in comparison to the last step
                        #reward -= 0.0001 * distance_last
                        pass                    
                    # add a reward for keeping a small buffer away from obstacles
                    """
                    obstacle_reward = 0
                    for ray in self.lidar_probe[1:]:
                        obstacle_reward += -0.05 * (1/(7.98*(ray+1e-8)))
                    #print("reward",reward)
                    obstacle_reward /= 5e6
                    #print("obstacle",obstacle_reward)
                    """
                    minimum_contact = 4
                    for id in self.obj_ids:
                        if id != self.robot:
                            minimum_contact = min(minimum_contact, min(np.array(pyb.getClosestPoints(self.robot, id, 4, self.end_effector_link_id), dtype=object)[:,8]))  # torch tip
                            minimum_contact = min(minimum_contact, min(np.array(pyb.getClosestPoints(self.robot, id, 4, self.end_effector_link_id-1), dtype=object)[:,8]))  # torch attachment point           
                    reward_2 = c2 * (1/(1+minimum_contact))**n
                    #proximity = 0 if distance_cur > fade_out*self.ee_pos_reward_thresh else (np.sqrt(distance_cur)/(np.sqrt(fade_out*self.ee_pos_reward_thresh)))
                    #proximity = 1 if False else (distance_cur**15/(np.sqrt(fade_out*self.ee_pos_reward_thresh)))
                    reward = reward_1 + reward_2 #* (1 - (1/(1+distance_cur)))
                    reward = reward * 1e-5
                    reward = -0.01 * distance_cur
                    """ 
                    print("------------")
                    print(minimum_contact, distance_cur)
                    print(reward_1, reward_2)
                    print(reward)
                    sleep(0.2)
                    """
            else:
                reward = -1
                reward = -10
                done = True
                is_success = False
                self.collisions += 1
        # score also on rotation
        else:
            if self._is_out_of_bounds():
                done = True
                is_success = False
                reward = -1
            elif not collided:
                rotation_done = util.quaternion_apx_eq(self.rot_internal, self.target_rot)
                pos_done = self._is_in_target_box()
                if rotation_done and pos_done:
                    reward = 1
                    done = True
                    is_success = True
                elif pos_done and not rotation_done:
                    reward = 0.5
                    done = False
                    is_success = False
                    reward += -0.01 * (1 - util.quaternion_similarity(self.rot_internal, self.target_rot))
                else:
                    pos_reward = -0.001 * (np.sqrt(distance_cur)**0.8)
                    quat_reward = -0.01 * (1 - util.quaternion_similarity(self.rot_internal, self.target_rot))  # TODO
                    reward = pos_reward + (1 - (distance_cur / self.max_dist)) * quat_reward  # the closer to the target the EE is, the more the rotation should be scored on
                    done = False
                    is_success = False
            else:
                reward = -1
                done = True
                is_success = False

        timeout = False
        if self.steps_current_episode > self.max_episode_steps:
            done = True
            timeout = True

        if self.train:
            if done:
                self.success_buffer.append(is_success)
                if len(self.success_buffer) > self.stats_buffer_size:
                    self.success_buffer.pop(0)
                self.collision_buffer.append(collided)
                if len(self.collision_buffer) > self.stats_buffer_size:
                    self.collision_buffer.pop(0)
                self.timeout_buffer.append(timeout)
                if len(self.timeout_buffer) > self.stats_buffer_size:
                    self.timeout_buffer.pop(0)
                self.out_of_bounds_buffer.append(out_of_bounds)
                if len(self.out_of_bounds_buffer) > self.stats_buffer_size:
                    self.out_of_bounds_buffer.pop(0)
        
        self.episode_reward += reward
        self.episode_distance += distance_cur
        self.is_success = is_success

        info = {
            'step': self.steps_current_episode,
            'steps_total': self.steps_total,
            'episodes': self.episodes,
            'done': done,
            'is_success': is_success,
            'success_rate': self.successes/self.episodes if not self.train else np.average(self.success_buffer),
            'collided': collided,
            'collision_rate': self.collisions/self.episodes if not self.train else np.average(self.collision_buffer),
            'timeout': timeout,
            'timeout_rate': 0 if not self.train else np.average(self.timeout_buffer),
            'out_of_bounds': out_of_bounds,
            'out_of_bounds_rate': 0 if not self.train else np.average(self.out_of_bounds_buffer),
            'reward': reward,
            'episode_reward': self.episode_reward,
            'distance': distance_cur,
            'episode_distance': self.episode_distance,
            'distance_threshold': self.ee_pos_reward_thresh
        }

        if done:
            info_string = ""
            for key in info:
                info_string += key + ": " + str(round(info[key], 3)) + ", "
            print(info_string)

        #print(reward-self.last_reward)
        #self.last_reward = reward
        #sleep(0.05)

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

    def _cast_lidar_rays(self, ray_min=0.02, ray_max=0.3, ray_num_side=6, ray_num_forward=12, render=True):
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

    def _get_lidar_indicator(self, buckets=20):
        ray_num_side=6
        ray_num_forward=12
        lidar_results = np.array(self._cast_lidar_rays(ray_num_side=ray_num_side, ray_num_forward=ray_num_forward), dtype=object)[:,2]  # only use the distance information
        indicator_raw = np.zeros((17,), dtype=np.float32)
        indicator = np.zeros((17,), dtype=np.float32)

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
        lidar_min = lidar_results[0:(2 * ray_num_forward + 1)].min()  # 1 ray going straight forward + 2 cones of ray_num_forward rays around it
        #indicator[0] = 1 if lidar_min >= 0.99 else (0.5 if 0.75 <= lidar_min < 0.99 else (0 if 0.5 <= lidar_min < 0.75 else (-0.5 if 0.25 <= lidar_min < 0.5 else -1)))
        indicator[0] = raw_to_indicator(lidar_min)
        indicator_raw[0] = lidar_min
        # tip sides indicators
        for i in range(8):
            lidar_min = lidar_results[(2 * ray_num_forward + 1 + i * ray_num_side):(2 * ray_num_forward + 1 + (i + 1) * ray_num_side)].min()
            #indicator[1+i] = 1 if lidar_min >= 0.99 else (0.5 if 0.75 <= lidar_min < 0.99 else (0 if 0.5 <= lidar_min < 0.75 else (-0.5 if 0.25 <= lidar_min < 0.5 else -1)))
            indicator[1+i] = raw_to_indicator(lidar_min)
            indicator_raw[1+i] = lidar_min
        # grip sides indicators
        for i in range(8):
            lidar_min = lidar_results[(2 * ray_num_forward + 1 + 8 * ray_num_side + i * ray_num_side):(2 * ray_num_forward + 1 + 8 * ray_num_side + (i + 1) * ray_num_side)].min()
            #indicator[1+8+i] = 1 if lidar_min >= 0.99 else (0.5 if 0.75 <= lidar_min < 0.99 else (0 if 0.5 <= lidar_min < 0.75 else (-0.5 if 0.25 <= lidar_min < 0.5 else -1)))
            indicator[1+8+i] = raw_to_indicator(lidar_min)
            indicator_raw[1+8+i] = lidar_min
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
        fwdzId = pyb.addUserDebugParameter("fwd_z", 0, 0.6, self.pos[2])
        fwdxIdbase = pyb.addUserDebugParameter("fwd_x_base", -4, 8, 0)
        fwdyIdbase = pyb.addUserDebugParameter("fwd_y_base", -4, 8, 0)
        x_base = 0
        y_base = 0
        oldybase = 0
        oldxbase = 0

        pyb.addUserDebugLine([0,0,0],[0,0,1],[0,0,1],parentObjectUniqueId=self.robot, parentLinkIndex= self.end_effector_link_id)
        pyb.addUserDebugLine([0,0,0],[0,1,0],[0,1,0],parentObjectUniqueId=self.robot, parentLinkIndex= self.end_effector_link_id)
        pyb.addUserDebugLine([0,0,0],[1,0,0],[1,0,0],parentObjectUniqueId=self.robot, parentLinkIndex= self.end_effector_link_id)

        while True:
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
        # test: get a random offset of at most length 0.5 to move the base away such that the task becomes more varied
        random_offset = np.random.uniform(low=-1, high=1, size=2)
        random_offset = random_offset / np.linalg.norm(random_offset)
        random_offset = random_offset * np.random.uniform(low=0, high=0.25)
        self.target_base = np.average(positions, axis=0)[:2] + self.base_offset + random_offset

        # set also a target rotation if needed:
        if self.use_set_poses:
            rotations = [util.matrix_to_quaternion(ele[:3,:3]) for ele in frame["pose_frames"]]
            self.target_rot = rotations[0]

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
            'episodes': self.episodes,
            'successes': self.successes,
            'collisions': self.collisions,
            'success_buffer': self.success_buffer,
            'collision_buffer': self.collision_buffer,
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
            self.episodes = load_dict["episodes"]
            self.successes = load_dict["successes"]
            self.success_buffer = load_dict["success_buffer"]
            self.collision_buffer = load_dict["collision_buffer"]
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
                self.ee_pos_reward_thresh += util.linear_interpolation(self.ee_pos_reward_thresh, self.ee_pos_reward_thresh_min, self.ee_pos_reward_thresh_max, self.ee_pos_reward_thresh_min_increment, self.ee_pos_reward_thresh_max_increment) / 15
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
            spatial[:6] = np.multiply(self.normalizing_constant_a[:6], self.joints) + self.normalizing_constant_b[:6]
            spatial[6:9] = np.multiply(self.normalizing_constant_a[6:9], (self.target - self.pos)) + self.normalizing_constant_b[6:9]
            spatial[9:12] = np.multiply(self.normalizing_constant_a[9:12], util.quaternion_to_rpy(self.rot_internal)) +  self.normalizing_constant_b[9:12]
            spatial[12] = self.normalizing_constant_a[12] * self.distance + self.normalizing_constant_b[12]
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



