import gym
import pybullet as pyb
import numpy as np
from util import xml_parser, util
import os

class PathingEnvironmentPybullet(gym.Env):

    def __init__(self,
                asset_files_path,
                train,
                display=False,
                show_target=False):

        # eval or train mode
        self.train = train

        # path for asset files
        self.asset_files_path = asset_files_path

        # bool flag for showing the target in the pybullet simulation
        self.show_target = show_target
        
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

        # angle conversion constants
        # these are used for converting the pybullet coordinate system of the end effector into the coordinate system used
        # by the MOSES ground truth, entry one is for MRW510, entry two is for TAND GERAD
        # they were derived by combining the rotation of the end effector link with the rotation of the torch mesh
        # both can be found in their respective urdf files
        self.ground_truth_conversion_angles = [np.array([-0.2726201, 0.2726201, -0.6524402, -0.6524402]), 
                                               np.array([-0.0676347, 0.0676347, -0.7038647, -0.7038647])]

        # maximum translational movement
        self.pos_speed = 0.02
        # maximum rotational movement
        self.rot_speed = 0.01 

        # offset at wich welding meshes will be placed into the world
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
            self.ee_pos_reward_thresh = 3.1e-1
            self.ee_pos_reward_thresh_min = 4e-3
            self.ee_pos_reward_thresh_max = 6e-1
            self.ee_pos_reward_thresh_increments = 2e-2
            self.ee_pos_reward_threshold_change_after_episodes = 50
            self.success_buffer = []
        else:
            self.ee_pos_reward_thresh = 5e-3

        # variables storing information about the current state, saves performance from having to 
        # access the pybullet interface all the time
        self.pos = None
        self.rot = None
        self.pos_last = None
        self.target = None
        self.lidar_probe = None
        self.joints = None

        # process the dataset of meshes and urdfs
        self.dataset = self._register_data()

        # steps taken in current epsiode and total
        self.steps_current_episode = 0
        self.steps_total = 0
        # maximum steps per episode
        if self.train:
            self.maximum_steps_per_episode = 1024
        else:
            self.maximum_steps_per_episode = 512  # reduce the overhead for model evaluation
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
        # spatial: 1-6: current joint angles, 7-9: difference between target and current position, 10-13: current rotation as quaternion, 14: distance between target and position
        # lidar: 10 ints that signify occupancy around the ee
        self.observation_space = gym.spaces.Dict(
            {
              'spatial': gym.spaces.Box(low=-2, high=2, shape=(14,), dtype=np.float32),
              'lidar': gym.spaces.Box(low=0, high=2, shape=(10,), dtype=np.int8)  
            }
        )

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
        self._add_object("objects/"+self.dataset["filenames"][file_index], self.xyz_offset)
        # set the target and base target
        target_index = np.random.choice(range(len(self.dataset["frames"][file_index])))  # pick a random target from the welding part's xml
        target_index = 0
        self._set_target(file_index, target_index)

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
        self.joints = self.resting_pose_angles
        self.lidar_probe = self._get_lidar_probe()

        if self.train and self.episodes % self.ee_pos_reward_threshold_change_after_episodes == 0:
            success_rate = np.average(self.success_buffer)
            if success_rate < 0.8 and self.ee_pos_reward_thresh < self.ee_pos_reward_thresh_max:
                self.ee_pos_reward_thresh += self.ee_pos_reward_thresh_increments/4
            elif success_rate > 0.8 and self.ee_pos_reward_thresh > self.ee_pos_reward_thresh_min:
                self.ee_pos_reward_thresh -= self.ee_pos_reward_thresh_increments
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
        spatial[:6] = self.joints
        spatial[6:9] = self.target - self.pos
        spatial[9:13] = self.rot
        spatial[13] = np.linalg.norm(self.target - self.pos)

        return {
            'spatial': spatial,
            'lidar': self.lidar_probe
        }

    def step(self, action):
        
        # all 6 elements of action are expected to be within [-1;1]
        pos_delta = action[:3] * self.pos_speed
        rpy_delta = action[3:] * self.rot_speed

        # get state info
        ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        self.pos = np.array(ee_link_state[4])
        self.pos_last = self.pos
        self.rot = np.array(ee_link_state[5])

        # add the action to the current state
        rot_rpy = util.quaternion_to_rpy(self.rot)
        pos_desired = self.pos + pos_delta
        rpy_desired = rot_rpy + rpy_delta
        quat_desired = util.rpy_to_quaternion(rpy_desired)

        # move the robot to the new positions and get the associated joint config
        self.joints = self._movep(pos_desired, quat_desired)

        # get new state info
        ee_link_state = pyb.getLinkState(self.robot, self.end_effector_link_id, computeForwardKinematics=True)
        self.pos = ee_link_state[4]
        self.rot = ee_link_state[5]
        self.lidar_probe = self._get_lidar_probe()

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

        if not collided:
            if distance_cur < self.ee_pos_reward_thresh:
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
                    reward += -0.0005 * distance_last
        else:
            reward = -10
            done = True
            is_success = False

        if self.steps_current_episode > self.maximum_steps_per_episode:
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
            'success_rate': self.successes/self.episodes,
            'collided': collided,
            'reward': reward,
            'episode_reward': self.episode_reward,
            'distance': distance_cur,
            'episode_distance': self.episode_distance
        }

        if done:
            print(info)

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
        Debug method for setting the joints of the robot to a certain configuration. Will kill all physics, momentum, movement etc. going on.
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
    def _movej(self, targj, speed):
        """
        Move UR5 to target joint configuration.
        Returns the reached joint config.
        """
        currj = self._get_joint_state()
        diffj = targj - currj
        while any(np.abs(diffj) > 1e-2):
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
            maxNumIterations=50,
            residualThreshold=5e-3)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def _set_lidar_cylinder(self, ray_min=0.02, ray_max=0.4, ray_num_ver=6, ray_num_hor=12, render=False):
        ray_froms = []
        ray_tops = []
        frame = util.quaternion_to_matrix(self.rot)
        frame[0:3,3] = self.pos
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

    def _set_target(self, file_index, target_index):
        """
        Fills the target variable with the weldseam at target_index from the xml associated with file_index.
        Also moves the base accordingly.
        """
        frame = self.dataset["frames"][file_index][target_index]
        self.tool = 1 if frame["torch"][3] == "TAND_GERAD_DD" else 0
        positions = [ele["position"] * self.pybullet_mesh_scale_factor + self.xyz_offset for ele in frame["weld_frames"]]
        self.target = positions[0]
        self.target_base = np.average(positions, axis=0)[:2] + self.base_offset
        if self.show_target:
            self._show_target()

    def _show_target(self):
        """
        Creates a green ball around the target of the size of the reward threshold
        """
        visual_id = pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=self.ee_pos_reward_thresh, rgbaColor=[0, 1, 0, 1])
        point_id = pyb.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_id,
                    basePosition=self.target,
                    )