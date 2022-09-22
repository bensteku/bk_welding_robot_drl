import gym
import pybullet as pyb
import numpy as np
from util import util, xml_parser
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

        # threshold for the end effector position for maximum reward
        # == a sphere around the target where maximum reward is applied
        # might be modified in the course of training
        self.ee_pos_reward_thresh = 2e-2

        # variables storing information about the current state, saves performance from having to 
        # access the pybullet interface all the time
        self.pos = None
        self.rot = None
        self.pos_last = None
        self.target = None
        self.lidar_probe = None

        # process the dataset of meshes and urdfs
        self.dataset = self._register_data()

        # steps taken in current epsiode
        self.step_counter = 0
        # maximum steps per episode
        self.maximum_steps_per_episode = 1024
        # episode counter
        self.episodes = 0

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

    ###############
    # Gym methods #
    ###############

    def reset(self):

        # clear out stored objects and reset the simulation
        self.obj_ids = []
        pyb.resetSimulation()

        # stop pybullet rendering for performance
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)

        # rebuild environment
        # load in the ground plane
        self._add_object("worksapce/plane.urdf", [0, 0, -0.01])

        # load in the mesh of the welding part
        # info: the method will load the mesh at given index within the self.dataset variable
        file_index = np.random.choice(range(len(self.dataset["filenames"]))) 
        file_index = self.dataset["filenames"].index("201910204483_R1.urdf")
        self._add_object("objects/"+self.dataset["filenames"][file_index], self.xyz_offset)
        # set the target
        target_index = 

        # load in the robot, the correct tool was set above while loading in the welding part
        if self.tool:
            self.robot = self._add_object("kr16/kr16_tand_gerad.urdf", [0, 0, self.ceiling_mount_height], pyb.getQuaternionFromEuler([np.pi, 0, 0]))
        else:
            self.robot = self._add_object("kr16/kr16_mrw510.urdf", [0, 0, self.ceiling_mount_height], pyb.getQuaternionFromEuler([np.pi, 0, 0]))      

        # get the joint ids of the robot and set the joints to their resting position
        joints = [pyb.getJointInfo(self.robot, i) for i in range(pyb.getNumJoints(self.robot))]
        self.joint_ids = [j[0] for j in joints if j[2] == pyb.JOINT_REVOLUTE]
        self._set_joint_state(self.resting_pose_angles)

        # turn on rendering again
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)

    def step(self, action):
        pass

    def _reward(self):
        pass

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
        return np.array([pyb.getJointState(self.robot, i)[0] for i in self.joints])
    
    def _set_joint_state(self, config):
        """
        Debug method for setting the joints of the robot to a certain configuration. Will kill all physics, momentum, movement etc. going on.
        """
        for i in range(len(self.joint_ids)):
            pyb.resetJointState(self.robot, self.joint_ids[i], config[i])

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

    def _load_in_welding_part(self, index):
        """
        Loads the welding part mesh from index in the self.dataset variable, picks a random 
        """