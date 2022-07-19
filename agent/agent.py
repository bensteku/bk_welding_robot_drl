import environment.environment as env
import xml.etree.ElementTree as ET
import os
import numpy as np
from util import xml_parser, util
from scipy.spatial.transform import Rotation
from collections import OrderedDict
import pybullet as pyb

PYBULLET_SCALE_FACTOR = 0.0005

class Agent:

    def __init__(self, asset_files_path):
        
        # contains the location of the objs and their respective xmls with welding seams
        self.asset_files_path = asset_files_path
        self.dataset = self._register_data()

        # state variable, 0: pathing, 1: welding
        self.welding = 0  

        # array of goals (start and finish positions per weld seam)
        self.goals = []
        self.current_objective = None
        self.next_target_pos = None

        # welding environment as defined in the other file, needs to be set after construction due to mutual dependence
        self.env = None

    def _set_env(self, senv: env.WeldingEnvironment):
        """
        Method for making known the env in which the agent is supposed to be working.
        May replace later on by making the env a param in the constructor.
        """

        self.env = senv

    def _register_data(self):
        """
        To be implemented in the subclasses.
        Should return dict with two keys: files with list of source files as value & frames with data from xml files given by xml_parser
        """

        raise NotImplementedError

    def _set_goals(self, index):
        """
        To be implemented in the subclasses.
        Should extract the relevant weld frame data from dataset, transform it as appropriate and stack it in order into the goals array.
        """

        raise NotImplementedError
        

class AgentPybullet(Agent):

    def __init__(self, asset_files_path):
        
        super().__init__(asset_files_path)
        self.xyz_offset = np.array((0, 0, 0.01))  # offset in pybullet coordinates, location to place the objects into

    def load_object_into_env(self, index):

        if self.env is None:
            raise ValueError("env needs to be set before agent can act upon it!")

        self.env.add_object(os.path.join(self.asset_files_path, self.dataset["filenames"][index]), 
        pose = (self.xyz_offset, [0, 0, 0, 1]),
        category = "fixed" )

    def _register_data(self):
        """
        Scans URDF(obj) files in asset path and creates a list, associating file name with weld seams and ground truths.
        """

        filenames = []
        frames = []
        for file in [file_candidate for file_candidate in os.listdir(self.asset_files_path) if os.path.isfile(os.path.join(self.asset_files_path, file_candidate))]:
            if ".urdf" in file:
                if ( os.path.isfile(os.path.join(self.asset_files_path, file.replace(".urdf",".xml"))) and 
                os.path.isfile(os.path.join(self.asset_files_path, file.replace(".urdf",".obj"))) ):
                    
                    frames.append(xml_parser.parse_frame_dump(os.path.join(self.asset_files_path, file.replace(".urdf",".xml"))))
                    filenames.append(file)
                else:
                    raise FileNotFoundError("URDF file "+file+" is missing its associated xml or obj!")

        return { "filenames": filenames, "frames": frames }

    def _set_goals(self, index):
        """
        Uses the dataset to load in the weldseams of the file indicated by index into the goals list.

        Args:
            index: int, signifies index in the dataset array
        """

        frames = self.dataset["frames"][index]
        for frame in frames:
            tmp = {}
            tmp["weldseams"] = [ele["position"] * PYBULLET_SCALE_FACTOR + self.xyz_offset for ele in frame["weld_frames"]]
            tmp["norm"] = [ele["norm"] for ele in frame["weld_frames"]]
            tmp["target_pos"] = [ele[:3,3] * PYBULLET_SCALE_FACTOR + self.xyz_offset for ele in frame["pose_frames"]]
            tmp["target_rot"] = [util.matrix_to_quaternion(ele[:3,:3]) for ele in frame["pose_frames"]]
            tmp["tool"] = 1 if frame["torch"][3] == "TAND_GERAD_DD" else 0           
            self.goals.append(tmp)

    def is_done(self):
        if self.goals:
            return False
        else:
            return True

    def _set_objective(self):
        """
        Method for translating elements of goal array into concrete instructions for actions
        """

        objs = self.goals.pop(0)
        


class AgentPybulletDemonstration(AgentPybullet):
    """
    Agent without NN that randomly tests out actions, used to generate a data set for training.
    """
    
    def __init__(self, asset_files_path):
        
        super().__init__(asset_files_path)
        

    def act(self, obs=None):
        """
        Returns a random sample of the space of possible actions, the obs paramter is just to conform with a regular Gym act method.
        """
        if self.env is None:
            raise ValueError("env needs to be set before agent can act upon it!")

        return self.env.action_space.sample()

    def create_dataset(self, num_episodes, num_steps):
        """
        Method to create a dataset consisting of state, action and rewards organized by run that is later used to train the NN.

        Args:
            num_episodes: int, number of overall runs (meaning that it will be the length of array given as return)
            num_steps: int, amount of actions taken per run (meaning that it will be the length of an element of the return array) #TODO: maybe per goal in run

        Returns:
            list containing all runs as chronologically ordered lists of state, actions and rewards
        """

        if self.env is None:
            raise ValueError("env needs to be set before agent can act upon it!")

        file_indices = list(range(len(self.dataset["filenames"])))

        episodes = []
        for i in range(num_episodes):
            self.env.reset()

            # pick a random .obj file, then remove it from the list
            # if the list is empty, refill it
            if not len(file_indices):
                random_file = np.random.choice(file_indices)
            else:
                file_indices = list(range(len(self.dataset["filenames"])))
            file_indices.remove(random_file)

            step = [{ "filename": self.dataset["filenames"][random_file], "steps":[] }]

            self.load_object_into_env(random_file)

            self.env.reset()

            # main act loop
            for i in range(num_steps):
                obs = self.env._get_obs()
                act = self.act()
                self.env.step(act)
                reward = self.reward()

        
class AgentPybulletOracle(AgentPybulletDemonstration):
    """
    Agent without NN that uses the ground truth to approximate an optimal trajectory, used to generate dataset
    """

    def __init__(self, asset_files_path):
        
        super().__init__(asset_files_path)

    def act(self, obs=None):

        if not obs:
            return None

        if not self.current_objective:
            self.current_objective = self.goals.pop(0)
        if not self.next_target_pos:
            # simple approximation of a suitable base position by averaging the all weldseams
            base_position_apx = np.average(self.current_objective["weldseams"], axis=0)
            self.next_target_pos = {"base_pos_target": base_position_apx[:2], "ee_pos_target": self.current_objective["target_pos"].pop(0), "ee_rot_target": self.current_objective["target_rot"].pop(0)}

        # base difference
        base_diff = self.next_target_pos["base_pos_target"] - obs["base_position"]
        base_diff_norm = np.linalg.norm(base_diff)
        if base_diff_norm > self.env.base_speed:
            base_diff_direction =  base_diff / np.linalg.norm(base_diff)
            base_position_command = base_diff_direction * self.env.base_speed
        else:
            base_position_command = base_diff

        # ee difference
        ee_diff = self.next_target_pos["ee_pos_target"] - obs["position"]
        ee_diff_norm = np.linalg.norm(ee_diff)
        if ee_diff_norm > self.env.pos_speed:
            ee_diff_direction =  ee_diff / np.linalg.norm(ee_diff)
            ee_position_command = ee_diff_direction * self.env.pos_speed
        else:
            ee_position_command = ee_diff

        # ee difference
        quat_diff = self.next_target_pos["ee_rot_target"] - obs["rotation"]
        quat_diff_norm = np.linalg.norm(quat_diff)
        if quat_diff_norm > self.env.pos_speed:
            quat_diff_direction =  quat_diff / np.linalg.norm(quat_diff)
            quat_position_command = quat_diff_direction * self.env.pos_speed
        else:
            quat_position_command = quat_diff
        
        action = OrderedDict()
        action["translate_base"] = base_position_command
        action["translate"] = ee_position_command
        action["rotate"] = quat_position_command
        
        return action


