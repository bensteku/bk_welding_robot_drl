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

        # goals: overall collection of weldseams that are left to be dealt with
        # objective: the weldseam that is currently being dealt with
        # plan: concrete collection of robot commands to achieve (the next part of) the objective
        self.goals = []
        self.objective = None
        self.plan = None

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
        self.current_weldseam = 0

    def act(self, obs=None):

        if not obs:
            return None

        # nomenclature:
        # goals: overall collection of weldseams that are left to be dealt with
        # objective: the weldseam that is currently being dealt with
        # plan: concrete collection of robot commands to achieve (a part of) the objective

        if not self.plan:
            if not self._set_plan(obs):
                return None
        #print(self.plan)
        #input("wait a minute")
        action, tool = self.plan.pop(0)
        self.env.switch_tool(tool)
        
        return action

    def _set_objective(self):
        # maybe safeguard against empty goals list here TODO
        if self.goals:
            self.objective = self.goals.pop(0)
            return True
        else:
            return False

    def _set_plan(self, obs):
        if not self.goals:
            return False
        else:
            objective = self.goals.pop(0)
            self.plan = []
            for idx in range(len(objective["weldseams"])-1):
                # general approach: first move the base inbetween the weldseam, then move the end effector to start point
                # then to end point, then move ee back up, then repeat for next seam
                
                # 1. move base:
                # approximate a good position for the robot base for the next weldseam by averaging between start and endpoint
                base_position_apx = np.average(objective["weldseams"][idx:idx+2], axis=0)[:2]
                # get a linear interpolation from current base pos to this estimate
                base_position_todo = util.pos_interpolate(obs["base_position"], base_position_apx, self.env.base_speed)
                # create an array of ee actions such that the ee keeps up with the base
                ee_pos_during_base_movement = [np.array([ele[0], ele[1], 0.5]) for ele in base_position_todo]
                ee_pos_after_base_movement = ee_pos_during_base_movement[-1]  # save for later
                ee_rot_during_base_movement = [obs["rotation"] for ele in base_position_todo]
                if self.env._relative_movement:
                    # convert to relative movement if needed
                    base_position_todo = [(base_position_todo[i]-base_position_todo[i-1] if i>0 else base_position_todo[0]-obs["base_position"]) for i in range(len(base_position_todo))]
                    ee_pos_during_base_movement = [(ee_pos_during_base_movement[i]-ee_pos_during_base_movement[i-1] if i>0 else ee_pos_during_base_movement[0]-obs["position"]) for i in range(len(ee_pos_during_base_movement))]
                    ee_rot_during_base_movement = [np.array([0, 0, 0, 0]) for i in range(len(ee_pos_during_base_movement))]
                for i in range(len(base_position_todo)):
                    act = OrderedDict()
                    act["translate_base"] = base_position_todo[i]
                    act["translate"] = ee_pos_during_base_movement[i]
                    act["rotate"] = ee_rot_during_base_movement[i]
                    self.plan.append((act, objective["tool"]))

                # 2. move ee to start position
                # get linear interpolation to start position
                ee_pos_todo = util.pos_interpolate(ee_pos_after_base_movement, objective["weldseams"][idx], self.env.pos_speed)
                ee_pos_at_welding_start = ee_pos_todo[-1]  # save for later
                base_position_todo = [base_position_apx for i in range(len(ee_pos_todo))]
                # get slerp interpolation of rotations to start rotation
                ee_rot_todo = util.quaternion_interpolate(obs["rotation"], objective["target_rot"][idx], max(len(ee_pos_todo)-2,0))
                if self.env._relative_movement:
                    # convert to relative movement if needed
                    ee_pos_todo = [(ee_pos_todo[i]-ee_pos_todo[i-1] if i>0 else ee_pos_todo[0]-ee_pos_after_base_movement) for i in range(len(ee_pos_todo))]
                    ee_rot_todo = [(ee_rot_todo[i]-ee_rot_todo[i-1] if i>0 else ee_rot_todo[0]-obs["rotation"]) for i in range(len(ee_rot_todo))]
                    base_position_todo = [np.array([0, 0]) for i in range(len(ee_pos_todo))]
                for i in range(len(base_position_todo)):
                    act = OrderedDict()
                    act["translate_base"] = base_position_todo[i]
                    act["translate"] = ee_pos_todo[i]
                    act["rotate"] = ee_rot_todo[i]
                    self.plan.append((act, objective["tool"]))

                # 3. move ee to end position
                # get linear interpolation to end position
                ee_pos_todo = util.pos_interpolate(ee_pos_at_welding_start, objective["weldseams"][idx+1], self.env.pos_speed/5)
                ee_pos_at_welding_end = ee_pos_todo[-1]
                base_position_todo = [base_position_apx for i in range(len(ee_pos_todo))]
                # get slerp interpolation of rotations to end rotation
                ee_rot_todo = util.quaternion_interpolate(objective["target_rot"][idx], objective["target_rot"][idx+1], max(len(ee_pos_todo)-2,0))
                if self.env._relative_movement:
                    # convert to relative movement if needed
                    ee_pos_todo = [(ee_pos_todo[i]-ee_pos_todo[i-1] if i>0 else ee_pos_todo[0]-ee_pos_at_welding_start) for i in range(len(ee_pos_todo))]
                    ee_rot_todo = [(ee_rot_todo[i]-ee_rot_todo[i-1] if i>0 else ee_rot_todo[0]-objective["target_rot"][idx]) for i in range(len(ee_rot_todo))]
                    base_position_todo = [np.array([0, 0]) for i in range(len(ee_pos_todo))]
                for i in range(len(base_position_todo)):
                    act = OrderedDict()
                    act["translate_base"] = base_position_todo[i]
                    act["translate"] = ee_pos_todo[i]
                    act["rotate"] = ee_rot_todo[i]
                    self.plan.append((act, objective["tool"]))

                # 4. move ee upwards
                # get linear interpolation to end position
                ee_pos_todo = util.pos_interpolate(ee_pos_at_welding_end, np.array([ee_pos_at_welding_end[0], ee_pos_at_welding_end[1], 0.5]), self.env.pos_speed)
                base_position_todo = [base_position_apx for i in range(len(ee_pos_todo))]
                ee_rot_todo = [objective["target_rot"][idx+1] for ele in base_position_todo]
                if self.env._relative_movement:
                    # convert to relative movement if needed
                    ee_pos_todo = [(ee_pos_todo[i]-ee_pos_todo[i-1] if i>0 else ee_pos_todo[0]-ee_pos_at_welding_end) for i in range(len(ee_pos_todo))]
                    ee_rot_todo = [np.array([0, 0, 0, 0]) for i in range(len(ee_rot_todo))]
                    base_position_todo = [np.array([0, 0]) for i in range(len(ee_pos_todo))]
                for i in range(len(base_position_todo)):
                    act = OrderedDict()
                    act["translate_base"] = base_position_todo[i]
                    act["translate"] = ee_pos_todo[i]
                    act["rotate"] = ee_rot_todo[i]
                    self.plan.append((act, objective["tool"]))
            
            
            return True
