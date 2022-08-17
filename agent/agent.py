import environment.environment as env
import os
import numpy as np
from util import xml_parser, util, planner
from collections import OrderedDict
from model import model
import pybullet as pyb
from model.model import AgentModelSimple

PYBULLET_SCALE_FACTOR = 0.0005

class Agent:

    def __init__(self, asset_files_path):
        
        # contains the location of the objs and their respective xmls with welding seams
        self.asset_files_path = asset_files_path
        self.dataset = self._register_data()

        # state variable, 0: base pathing, 1: moving ee down to weld seam, 2: welding, 3: moving ee back up
        self.state = 0  

        # goals: overall collection of weldseams that are left to be dealt with
        # plan: expansion of intermediate steps containted within one weldseam
        # objective: the next part of the plan    
        self.goals = []
        self.objective = None
        self.plan = None

        # welding environment as defined in the other file, needs to be set after construction due to mutual dependence
        self.env = None

    def next_state(self):
        """
        Method that iterates the state of the agent. Called from outside if certain goal conditions are fulfilled.
        """

        # if agent was in state 1 or 3 the objective is now succesfully completed
        if self.state == 1 or self.state == 3:
            self.objective = None
        # increment state or wrap back around if in state 3
        self.state = self.state + 1 if self.state < 3 else 0
        return self.state

    def _set_env(self, senv: env.WeldingEnvironment):
        """
        Method for making known the env in which the agent is supposed to be working.
        Is supposed to be called from the env constructor.
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
        Should extract the relevant weld frame data from dataset, transform it as appropriate and stack it in its original order into the goals array.
        """

        raise NotImplementedError
        

class AgentPybullet(Agent):

    def __init__(self, asset_files_path):
        
        super().__init__(asset_files_path)

        # offset in pybullet coordinates, location to place the objects into
        self.xyz_offset = np.array((0, 0, 0.01))  

        # radius of the circle around the goal position for the robot base in which a full reward will be given
        self.base_pos_reward_thresh = 0.5
        # same as above, just for the end effector position
        self.ee_pos_reward_thresh = 8e-2  # might need adjustment
        # same as above, just for quaternion similarity (see util/util.py method)
        self.quat_sim_thresh = 4e-2  # this probably too

        # pybullet id of the part the agent is currently dealing with
        self.current_part_id = None

        # height at which the ee is transported when not welding
        self.safe_height = 0.5

    def load_object_into_env(self, index):
        """
        Method for loading an object into the simulation.
        Args:
            - index: index of the desired file in the dataset list
        """

        self.current_pard_id = self.env.add_object(os.path.join(self.asset_files_path, self.dataset["filenames"][index]), 
                                                    pose = (self.xyz_offset, [0, 0, 0, 1]),
                                                    category = "fixed" )
        self._set_goals(index)

    def _get_obs(self):
        """
        Returns the an observation of the agent (in reference to the env's observation).
        This consists of an observation of the environment and the agent's current objective.
        Arranged in such a way that it's convenient for usage with PyTorch.
        """
        obs = self.env._get_obs()
        if self.objective:
            state = np.hstack((obs["base_position"], obs["position"], obs["rotation"], self.objective[0], self.objective[1][0], self.objective[1][1]))
        else:
            state = np.hstack((obs["base_position"], obs["position"], obs["rotation"], [0, 0, 0], [1, 0, 0], [1, 0, 0]))
        return state

    def _register_data(self):
        """
        Scans URDF(obj) files in asset path and creates a list, associating file name with weld seams and ground truths.
        This can can later be used to load these objects into the simulation, see load_object_into_env method.
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
        One element in goal array <=> one weldseam in the original xml

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
        """
        Indicates whether the agent is done with the current welding piece.
        Determinde by whether the goal list is empty or not.
        """
        if self.goals:
            return False
        else:
            return True

    def _set_plan(self):
        """
        Sets the plan by extracting all intermediate steps from the foremost element in the goals list.
        """
        if self.goals:
            self.plan = []
            goal = self.goals.pop(0)
            for idx in range(len(goal["weldseams"])):
                tpl = (goal["weldseams"][idx], goal["norm"][idx], goal["target_rot"][idx], goal["tool"])
                self.plan.append(tpl)
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

    def reward(self, obs=None, timeout=False):
        """
        Method for calculating the reward the agent gets for the state it's currently in.
        Args:
            - obs: env observation (not agent!)
            - timeout: bool flag for when the movement of the last action timed out
        Returns:
            - reward as float number
            - boolean indicating if the objectives for the current state have been fulfilled
        """
        
        # bool flags for when the objective is achieved
        pos_done, rot_done, base_done = False, False, False
        # base reward
        reward = 0

        if self.state == 0:
            # if the arm is in moving mode, give out rewards for moving towards the general region of the objective
            # give full reward if a circular region around the xy-position of the next objective is reached
            distance = np.linalg.norm(self.objective[0][:2] - obs["base_position"])
            
            if distance < self.base_pos_reward_thresh:
                reward += 10
                base_done = True
            else:
                # exponential decay up to threshold
                reward += util.exp_decay(distance, 10, 3*self.base_pos_reward_thresh)
        elif self.state == 3:
            # state for moving the ee upwards after one line has been welded

            # simply measure distance up to safe height
            distance = np.linalg.norm(np.array([obs["position"][0], obs["position"][1], self.safe_height]) - obs["position"]) 
            if distance < self.ee_pos_reward_thresh:
                reward += 10
                pos_done = True
                rot_done = True
            else:
                reward += util.exp_decay(distance, 10, 5*self.ee_pos_reward_thresh)
        else:
            # if the arm is in welding mode or moving to the start position for welding give out a reward in concordance to how far away it is from the desired position and how closely
            # it matches the ground truth rotation
            # if the robot is in a problematic configuration (collision or not reachable(timeout)) give out a negative reward
            distance = np.linalg.norm(self.objective[0] - obs["position"]) 
            
            if distance < self.ee_pos_reward_thresh:
                reward += 20
                pos_done = True  # objective achieved
            else:
                reward += util.exp_decay(distance, 20, 20*self.ee_pos_reward_thresh)
            
            quat_sim = util.quaternion_similarity(self.objective[2], obs["rotation"])    
            if quat_sim > 1-self.quat_sim_thresh:
                rot_done = True

            print(distance, self.ee_pos_reward_thresh)
            print(quat_sim, 1-self.quat_sim_thresh)
            
            reward = reward * (quat_sim**0.5)  # take root of quaternion similarity to dampen its effect a bit
        
        # hand out penalties
        if timeout:
            reward -= 20
        if self.env.is_in_collision(self.current_pard_id): #and not (pos_done and rot_done):
            reward -= 50
            pos_done = False
            rot_done = False

        if pos_done and rot_done:
            self.objective = None

        return reward, pos_done and rot_done or base_done
        
class AgentPybulletNN(AgentPybullet):
    """
    Pybullet agent that is driven by neural net implemented in PyTorch.
    """
    
    def __init__(self, asset_files_path):
        super().__init__(asset_files_path)
        self.model = AgentModelSimple()

    def act(self, agent_obs):
        """
        Act function for this agent. Parameters are slightly different: instead of a gym obs (OrderedDict) it takes a Pytorch tensor
        which contains the gym information + information about the agents current goals (to differentiate between the two, it's called a agent_obs instead of obs)
        """
        
        if not self.plan and not self.objective:  # only create new plan if there's also no current objective
            self._set_plan()
        if not self.objective:
            self._set_objective()

        tool = self.objective[3]
        self.env.switch_tool(tool)

        ## call neural net for action TODO
        action_tensor = self.model.choose_action(agent_obs)
        action_array = action_tensor.cpu().detach().numpy()
        action = OrderedDict()
        action["translate_base"] = action_array[:2]
        action["translate"] = action_array[2:5]
        action["rotate"] = action_array[5:]

        if self.state == 0:
            action["translate"] = np.array([action["translate_base"][0], action["translate_base"][1], 0])
            action["rotate"] = np.array([0, 0, 0, 1])

        return action

class AgentPybulletRRTPlanner(AgentPybullet):
    """
    Agent that acts based on an explicit algorithm:
    - always moves base towards next objective
    - uses RRT with ground truth to get trajectory to starting position
    - interpolates linearly between starting and finishing position while welding
    - again uses RRT to move ee back up to safe height after finishing welding
    It uses actions that are expressed in joint states and must therefore use the env subclass dedicated to that.
    """

    def __init__(self, asset_files_path):
        super().__init__(asset_files_path)
        self.trajectory = []

    def act(self, obs):

        if not self.plan and not self.objective:  # only create new plan if there's also no current objective
            self._set_plan()
        if not self.objective:
            self._set_objective()

        action = OrderedDict()
        action["translate_base"] = np.array([0, 0])
        action["joints"] = obs["joints"]
        #action["translate"] = np.array([0, 0, 0])
        #action["rotate"] = np.array([0, 0, 0, 1])

        if self.state == 0:
            diff = self.objective[0][:2] - obs["base_position"]
            dist = np.linalg.norm(diff)
            if dist < self.env.base_speed:
                action["translate_base"] = diff
                #action["translate"] = np.array([diff[0], diff[1], 0.5 - obs["position"][2]])
            else:
                diff = diff * (self.env.base_speed/dist)
                action["translate_base"] = diff
                #action["translate"] = np.array([diff[0], diff[1], 0.5 - obs["position"][2]])
        elif self.state == 1:
            if not self.trajectory:
                q_cur = self.env.get_joint_state()
                q_goal = self.env.solve_ik((self.objective[0] + self.objective[1][0] * 0.025 + self.objective[1][1] * 0.025, self.env._quat_w_to_ee(self.objective[2])))
                traj_raw = planner.bi_rrt(q_cur, q_goal, 0.15, self.env.robot, self.env.joints, self.env.obj_ids["fixed"][0], 500000, 5e-4, 300)
                self.trajectory = planner.interpolate_path(traj_raw)
            next_q = self.trajectory.pop(0)
            #pos, quat = self.env.solve_fk(next_q)
            #action["translate"] = pos - obs["position"]
            #action["rotate"] = quat
            action["joints"] = next_q
        elif self.state == 2:
            if not self.trajectory:
                pos = util.pos_interpolate(obs["position"], self.objective[0] + self.objective[1][0] * 0.025 + self.objective[1][1] * 0.025, self.env.pos_speed/2)
                quat = util.quaternion_interpolate(obs["rotation"], self.objective[2], len(pos)-2)
                quat = [self.env._quat_w_to_ee(qu) for qu in quat]
                self.trajectory = list(zip(pos, quat))
            next_pose = self.trajectory.pop(0)
            q_next = self.env.solve_ik(next_pose)
            action["joints"] = q_next
        elif self.state == 3:
            if not self.trajectory:
                q_cur = self.env.get_joint_state()
                # move ee to halfway between base and current position and height 0.5
                diff = (obs["base_position"] - obs["position"][:2])/2
                goal_pos = diff + obs["position"][:2]
                goal_pos = np.array([goal_pos[0], goal_pos[1], 0.5])
                q_goal = self.env.solve_ik((goal_pos, self.env._quat_w_to_ee(np.array([0, 0, 0, 1]))))
                traj_raw = planner.bi_rrt(q_cur, q_goal, 0.15, self.env.robot, self.env.joints, self.env.obj_ids["fixed"][0], 500000, 1e-3, 300)
                self.trajectory = planner.interpolate_path(traj_raw)
            next_q = self.trajectory.pop(0)
            #pos, quat = self.env.solve_fk(next_q)
            #action["translate"] = pos - obs["position"]
            #action["rotate"] = quat
            action["joints"] = next_q

        return action
