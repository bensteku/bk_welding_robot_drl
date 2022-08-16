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
        # objective: the weldseam that is currently being dealt with
        # plan: concrete collection of robot commands to achieve (the next part of) the objective
        self.goals = []
        self.objective = None
        self.plan = None

        # welding environment as defined in the other file, needs to be set after construction due to mutual dependence
        self.env = None

    def next_state(self):
        if self.state == 1 or self.state == 3:
            self.objective = None
        self.state = self.state + 1 if self.state < 3 else 0
        return self.state

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
        self.base_pos_reward_thresh = 0.5
        self.ee_pos_reward_thresh = 8e-2  # really TODO
        self.quat_sim_thresh = 4e-2  # this probably too
        self.current_part_id = None

    def load_object_into_env(self, index):

        if self.env is None:
            raise ValueError("env needs to be set before agent can act upon it!")

        self.current_pard_id = self.env.add_object(os.path.join(self.asset_files_path, self.dataset["filenames"][index]), 
                                                    pose = (self.xyz_offset, [0, 0, 0, 1]),
                                                    category = "fixed" )
        self._set_goals(index)

    def get_state(self):
        """
        Returns the state of the agent.
        This consists of an observation of the environment and the agent's current objective.
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
        if self.goals:
            return False
        else:
            return True

    def _set_plan(self):
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
        Method for translating elements of goal array into concrete instructions for actions
        """
        if self.plan:
            self.objective = self.plan.pop(0)
            return True
        else:
            return False

    def reward(self, obs=None, timeout=False):
        
        # TODO: implement collision check
        pos_done, rot_done, base_done = False, False, False
        reward = 0
        if self.state == 0:
            # if the arm is in moving mode, give out rewards for moving towards the general region of the objective
            # quadratic reward to create smooth gradient
            distance = np.linalg.norm(self.objective[0][:2] - obs["base_position"])
            
            if distance < self.base_pos_reward_thresh:
                reward += 10
                base_done = True
            else:
                reward += (-10.0/(9*self.base_pos_reward_thresh**2)) * distance ** 2 + 10  # quadratic function: 10 at threshold, 0 at 3*threshold
        elif self.state == 3:
            # move upwards
            distance = np.linalg.norm(np.array([obs["position"][0], obs["position"][1], 0.5]) - obs["position"]) 
            #reward += (-10.0/(9*self.base_pos_reward_thresh**2)) * distance ** 2 + 10
            if distance < self.ee_pos_reward_thresh:
                reward += 10
                pos_done = True
                rot_done = True
            else:
                reward += util.exp_decay(distance, 10, 5*self.ee_pos_reward_thresh)
        else:
            # if the arm is in welding mode give out a reward in concordance to how far away it is from the desired position and how closely
            # it matches the ground truth rotation
            # if the robot is in an invalid state (as determined by a timeout in the movement method) give a negative reward
            distance = np.linalg.norm(self.objective[0] - obs["position"]) 
            
            if distance < self.ee_pos_reward_thresh:
                reward += 20
                pos_done = True  # objective achieved
            else:
                #reward += (-20.0/(9*self.ee_pos_reward_thresh**2)) * distance ** 2 + 20  # quadratic function: 20 at threshold, 0 at 3*threshold
                reward += util.exp_decay(distance, 20, 20*self.ee_pos_reward_thresh)
            
            quat_sim = util.quaternion_similarity(self.objective[2], obs["rotation"])    
            if quat_sim > 1-self.quat_sim_thresh:
                rot_done = True

            print(distance, self.ee_pos_reward_thresh)
            print(quat_sim, 1-self.quat_sim_thresh)
            
            reward = reward * (quat_sim**0.5)
        if timeout:
            reward -= 75
        if self.env.is_in_collision(self.current_pard_id) and not (pos_done and rot_done):
            reward -= 150
            pos_done = False
            rot_done = False

        if pos_done and rot_done:
            self.objective = None

        return reward, pos_done and rot_done or base_done
        
class AgentPybulletNN(AgentPybullet):
    
    def __init__(self, asset_files_path):
        super().__init__(asset_files_path)
        self.model = AgentModelSimple()

    def act(self, state):
        """
        Act function for this agent. Parameters are slightly different: instead of a gym obs (OrderedDict) it takes a Pytorch tensor
        which contains the gym information + information about the agents current goals (to differentiate between the two, it's called a state instead of obs)
        """

        
        if not self.plan and not self.objective:  # only create new plan if there's also no current objective
            self._set_plan()
        if not self.objective:
            self._set_objective()

        tool = self.objective[3]
        self.env.switch_tool(tool)

        ## call neural net for action TODO
        action_tensor = self.model.choose_action(state)
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
