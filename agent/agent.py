import environment.environment as env
import os
import numpy as np
from util import util, planner
from model.model import AgentModelSimple
import pybullet as pyb

class Agent:

    def __init__(self):

        # welding environment as defined in the other file, needs to be set after construction due to mutual dependence
        self.env = None

    def _set_env(self, senv):
        """
        Method for making known the env in which the agent is supposed to be working.
        Is supposed to be called from the env constructor.
        """

        self.env = senv
        
class AgentPybulletNN(Agent):
    """
    Pybullet agent that is driven by neural net implemented in PyTorch.
    """
    
    def __init__(self):
        super().__init__()
        self.model = AgentModelSimple()

    def act(self, agent_state):
        """
        Act function for this agent. Parameters are slightly different: instead of a gym obs (OrderedDict) it takes a Pytorch tensor
        which contains the gym information + information about the agents current goals (to differentiate between the two, it's called an agent_state instead of obs)
        """

        # ask the NN
        action_tensor = self.model.choose_action(agent_state)
        action = action_tensor.cpu().detach().numpy()

        return action

class AgentPybulletRRTPlanner(Agent):
    """
    Agent that acts based on an explicit algorithm:
    - always moves base towards next objective
    - uses RRT with ground truth to get trajectory to starting position
    - interpolates linearly between starting and finishing position while welding
    - again uses RRT to move ee back up to safe height after finishing welding
    It uses actions that are expressed in joint states and must therefore use the env subclass dedicated to that.
    """

    def __init__(self):
        super().__init__()
        self.trajectory = []

    def act(self, obs):

        # action as flat array of floats
        # index 0-5: joints
        # set to current joints as defaults
        action =  obs[8:14]

        if self.env.path_state == 0:
            if not self.trajectory:
                objective_with_slight_offset = self.env.objective[0] + self.env.objective[1][0] * 0.01 + self.env.objective[1][1] * 0.01  # target position + a small part of the face norms of the weld seam
                q_cur = self.env.get_joint_state()
                q_goal = self.env.solve_ik((objective_with_slight_offset, self.env._quat_w_to_ee(self.env.objective[2])))
                # the following line is needed because the configuration returned by inverse kinematics is often larger than 5e-3 away from the objective in cartesian space
                pos_goal_from_q_goal = self.env.solve_fk(q_goal)[0]
                traj_raw = planner.bi_rrt(q_cur, q_goal, 0.35, self.env.robot, self.env.joints, self.env.obj_ids, 500000, 1e-3, self.env.end_effector_link_id[self.env.robot_name], obs[1], pos_goal_from_q_goal, 25e-3, 300, save_all_paths=False, base_position=obs[:2])
                self.trajectory = planner.interpolate_path(traj_raw)
            next_q = self.trajectory.pop(0)
            action = next_q
        elif self.env.path_state == 1:
            if not self.trajectory:
                pos = util.pos_interpolate(obs[2:5], self.env.objective[0] + self.env.objective[1][0] * 0.01 + self.env.objective[1][1] * 0.01, self.env.pos_speed/3.25)
                quat = util.quaternion_interpolate(pyb.getQuaternionFromEuler(obs[5:8]), self.env.objective[2], len(pos)-2)
                quat = [self.env._quat_w_to_ee(qu) for qu in quat]
                self.trajectory = list(zip(pos, quat))
            next_pose = self.trajectory.pop(0)
            q_next = self.env.solve_ik(next_pose)
            action = q_next
        elif self.env.path_state == 2:
            if not self.trajectory:
                q_cur = self.env.get_joint_state()
                # move ee to halfway between base and current position and height 0.5
                diff = (obs[:2] - obs[2:4])/2
                goal_pos = diff + obs[2:4]
                goal_pos = np.array([goal_pos[0], goal_pos[1], 0.5])
                q_goal = self.env.solve_ik((goal_pos, self.env._quat_w_to_ee(np.array([0, 0, 0, 1]))))
                # the following line is needed because the configuration returned by inverse kinematics is often larger than 5e-3 away from the objective in cartesian space
                pos_goal_from_q_goal = self.env.solve_fk(q_goal)[0]
                traj_raw = planner.bi_rrt(q_cur, q_goal, 0.35, self.env.robot, self.env.joints, self.env.obj_ids, 500000, 1e-3, self.env.end_effector_link_id[self.env.robot_name], obs[1], pos_goal_from_q_goal, 25e-3, 300, save_all_paths=False, base_position=obs[:2])
                self.trajectory = planner.interpolate_path(traj_raw)
            next_q = self.trajectory.pop(0)
            action = next_q

        return action
