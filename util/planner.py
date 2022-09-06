from argparse import ArgumentError
import numpy as np
import pybullet as pyb
from scipy.interpolate import interp1d
from time import time
import uuid

class Node:
    """
    Class for nodes in the Rapidly Exploring Random Tree.
    """
    
    def __init__(self, q, parent):
        self.q = np.array(q)  # configuration of a robot
        self.parent = parent
        #self.children = []  # not needed atm
        self.delta = 0  # attribute that gets incremented every time a node is added further down this branch

class Tree:
    """
    Class for a Rapidly Exploring Random Tree.
    """
    
    def __init__(self, q):
        self.root = Node(q, None)
        self.nodes = [self.root]

    def add_node(self, q, parent = None):
        """
        Adds a new node with the config q and a parent node.
        """
        if not parent:
            closest = self.nearest_neighbor(q)
        else:
            closest = parent
        new = Node(q, closest)
        #closest.children.append(new)  # not needed atm
        self.nodes.append(new)
        tmp = new.parent
        tmp.delta += 1
        # increment delta values up the chain
        while tmp.parent is not None:
            tmp = tmp.parent
            tmp.delta += 1
        return new

    def nearest_neighbor(self, q, control):
        """
        Gets the node within the tree that is closest in Euclidean distance to configuration q and has delta smaller than control.
        """
        min_dist = np.Inf
        closest = None
        for node in self.nodes:
            dist = np.linalg.norm(np.array(q) - np.array(node.q))
            if dist < min_dist and node.delta < control:
                min_dist = dist
                closest = node
        return closest

    def all_possible_paths(self):
        start_nodes = []
        for node in self.nodes:
            if node.delta == 0:
                start_nodes.append(node)
        paths = []
        for node in start_nodes:
            tmp = node
            path = []
            path.append(tmp.q)
            while tmp.parent is not None: 
                tmp = tmp.parent
                path.append(tmp.q)
            paths.append(list(reversed(path)))
        if not start_nodes:
            paths.append([self.root.q])
        return paths

def collision_fn(robot, joints, objs, ee, pos_start, pos_goal, thresh):
    """
    Wrapper function, returns a collision check for the given arguments.
    Additionally, the function will ignore collision if the current configuration is close enough to
    either goal or start configuration in cartesian space as determined by a threshold.
    This serves to make it so that the algorithm will be able to deal with start/goal configurations
    that are placed inside a mesh.
    Args:
        - robot: Pybullet object id of the robot
        - joints: list/array of Pybullet joint ids of the robot
        - obj: Pybullet object id for which to check collision with robot
        - ee: Pybullet link id of the end effector of the robot
        - pos_start: Cartesian position of the start configuration
        - pos_goal: Same for the goal configuration
        - thresh: threshold distance for ignoring the collision check around start and goal position
    """
    def collision(q):
        # get current config
        currj = [pyb.getJointState(robot, i)[0] for i in joints]
        for joint, val in zip(joints, q):
            pyb.resetJointState(robot, joint, val)    


        # if the q is within the vicinity of the goal or start configurations in cartesian space, skip collision detection
        # so that weld seams that are positioned inside a mesh can be dealt with
        
        cartesian_pos = np.array(pyb.getLinkState(
                            bodyUniqueId=robot,
                            linkIndex=ee,
                            computeForwardKinematics=True
            )[0])
        if np.linalg.norm(cartesian_pos - pos_start) < thresh or np.linalg.norm(cartesian_pos - pos_goal) < thresh:
            return False
        
        pyb.performCollisionDetection()  # perform just the collision detection part of the PyBullet engine
        col = False
        for obj in objs:
            if pyb.getContactPoints(robot, obj):
                col = True
                break

        # reset joints to original config
        for joint, val in zip(joints, currj):
            pyb.resetJointState(robot, joint, val)
        #col_self = True if pyb.getContactPoints(robot, robot, 1, 7) else False
        
        return col #and col_self
    return collision

def sample_fn(joints, lower, upper, collision):
    """
    Wrapper function, returns a uniform sampler for a configuration space defined by the arguments.
    Args:
        - joints: list/array of Pybullet joint ids of the robot
        - lower: list/array of lower joint limits
        - upper: list/array of upper joint limits
        - collision: collision function as obtained by calling collision_fn
    """
    def sample():
        sample = np.random.uniform(low=np.array(lower), high=np.array(upper), size=len(joints))
        while collision(sample):
            sample = np.random.uniform(low=np.array(lower), high=np.array(upper), size=len(joints))
        return sample
    return sample

def connect_fn(collision, epsilon = 1e-3):
    """
    Wrapper function, returns a connect function for a given collision function and step size epsilon.
    Args:
        - collision: collision function as obtained by calling collision_fn
        - epsilon: step size in configuration space for collision checks when connecting two nodes
    """
    def connect(tree, q, control):
        node_near = tree.nearest_neighbor(q, control)
        q_cur = node_near.q
        q_old = q_cur
        # connect loop
        while True:   
            diff = q - q_cur
            dist = np.linalg.norm(diff)
            if epsilon > dist:
                q_cur = q
            else:
                q_old = q_cur
                q_cur = q_cur + (epsilon/dist) * diff
            if np.array_equal(q_cur,q):
                return tree.add_node(q, node_near), 0
            col = collision(q_cur)
            if col and np.array_equal(q_old, node_near.q):
                return node_near, 2  
            elif col and not np.array_equal(q_old, node_near.q):
                return tree.add_node(q_old, node_near), 1
    return connect

def bi_path(node1, node2, tree1, tree2, q_start):
    """
    For two RRtrees that have connected to the same configuration q_con, returns a path from start to finish.
    Args:
        - node1: node of q_con in tree1
        - node2: node of q_con in tree2
        - tree1: tree for node1
        - tree2: tree for node2
        - q_start: the starting configuration, used to determine which tree is the one starting at the starting node such that the path is given in correct order
    Returns:
        - list containing a path of collision-free configurations from start to goal
    """
    
    # find out which tree contains starting node
    if np.array_equal(tree1.root.q, q_start):
        nodeA = node1
        nodeB = node2
    else:
        nodeA = node2
        nodeB = node1

    # go from connection node to root for both trees
    tmp = nodeA
    a_traj = [nodeA.q]
    while tmp.parent != None:
        tmp = tmp.parent
        a_traj.append(tmp.q)
    tmp = nodeB
    b_traj = [nodeB.q]
    while tmp.parent != None:
        tmp = tmp.parent
        b_traj.append(tmp.q)

    return list(reversed(a_traj)) + b_traj
    
def free_fn(collision):
    """
    Wrapper function, returns a function that checks if the path between two configurations is collision-free (basically the same thing as the connect method without any trees involved).
    """
    
    def free(q_start, q_end, epsilon):
        tmp = q_start
        while True:
            dist = np.linalg.norm(q_end - tmp)
            if epsilon > dist:
                return True
            else:
                tmp = tmp + (epsilon/dist) * (q_end - tmp)
                if collision(tmp):
                    return False       
    
    return free

def smooth_path(path, epsilon, free):
    """
    Takes a working path and smoothes it by checking if intermediate steps can be skipped.
    Greedy and thus pretty slow.
    """
    path_smooth = [path[0]]
    cur = 0

    while cur < len(path) - 1:
        for idx, pose in reversed(list(enumerate(path[cur+1:]))):
            if free(path[cur], pose, epsilon):
                path_smooth.append(pose)
                cur = idx+cur
                break
        cur += 1

    return path_smooth

def bi_rrt(q_start, q_final, goal_bias, robot, joints, objs, max_steps, epsilon, ee, pos_start, pos_goal, thresh, force_swap=100, smooth=True, save_all_paths=False):
    """
    Performs RRT algorithm with two trees that are swapped out if certain conditions apply.
    Params:
        - q_start: list/array of starting configuration for robot
        - q_final: list/array of goal configuration for robot
        - goal_bias: number between 0 and 1, determines the probability that the random sample of the algorithm is replaced with the goal configuration
        - robot: Pybullet id of the robot which the algorithm is running for
        - joints: list of Pybullet joint ids of the above robot
        - objs: Pybullet object ids for which the collision check is performed
        - max_steps: cutoff for iterations of the algorithm before no solution will be returned
        - epsilon: config space step size for collision check
        - ee: Pybullet link id of end effector of the robot, needed for collision
        - pos_start: Cartesian position of q_start, needed for collision
        - pos_goal: Cartesian position of q_goal, needed for collision
        - thresh: radius around pos_start/goal in which collision will be ignored
        - force_swap: number of times one of the two trees can stay swapped in before a swap will be forced
        - smooth: determines if the result path will be smoothed, takes some time, potentially even more than the initial search
        - give_all_paths: determines if a list of all paths possible within both tress (even the ones that don't containt start or goal) is saved to a file
    """
    # start time for running time output down below
    start = time()

    # init the two trees
    treeA = Tree(q_start)
    treeB = Tree(q_final)

    # init the two max radii for biased sampling
    rA = np.linalg.norm(np.array(q_final) - np.array(q_start))
    rB = rA

    # init control factor to preferably connect to boundary nodes
    controlA, controlB = 1, 1

    # init the variables that count the ratio of failed connect attempts, used for determining swap condition below
    collisionA, triesA, ratioA = 0., 0., 1
    collisionB, triesB, ratioB = 0., 0., 1

    # free run, number of times after a force swap of trees that the swap condition will be ignored
    free_runs = 50
    free_run = 0
    force_swap_active = False

    # variables for anchorpoint sampling
    anchorpoint = np.zeros(len(q_start))
    scale = 1

    # stop Pybullet rendering to save performance
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
    # get collision function for our robot and obstacle
    collision = collision_fn(robot, joints, objs, ee, pos_start, pos_goal, thresh)

    # get the free space function for smoothing if needed
    if smooth:
        free = free_fn(collision)
    
    # prepare sample and connect function for our robot
    lower = []
    upper = []
    for joint in joints:
        lower.append(pyb.getJointInfo(robot, joint)[8])
        upper.append(pyb.getJointInfo(robot, joint)[9])
    sample = sample_fn(joints, lower, upper, collision)
    connect = connect_fn(collision, epsilon)

    # main algorithm
    for i in range(max_steps):

        # sampling:
        random = np.random.random()
        if random > goal_bias:
            # get random configuration
            q_rand = sample() * scale + anchorpoint
            if np.linalg.norm(q_rand - treeB.root.q) > rA:
                # resample if it's not in the allowable radius
                continue 
        else:
            # try goal node
            q_rand = treeB.root.q

        # try to connect sampled node to tree
        # status is an int giving information about the process, reachedNode is the tree node if connected or None if unsuccesful
        reached_nodeA, status = connect(treeA, q_rand, controlA)
        # increment connect tries
        triesA += 1.

        # anchorpoint sampling handling
        if status == 1 or status == 2:
            # connection unsuccesful, reduce sampling radius and use last node before collision as anchorpoint
            scale = max(0.05, scale - 0.05)
            anchorpoint = reached_nodeA.q
        if status == 0:
            # connection succesful, reset anchorpoint and sampling radius
            scale = 1
            anchorpoint = np.zeros(len(q_start))

        
        if status == 2:  # case: hit an obstacle in one epsilon after nearest neighbor
            # increment direct collisions
            collisionA += 1.
            # make sampling radius larger
            rA = rA + epsilon * 50
            # (temporarily) allow nodes from inside the tree to be connected to
            controlA = 3
        else:  # case: succesful connection or hit an obstacle after progressing at least one epsilon towards sampled config
            # update search radius
            rA = rA = np.linalg.norm(reached_nodeA.q - treeB.root.q)
            # reset allowed connects to only boundary nodes
            controlA = 1
            # try to reach the new node from the other tree
            reached_nodeB, status = connect(treeB, reached_nodeA.q, controlB)
            triesB += 1.
            if status == 0:  # could connect new node to other tree
                # get path from start to finish
                sol = bi_path(reached_nodeA, reached_nodeB, treeA, treeB, q_start)
                solution_found = time()
                print("Solution found")
                print("Time for solution: "+str(solution_found-start))
                if save_all_paths:
                    treeA_paths = treeA.all_possible_paths()
                    treeA_paths = np.array(treeA_paths)
                    treeB_paths = treeB.all_possible_paths()
                    treeB_paths = np.array(treeB_paths)
                    # generate random file name for this run of RRT planning
                    filename = str(uuid.uuid4())
                    with open("./scripts/saved_trees/"+filename+"_0.npy", "wb") as outfile:
                        np.save(outfile, treeA_paths)
                    with open("./scripts/saved_trees/"+filename+"_1.npy", "wb") as outfile:
                        np.save(outfile, treeB_paths)   
                if not smooth:
                    ret = sol
                else:
                    ret = smooth_path(sol, epsilon, free)
                    print("Time for smoothing: "+str(time()-solution_found))                   
                pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
                return ret
            elif status == 2:
                collisionB += 1

        # tree swap
        # calculate ratio of collisions to succesful connects
        ratioA = collisionA / triesA
        ratioB = 1 if triesB == 0 else collisionB / triesB
        if force_swap_active:
            free_run += 1
        # the tree with the higher ratio stays in, such that more samples to go the tree that is near obstacles
        if (ratioB > ratioA and free_run > free_runs) or i%force_swap==0:
            free_run = 0
            force_swap_active = False
            if i%force_swap==0:
                force_swap_active = True
            treeA, treeB = treeB, treeA
            collisionA, collisionB = collisionB, collisionA
            triesA, triesB = triesB, triesA
            ratioA, ratioB = ratioB, ratioA
            rA, rB = rB, rA
            controlA, controlB = controlB, controlA
            anchorpoint = np.zeros(len(q_start))
            scale = 1
        if i % 250 == 0:
            print(str(i+1) + " Steps")
            print("Current Tree has "+str(len(treeA.nodes))+" nodes.")
            print("Failure to connect rate: "+str(ratioA))
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
    
    return None

def path(tree, end_node):
    """
    Path algorithm for single tree RRT.
    """
    path = [end_node.q]
    tmp = end_node
    while tmp.parent is not None:
        tmp = tmp.parent
        path.append(tmp.q)

    return list(reversed(path))

def rrt(q_start, q_final, goal_bias, robot, joints, obj, max_steps, epsilon, smooth=False):
    """
    Single tree RRT, doesn't work nearly as well as the Bi-RRT above in this context for some reason.
    """
    start = time()
    tree = Tree(q_start)

    R = np.linalg.norm(np.array(q_final) - np.array(q_start))
    control = 1

    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
    collision = collision_fn(robot, joints, obj)
    if smooth:
        free = free_fn(collision)
    lower = []
    upper = []
    for joint in joints:
        lower.append(pyb.getJointInfo(robot, joint)[8])
        upper.append(pyb.getJointInfo(robot, joint)[9])
    sample = sample_fn(joints, lower, upper, collision)
    connect = connect_fn(collision, epsilon)

    for i in range(max_steps):
        random = np.random.random()
        if random > goal_bias:
            q_rand = sample()
            if np.linalg.norm(q_rand - q_final) > R:
                continue 
        else:
            q_rand = q_final

        reached_node, status = connect(tree, q_rand, control)
        if status == 2:  # hit an obstacle an epsilon after nearest neighbor
            R = R + epsilon * 50
            control = 3
        else:
            R = np.linalg.norm(reached_node.q - q_final)
            control = 1
            if R == 0:
                sol = path(tree, reached_node)
                solution_found = time()
                print("Solution found")
                print("Time for solution: "+str(solution_found-start))
                if smooth:
                    smooth = smooth_path(sol, epsilon, free)
                    print("Time for smoothing: "+str(time()-solution_found))
                    return smooth


        if i % 10 == 0:
            print(str(i+1) + " Steps")
            print("Current Tree has "+str(len(tree.nodes))+" nodes.")
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
    
    return None

def interpolate_path(path, global_scale=1.0):
    """
    Method that takes a path of configurations and interpolates between its elements,
    resulting in the same trajectory but with more intermediate steps. The number of steps
    between two configurations depends on their Euclidean distance in config space.
    """
    ret = []
    scale_factors = []
    for idx in range(len(path)-1):
        dist = np.linalg.norm(path[idx] - path[idx+1])
        scale_factors.append(dist)
    max_scale = max(scale_factors)
    for idx in range(len(scale_factors)):
        scale_factors[idx] = scale_factors[idx]/max_scale if max_scale != 0 else 1
    for idx in range(len(path)-1):
        values = np.vstack([path[idx], path[idx+1]])
        interpolator = interp1d([0,1], values, axis=0)
        interp_values = np.linspace(0,1, int(50*scale_factors[idx]*global_scale))
        for value in interp_values:
            ret.append(interpolator(value))

    return ret