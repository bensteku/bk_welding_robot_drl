from argparse import ArgumentError
import numpy as np
import pybullet as pyb
from scipy.interpolate import interp1d
from time import time

class Node:
    
    def __init__(self, q, parent):
        self.q = np.array(q)
        self.parent = parent
        self.children = []
        self.delta = 0

    def add_child(self, node):
        pass

class Tree:
    
    def __init__(self, q):
        self.root = Node(q, None)
        self.nodes = [self.root]

    def add_node(self, q, parent = None):
        if not parent:
            closest = self.nearest_neighbor(q)
        else:
            closest = parent
        new = Node(q, closest)
        closest.children.append(new)
        self.nodes.append(new)
        tmp = new.parent
        while tmp.parent is not None:
            tmp.delta += 1
            tmp = tmp.parent
        return new

    def nearest_neighbor(self, q, control):
        min_dist = np.Inf
        closest = None
        for node in self.nodes:
            dist = np.linalg.norm(np.array(q) - np.array(node.q))
            if dist < min_dist and node.delta < control:
                min_dist = dist
                closest = node
        return closest

def collision_fn(robot, joints, obj):
    def collision(q):
        currj = [pyb.getJointState(robot, i)[0] for i in joints]
        for joint, val in zip(joints, q):
            pyb.resetJointState(robot, joint, val)    
        #pyb.addUserDebugPoints([pyb.getLinkState(robot, 7, computeForwardKinematics=True)[0]],[[0,0,1]], 3)
        pyb.performCollisionDetection()  # perform just the collision detection part of the PyBullet engine
        col_env = True if pyb.getContactPoints(robot, obj) or pyb.getContactPoints(robot, 1) else False  # 1 will always be the ground plane
        for joint, val in zip(joints, currj):
            pyb.resetJointState(robot, joint, val)
        #col_self = True if pyb.getContactPoints(robot, robot, 1, 7) else False
        
        return col_env #and col_self
    return collision

def sample_fn(joints, lower, upper, collision):
    def sample():
        sample = np.random.uniform(low=np.array(lower), high=np.array(upper), size=len(joints))
        while collision(sample):
            sample = np.random.uniform(low=np.array(lower), high=np.array(upper), size=len(joints))
        return sample
    return sample

def connect_fn(collision, epsilon = 1e-3):
    def connect(tree, q, control):
        node_near = tree.nearest_neighbor(q, control)
        q_cur = node_near.q
        q_old = q_cur
        while True:   
            dist = np.linalg.norm(q - q_cur)
            if epsilon > dist:
                q_cur = q
            else:
                q_old = q_cur
                q_cur = q_cur + (epsilon/dist) * (q - q_cur)
            if np.array_equal(q_cur,q):
                return tree.add_node(q, node_near), 0
            col = collision(q_cur)
            if col and np.array_equal(q_old, node_near.q):
                return None, 2  
            elif col and not np.array_equal(q_old, node_near.q):
                return tree.add_node(q_old, node_near), 1
    return connect

def bi_path(node1, node2, tree1, tree2, q_start):
    
    if np.array_equal(tree1.root.q, q_start):
        nodeA = node1
        nodeB = node2
    else:
        nodeA = node2
        nodeB = node1

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

def bi_rrt(q_start, q_final, goal_bias, robot, joints, obj, max_steps, epsilon, force_swap=100, smooth=True):
    """
    Performs RRT algorithm with two trees that are swapped out if certain conditions apply.
    Params:
        - q_start: list/array of starting configuration for robot
        - q_final: list/array of goal configuration for robot
        - goal_bias: number between 0 and 1, determines the probability that the random sample of the algorithm is replaced with the goal configuration
        - robot: Pybullet of the robot which the algorithm is running for
        - joints: list of Pybullet joint ids of the above robot
        - obj: Pybullet object id for which the collision check is performed
        - max_steps: cutoff for iterations of the algorithm before no solution will be returned
        - epsilon: config space step size for collision check
        - force_swap: number of times one of the two trees can stay swapped in before a swap will be forced
        - smooth: determines if the result path will be smoothed, takes some time
    """
    # start time for time output down below
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

    # stop Pybullet rendering to save performance
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
    # get collision function for our robot and obstacle
    collision = collision_fn(robot, joints, obj)

    # check if either start or finish is in collision
    if collision(q_final):
        raise ValueError("Goal configuration is in collision in work space!")
    elif collision(q_start):
        raise ValueError("Starting configuration is in collision in work space!")
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
            q_rand = sample()
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
                if not smooth:
                    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
                    return sol
                else:
                    smoothed = smooth_path(sol, epsilon, free)
                    print("Time for smoothing: "+str(time()-solution_found))
                    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
                    return smoothed
            elif status == 2:
                collisionB += 1

        # tree swap
        # calculate ratio of collisions to succesful connects
        ratioA = collisionA / triesA
        ratioB = 1 if triesB == 0 else collisionB / triesB
        # the tree with the higher ratio stays in, such that more samples to go the tree that is near obstacles
        if ratioB > ratioA or i%force_swap==0:
            treeA, treeB = treeB, treeA
            collisionA, collisionB = collisionB, collisionA
            triesA, triesB = triesB, triesA
            ratioA, ratioB = ratioB, ratioA
            rA, rB = rB, rA
            controlA, controlB = controlB, controlA
        if i % 250 == 0:
            print(str(i+1) + " Steps")
            print("Current Tree has "+str(len(treeA.nodes))+" nodes.")
            print("Failure to connect rate: "+str(ratioA))
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
    
    return None

def path(tree, end_node):
    path = [end_node.q]
    tmp = end_node
    while tmp.parent is not None:
        tmp = tmp.parent
        path.append(tmp.q)

    return list(reversed(path))

def rrt(q_start, q_final, goal_bias, robot, joints, obj, max_steps, epsilon, smooth=False):
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

def interpolate_path(path):
    ret = []
    scale_factors = []
    for idx in range(len(path)-1):
        dist = np.linalg.norm(path[idx] - path[idx+1])
        scale_factors.append(dist)
    max_scale = max(scale_factors)
    for idx in range(len(scale_factors)):
        scale_factors[idx] = scale_factors[idx]/max_scale
    for idx in range(len(path)-1):
        values = np.vstack([path[idx], path[idx+1]])
        interpolator = interp1d([0,1], values, axis=0)
        interp_values = np.linspace(0,1, int(50*scale_factors[idx]))
        for value in interp_values:
            ret.append(interpolator(value))

    return ret