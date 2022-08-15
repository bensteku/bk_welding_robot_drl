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
    start = time()
    treeA = Tree(q_start)
    treeB = Tree(q_final)

    rA = np.linalg.norm(np.array(q_final) - np.array(q_start))
    rB = rA
    controlA, controlB = 1, 1

    collisionA, triesA, ratioA = 0., 0., 1
    collisionB, triesB, ratioB = 0., 0., 1

    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
    collision = collision_fn(robot, joints, obj)
    if collision(q_final) or collision(q_start):
        raise ValueError("Either starting or goal configuration is in collision in work space!")
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
            if np.linalg.norm(q_rand - treeB.root.q) > rA:
                continue 
        else:
            q_rand = treeB.root.q

        reached_nodeA, status = connect(treeA, q_rand, controlA)
        triesA += 1.
        if status == 2:  # hit an obstacle an epsilon after nearest neighbor
            collisionA += 1.
            rA = rA + epsilon * 50
            controlA = 3
        else:  # expand at least one epsilon into the direction of q_rand, but hit an obstacle before reaching it
            rA = rA = np.linalg.norm(reached_nodeA.q - treeB.root.q)
            controlA = 1
            reached_nodeB, status = connect(treeB, reached_nodeA.q, controlB)
            triesB += 1.
            if status == 0:  # could connect new node to other tree
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
        ratioA = collisionA / triesA
        ratioB = 1 if triesB == 0 else collisionB / triesB
        if ratioB > ratioA or i%force_swap==0:
            treeA, treeB = treeB, treeA
            collisionA, collisionB = collisionB, collisionA
            triesA, triesB = triesB, triesA
            ratioA, ratioB = ratioB, ratioA
            rA, rB = rB, rA
            controlA, controlB = controlB, controlA
        if i % 10 == 0:
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