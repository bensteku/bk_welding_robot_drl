import numpy as np
import pybullet as pyb
from scipy.interpolate import interp1d

class Node:
    
    def __init__(self, q, parent):
        self.q = np.array(q)
        self.parent = parent
        self.children = []

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
        return new

    def nearest_neighbor(self, q):
        min_dist = np.Inf
        closest = None
        for node in self.nodes:
            dist = np.linalg.norm(np.array(q) - np.array(node.q))
            if dist < min_dist:
                min_dist = dist
                closest = node
        return closest

def collision_fn(robot, joints, obj):
    def collision(q):
        currj = [pyb.getJointState(robot, i)[0] for i in joints]
        for joint, val in zip(joints, q):
            pyb.resetJointState(robot, joint, val)
        pyb.performCollisionDetection()  # perform just the collision detection part of the PyBullet engine
        col = True if pyb.getContactPoints(robot, obj) or pyb.getContactPoints(robot, 1) else False
        for joint, val in zip(joints, currj):
            pyb.resetJointState(robot, joint, val)
        return col
    return collision

def sample_fn(joints, lower, upper, collision):
    def sample():
        sample = np.random.uniform(low=np.array(lower), high=np.array(upper), size=len(joints))
        while collision(sample):
            sample = np.random.uniform(low=np.array(lower), high=np.array(upper), size=len(joints))
        return sample
    return sample

def connect_fn(collision):
    epsilon = 5e-3
    def connect(tree, q):
        node_near = tree.nearest_neighbor(q)
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
                return tree.add_node(q, node_near)
            col = collision(q_cur)
            if col and np.array_equal(q_old, node_near.q):
                return None  
            elif col and not np.array_equal(q_old, node_near.q):
                return tree.add_node(q_old, node_near)
    return connect

def path(node1, node2, tree1, tree2, q_start):
    
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
    

def bi_rrt(q_start, q_final, goal_bias, robot, joints, obj, max_steps):
    treeA = Tree(q_start)
    treeB = Tree(q_final)

    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
    collision = collision_fn(robot, joints, obj)
    lower = []
    upper = []
    for joint in joints:
        lower.append(pyb.getJointInfo(robot, joint)[8])
        upper.append(pyb.getJointInfo(robot, joint)[9])
    sample = sample_fn(joints, lower, upper, collision)
    connect = connect_fn(collision)

    for i in range(max_steps):
        random = np.random.random()
        if random > goal_bias:
            q_rand = sample()
        else:
            q_rand = treeB.root.q

        reached_nodeA = connect(treeA, q_rand)
        if reached_nodeA is not None:
            reached_nodeB = connect(treeB, reached_nodeA.q)
            if reached_nodeB is not None and np.array_equal(reached_nodeA.q, reached_nodeB.q):
                pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
                return path(reached_nodeA, reached_nodeB, treeA, treeB, q_start)

        treeA, treeB = treeB, treeA
        if i % 10 == 0:
            print(str(i+1) + " Steps")
            print(len(treeA.nodes))
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
    
    return None

def interpolate_path(path):
    ret = [path[0]]
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
        interp_values = np.linspace(0,1, int(100*scale_factors[idx]))
        for value in interp_values:
            ret.append(interpolator(value))

    return ret