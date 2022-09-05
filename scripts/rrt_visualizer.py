import numpy as np
import pybullet as pyb
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util.planner import interpolate_path
import matplotlib.pyplot as plt
import open3d as o3d

folder = "./scripts/saved_trees/"
filename = "2acee10b-563b-4484-bacf-412ea7c6de3e"
mesh_folder = "./assets/objects/"
meshfile = "201910204483_R1.obj"

# TODO: add base position such that the open3d cartesian positions are correct, fix point cloud rendering in open3d

# pybullet stuff

client = pyb.connect(pyb.DIRECT)

pyb.setTimeStep(1. / 240)
pyb.setAdditionalSearchPath("./assets/")

pyb.resetSimulation(pyb.RESET_USE_DEFORMABLE_WORLD)
pyb.setGravity(0, 0, -9.8)
pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)


plane_id = pyb.loadURDF("workspace/plane.urdf", basePosition=[0, 0, -0.001])
robot = pyb.loadURDF("kr16/kr16_tand_gerad.urdf", useFixedBase=True, basePosition=[0, 0, 2], baseOrientation=pyb.getQuaternionFromEuler([np.pi, 0., 0.]))

joints = [pyb.getJointInfo(robot, i) for i in range(pyb.getNumJoints(robot))]
joints = [j[0] for j in joints if j[2] == pyb.JOINT_REVOLUTE]

def get_joints():
    return np.array([pyb.getJointState(robot, i)[0] for i in joints])

def set_joints(q):
    for i in range(len(q)):
        pyb.resetJointState(robot, joints[i], q[i])

def get_pos():
    return np.array(pyb.getLinkState(
                            bodyUniqueId=robot,
                            linkIndex=7,
                            computeForwardKinematics=True
            )[0])

# read the files containing the tree paths

paths_0 = np.load(folder+filename+"_0.npy", allow_pickle=True)
paths_1 = np.load(folder+filename+"_1.npy", allow_pickle=True)

# get the position of the roots of the trees (aka the start and goal)

root_0_config = paths_0[0][0]
root_1_config = paths_1[0][0]

set_joints(root_0_config)
root_0_xyz = get_pos()
set_joints(root_1_config)
root_1_xyz = get_pos()

# interpolate a bit sucht that the drawings end up lookin better
interpolated_paths_0 = []
for path in paths_0:
    interpolated_paths_0.append(interpolate_path(path, 5))
#print(interpolated_paths_0)

interpolated_paths_1 = []
for path in paths_1:
    interpolated_paths_1.append(interpolate_path(path, 5))

# run forward kinematics to get the actual trajectories in cartesian space for both trees

cartesian_paths_x_0 = []
cartesian_paths_y_0 = []
cartesian_paths_z_0 = []
for path in interpolated_paths_0:
    cart_path_x = []
    cart_path_y = []
    cart_path_z = []
    for config in path:
        set_joints(config)
        x, y, z = get_pos()
        cart_path_x.append(x)
        cart_path_y.append(y)
        cart_path_z.append(z)
    cartesian_paths_x_0.append(cart_path_x)
    cartesian_paths_y_0.append(cart_path_y)
    cartesian_paths_z_0.append(cart_path_z)

cartesian_paths_x_1 = []
cartesian_paths_y_1 = []
cartesian_paths_z_1 = []
for path in interpolated_paths_1:
    cart_path_x = []
    cart_path_y = []
    cart_path_z = []
    for config in path:
        set_joints(config)
        x, y, z = get_pos()
        cart_path_x.append(x)
        cart_path_y.append(y)
        cart_path_z.append(z)
    cartesian_paths_x_1.append(cart_path_x)
    cartesian_paths_y_1.append(cart_path_y)
    cartesian_paths_z_1.append(cart_path_z)
# open3d tests
"""
cartesian_paths_0 = []
for path in interpolated_paths_0:
    cart_path = []
    for config in path:
        set_joints(config)
        xyz = get_pos()
        cart_path.append(xyz)
    cartesian_paths_0.append(cart_path)
cartesian_paths_0 = np.array(cartesian_paths_0)

cartesian_paths_1 = []
for path in interpolated_paths_1:
    cart_path = []
    for config in path:
        set_joints(config)
        xyz = get_pos()
        cart_path.append(xyz)
    cartesian_paths_1.append(cart_path)
cartesian_paths_1 = np.array(cartesian_paths_1)
"""
# plot the trajectories

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

for idx in range(len(cartesian_paths_z_0)):
    ax.scatter(cartesian_paths_x_0[idx], cartesian_paths_y_0[idx], cartesian_paths_z_0[idx], color="red")
for idx in range(len(cartesian_paths_z_1)):
    ax.scatter(cartesian_paths_x_1[idx], cartesian_paths_y_1[idx], cartesian_paths_z_1[idx], color="blue")

# emphasize the roots

ax.scatter([root_0_xyz[0]], [root_0_xyz[1]], [root_0_xyz[2]], color="red", s=75)
ax.scatter([root_1_xyz[0]], [root_1_xyz[1]], [root_1_xyz[2]], color="blue", s=75)

plt.show()
#open3d tests
"""
elements = []
mesh_model = o3d.io.read_triangle_mesh(mesh_folder+meshfile)
pcd0 = o3d.geometry.PointCloud()
pcd0.points = o3d.utility.Vector3dVector(np.squeeze(cartesian_paths_0))
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(np.squeeze(cartesian_paths_1))
elements.append(mesh_model)
elements.append(pcd0)
elements.append(pcd1)
o3d.visualization.draw_geometries(elements)
"""
