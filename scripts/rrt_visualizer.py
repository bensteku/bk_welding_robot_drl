import numpy as np
import pybullet as pyb
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util.planner import interpolate_path
import matplotlib.pyplot as plt
import open3d as o3d
import meshio
import plotly
import plotly.graph_objects as go

folder = "./scripts/saved_trees/"
filename = "e870f74e-9a55-4629-a322-46bd956b0da6_0-68_1-22"
mesh_folder = "./assets/objects/"
meshfile = "201910204483_R1.obj"

open3d_or_matplotlib = 2

# pybullet stuff

base_pos_0 = float(filename.split("_")[1].replace("-","."))
base_pos_1 = float(filename.split("_")[2].replace("-","."))

client = pyb.connect(pyb.DIRECT)

pyb.setTimeStep(1. / 240)
pyb.setAdditionalSearchPath("./assets/")

pyb.resetSimulation(pyb.RESET_USE_DEFORMABLE_WORLD)
pyb.setGravity(0, 0, -9.8)
pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)


plane_id = pyb.loadURDF("workspace/plane.urdf", basePosition=[0, 0, -0.001])
robot = pyb.loadURDF("kr16/kr16_tand_gerad.urdf", useFixedBase=True, basePosition=[base_pos_0, base_pos_1, 2], baseOrientation=pyb.getQuaternionFromEuler([np.pi, 0., 0.]))

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
if open3d_or_matplotlib == 0 or open3d_or_matplotlib == 2:
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
else:
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

# plot the trajectories
if not open3d_or_matplotlib:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for idx in range(len(cartesian_paths_z_0)):
        ax.scatter(cartesian_paths_x_0[idx], cartesian_paths_y_0[idx], cartesian_paths_z_0[idx], color="red", s=2)
    for idx in range(len(cartesian_paths_z_1)):
        ax.scatter(cartesian_paths_x_1[idx], cartesian_paths_y_1[idx], cartesian_paths_z_1[idx], color="blue", s=2)

    # emphasize the roots

    ax.scatter([root_0_xyz[0]], [root_0_xyz[1]], [root_0_xyz[2]], color="red", s=100, marker='x')
    ax.scatter([root_1_xyz[0]], [root_1_xyz[1]], [root_1_xyz[2]], color="blue", s=100, marker='x')

    plt.show()
elif open3d_or_matplotlib == 1:
    # create a dataset of points and indices such that open3d can recognize these as lines
    points = []
    lines = []
    colors = []
    for path in cartesian_paths_0:
        points.append(path[0])
        for idx, point in enumerate(path[1:]):
            points.append(point)
            lines.append([idx, idx+1])
            colors.append([1, 0, 0])
    for path in cartesian_paths_1:
        points.append(path[0])
        for idx, point in enumerate(path[1:]):
            points.append(point)
            lines.append([idx, idx+1])
            colors.append([0, 0, 1])

    elements = []
    mesh_model = o3d.io.read_triangle_mesh(mesh_folder+meshfile)
    mesh_model.scale(scale=0.0005, center=[0, 0, 0])
    mesh_model.compute_vertex_normals()
    #pcd0 = o3d.geometry.PointCloud()
    #pcd0.points = o3d.utility.Vector3dVector(np.squeeze(cartesian_paths_0))
    #pcd1 = o3d.geometry.PointCloud()
    #pcd1.points = o3d.utility.Vector3dVector(np.squeeze(cartesian_paths_1))
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    elements.append(mesh_model)
    elements.append(line_set)
    o3d.visualization.draw_geometries(elements)
else:
    mesh_model = meshio.read(mesh_folder+meshfile)
    vertices = mesh_model.points * 0.0005
    print(mesh_model.cells_dict)
    triangles = mesh_model.cells_dict['triangle']

    x, y, z = vertices.T
    I, J, K = triangles.T

    pl_mygrey=[0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']
                           
    pl_mesh = go.Mesh3d(x=x,
                        y=y,
                        z=z,
                        colorscale=pl_mygrey, 
                        intensity= z,
                        flatshading=True,
                        i=I,
                        j=J,
                        k=K,
                        name='welding mesh',
                        showscale=False
                        )

    pl_mesh.update(cmin=-7,# atrick to get a nice plot (z.min()=-3.31909)
               lighting=dict(ambient=0.18,
                             diffuse=1,
                             fresnel=0.1,
                             specular=1,
                             roughness=0.05,
                             facenormalsepsilon=1e-15,
                             vertexnormalsepsilon=1e-15),
               lightposition=dict(x=100,
                                  y=200,
                                  z=0
                                 )
                      )
    
    lines_0 = []
    for idx in range(len(cartesian_paths_x_0)):
        lines_0.append(
            go.Scatter3d(
                x = cartesian_paths_x_0[idx],
                y = cartesian_paths_y_0[idx],
                z = cartesian_paths_z_0[idx],
                mode = 'lines',
                name = '',
                line=dict(color= 'rgb(255,0,0)', width=1)
            )
        )

    lines_1 = []
    for idx in range(len(cartesian_paths_x_1)):
        lines_1.append(
            go.Scatter3d(
                x = cartesian_paths_x_1[idx],
                y = cartesian_paths_y_1[idx],
                z = cartesian_paths_z_1[idx],
                mode = 'lines',
                name = '',
                line=dict(color= 'rgb(0,255,0)', width=1)
            )
        )

    layout = go.Layout(
         title="welding mesh",
         font=dict(size=16, color='white'),
         width=700,
         height=700,
         scene_xaxis_visible=False,
         scene_yaxis_visible=False,
         scene_zaxis_visible=False,
         paper_bgcolor='rgb(50,50,50)',
         scene=dict(
                 aspectmode='data'
         )
        )
    
    fig = go.Figure(data=[pl_mesh] + lines_0 + lines_1, layout=layout)

    fig.show()