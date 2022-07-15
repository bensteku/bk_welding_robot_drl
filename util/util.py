import numpy as np
import pybullet as pyb
import pyquaternion as pyq

def matrix_to_quaternion(matrix):
    """Utility method to get quaternion from a 3x3 rotation matrix
    
    Args:
        matrix: List containing the matrix elements in row-column-order (i.e. m00,m01,m02,m10,.. etc.)
        
    Returns:
        List with 4 elements [x,y,z,w] representing the quaternion built from the input matrix
    """

    if type(matrix) == list:
        m00, m10, m20, m01, m11, m21, m02, m12, m22 = matrix
    else:
        m00, m10, m20, m01, m11, m21, m02, m12, m22 = matrix.flatten()
    if (m22 < 0): 
        if (m00 >m11):
            t = 1 + m00 -m11 -m22
            q = [t, m01+m10, m20+m02, m12-m21]
        else: 
            t = 1 -m00 + m11 -m22
            q = [m01+m10, t, m12+m21, m20-m02]
    else:
        if (m00 < -m11):
            t = 1 -m00 -m11 + m22
            q = [m20+m02, m12+m21, t, m01-m10]
        else:
            t = 1 + m00 + m11 + m22
            q = [m12-m21, m20-m02, m01-m10, t]
    q = np.array([ele * 0.5 / np.sqrt(t) for ele in q])
    return q

def rpy_to_quaternion(rpy):
    """
    Utility method to get quaternion from roll-pitch-yaw angles.

    Args:
        rpy: List containing three floats representing rpy angles in rad.
    
    Returns:
        List with 4 elements [x,y,z,w] representing the quaternion built from the input angles 
    """

    roll, pitch, yaw = rpy
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return np.array([qx, qy, qz, qw])

def quaternion_to_rpy(quat):
    x, y, z, w = quat
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return X, Y, Z 

def quaternion_multiply(quat1, quat2):
    q1 = pyq.Quaternion(quat1[3], quat1[0], quat1[1], quat1[2])
    q2 = pyq.Quaternion(quat2[3], quat2[0], quat2[1], quat2[2])

    res = q1 * q2
    im = res.imaginary
    re = res.real
    return [im[0], im[1], im[2], re]

def quaternion_invert(quat):
    q = pyq.Quaternion(quat[3], quat[0], quat[1], quat[2])
    inv = q.inverse
    im = inv.imaginary
    re = inv.real
    return [im[0], im[1], im[2], re]