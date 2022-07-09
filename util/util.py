import numpy as np

def matrix_to_quaternion(matrix):
    """Utility method to get quaternion from a 3x3 rotation matrix
    
    Args:
        matrix: List containing the matrix elements in row-column-order (i.e. m00,m01,m02,m10,.. etc.)
        
    Returns:
        List with 4 elements [x,y,z,w] representing the quaternion built from the input matrix
    """

    if type(matrix) == list:
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix
    else:
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.flatten()
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
    return [qx, qy, qz, qw]

def quaternion_to_euler_angle(quat):
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