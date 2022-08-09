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
    return np.array([im[0], im[1], im[2], re])

def quaternion_invert(quat):
    q = pyq.Quaternion(quat[3], quat[0], quat[1], quat[2])
    inv = q.inverse
    im = inv.imaginary
    re = inv.real
    return np.array([im[0], im[1], im[2], re])

def quaternion_diff(quat1, quat2):
    # return = quat2*inv(quat1) -> return*quat1=quat2, in words: which quaternion times quat1 results in quat2?
    return np.array(quaternion_multiply(quat2, quaternion_invert(quat1)))

def quaternion_norm(quat):
    return np.linalg.norm(quat)

def quaternion_normalize(quat):
    return quat / quaternion_norm(quat)
    
def quaternion_interpolate(quat1, quat2, n):
    q1 = pyq.Quaternion(quat1[3], quat1[0], quat1[1], quat1[2])
    q2 = pyq.Quaternion(quat2[3], quat2[0], quat2[1], quat2[2])

    res = pyq.Quaternion.intermediates(q1, q2, n, include_endpoints=True)
    res = list(res)
    res = [np.array([q[1], q[2], q[3], q[0]]) for q in res]
    return res

def pos_interpolate(pos1, pos2, speed):

    delta = pos2 - pos1
    delta_norm = np.linalg.norm(delta)

    if delta_norm <= speed:
        return [pos2]
    else:
        segments = delta_norm / speed
        delta = delta / delta_norm
        delta = delta * speed
        res = []
        cur = pos1
        counter = 1
        while counter <= int(segments):
            cur = cur + delta
            res.append(cur)
            counter += 1
        # add missing piece to final position
        missing_mul = segments - int(segments)
        cur = cur + missing_mul * delta
        res.append(cur)
        return res

def quaternion_similarity(quat1, quat2):
    """
    Measure of similarity between two quaternions via the angle distance between the two
    """
    return 1 - np.arccos(2 * np.dot(quat1, quat2)**2 - 1)/np.pi

def quaternion_apx_eq(quat1, quat2, thresh=5e-2):
    return  quaternion_similarity(quat1, quat2) < thresh

def exp_decay(x, max, zero_crossing):
    if x >= zero_crossing:
        return 0
    half = max/2.0
    three_halfs = max * 1.5
    return three_halfs * np.exp((np.log(1./3.)/(zero_crossing))*x) - half

def rotate_vec(quat, vec):
    work_vec = np.array([vec[0], vec[1], vec[2], 0])
    return quaternion_multiply(quaternion_multiply(quat, work_vec), quaternion_invert(quat))[:3]

def cosine_similarity(vec1, vec2):
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / norm
    
