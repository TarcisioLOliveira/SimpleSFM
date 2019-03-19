import numpy as np
import scipy.sparse

def sqrt(x):
    return x**0.5

# http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
def R_to_quaternion(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0: 
        S = sqrt(tr+1.0) * 2; # S=4*qw 
        qw = 0.25 * S;
        qx = (R[2, 1] - R[1, 2]) / S;
        qy = (R[0, 2] - R[2, 0]) / S; 
        qz = (R[1, 0] - R[0, 1]) / S; 
    elif ((R[0, 0] > R[1, 1])&(R[0, 0] > R[2, 2])):
        S = sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; # S=4*qx 
        qw = (R[2, 1] - R[1, 2]) / S;
        qx = 0.25 * S;
        qy = (R[0, 1] + R[1, 0]) / S; 
        qz = (R[0, 2] + R[2, 0]) / S; 
    elif R[1, 1] > R[2, 2]:
        S = sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; # S=4*qy
        qw = (R[0, 2] - R[2, 0]) / S;
        qx = (R[0, 1] + R[1, 0]) / S; 
        qy = 0.25 * S;
        qz = (m12 + m21) / S; 
    else:
        S = sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; # S=4*qz
        qw = (R[1, 0] - R[0, 1]) / S;
        qx = (R[0, 2] + R[2, 0]) / S;
        qy = (R[1, 2] + R[2, 1]) / S;
        qz = 0.25 * S;

    return np.array([qx, qy, qz, qw])

def jacobian_get_camera(K, P, X):
    R = P[:, 0:3]
    t = P[:, 3]
    C = -np.dot(np.linalg.inv(np.dot(K, R)), np.dot(K, P))[:, 3]

    x = np.dot(np.dot(K, R), X-C)
    u = x[0]
    v = x[1]
    w = x[2]

    dx = np.dot(K, R)
    dudC = -dx[0, :]
    dvdC = -dx[1, :]
    dwdC = -dx[2, :]
    dudR = np.array([K[0, 0]*(X-C), K[0, 1]*(X-C), K[0, 2]*(X-C)]).flatten()
    dvdR = np.array([K[1, 0]*(X-C), K[1, 1]*(X-C), K[1, 2]*(X-C)]).flatten()
    dwdR = np.array([K[2, 0]*(X-C), K[2, 1]*(X-C), K[2, 2]*(X-C)]).flatten()

    qx, qy, qz, qw = R_to_quaternion(R)
    dR11dq = np.array([0, -4*qy, -4*qz, 0])
    dR12dq = np.array([2*qy, 2*qx, -2*qw, -2*qz])
    dR13dq = np.array([2*qz, 2*qw, 2*qx, 2*qy])
    dR21dq = np.array([2*qy, 2*qx, 2*qw, 2*qz])
    dR22dq = np.array([-4*qx, 0, -4*qz, 0])
    dR23dq = np.array([-2*qw, 2*qz, 2*qy, 2*qx])
    dR31dq = np.array([2*qz, -2*qw, 2*qx, -2*qy])
    dR32dq = np.array([2*qw, 2*qz, 2*qy, 2*qx])
    dR33dq = np.array([-4*qx, -4*qy, 0, 0])

    dRdq = np.array([
        dR11dq,
        dR12dq,
        dR13dq,
        dR21dq,
        dR22dq,
        dR23dq,
        dR31dq,
        dR32dq,
        dR33dq
    ])

    dfdC = np.array([
        (w*dudC - u*dwdC)/w**2,
        (w*dvdC - v*dwdC)/w**2
    ])
    dfdR = np.array([
        (w*dudR - u*dwdR)/w**2,
        (w*dvdR - v*dwdR)/w**2
    ])
    dfdq = np.dot(dfdR, dRdq)

    return np.hstack((dfdq, dfdC))


        
def jacobian_get_point(K, P, X):
    R = P[:, 0:3]
    t = P[:, 3]
    C = -np.dot(np.linalg.inv(np.dot(K, R)), np.dot(K, P))[:, 3]

    x = np.dot(np.dot(K, R), X-C)
    u = x[0]
    v = x[1]
    w = x[2]

    dx = np.dot(K, R)
    dudX = dx[0, :]
    dvdX = dx[1, :]
    dwdX = dx[2, :]

    dfdX = np.array([
        (w*dudX - u*dwdX)/w**2,
        (w*dvdX - v*dwdX)/w**2
    ])

    return dfdX

def reproject_point(K, P, X):
    R = P[:, 0:3]
    t = P[:, 3]
    C = -np.dot(np.linalg.inv(np.dot(K, R)), np.dot(K, P))[:, 3]

    x = np.dot(np.dot(K, R), X-C)
    u = x[0]
    v = x[1]
    w = x[2]

    return np.array([u/w, v/w])

def adjust(point_cloud_data, K, poses, n_images):
    n_points = 0
    all_3dpoints = []
    for data in point_cloud_data:
        n_points += len(data['3dpoints'])
        for p in data['3dpoints']:
            all_3dpoints.append(p)

    list_3dpoints = map(np.unique, all_3dpoints)
    J_rows = n_points*2
    J_cols = n_images*7 + n_points*3
    J = scipy.sparse.coo_matrix((J_rows, J_cols), dtype=np.float32)
    J = scipy.sparse.lil_matrix(J)
    e = np.zeros((J_rows, 1))

    X_col_start = n_images*7

    row = 0
    X_num = 0

    for X in list_3dpoints:
        seen_in_cameras = []
        i = 0
        for data in point_cloud_data:
            if X in data['3dpoints']:
                seen_in_cameras.append(i)
            i += 1
        for j in seen_in_cameras:
            #J[row:row+1, j*7:(j+1)*7] = jacobian_get_camera(K, poses[j], X)
            #J[row:row+1, (X_col_start+X_num*3):(X_col_start+(X_num+1)*3)] = jacobian_get_point(K, poses[j], X)
            camera_w= list(range((j*7), ((j+1)*7)))
            point_w = list(range((X_col_start+X_num*3), (X_col_start+(X_num+1)*3)))
            J[np.ix_([row, row+1], camera_w)] = jacobian_get_camera(K, poses[j], X)
            J[np.ix_([row, row+1], point_w)] = jacobian_get_point(K, poses[j], X)
            e_tmp = point_cloud_data[j]['2dpoints'][np.where(point_cloud_data[j]['3dpoints'] == X)[0][0], :] - reproject_point(K, poses[j], X)
            e[row] = e_tmp[0]
            e[row+1] = e_tmp[1]
            row += 2
        X_num += 1

    J = scipy.sparse.csr_matrix(J)
    dx = np.linalg.lstsq(np.dot(J.T, J), np.dot(J.T, e))
    print dx.shape
