import numpy as np
import scipy.sparse
import sys
from scipy.sparse import linalg
from scipy.linalg import interpolative

def rank(A):
    u, s, v = scipy.sparse.linalg.svds(A)
    rank = np.sum(s > 1e-10)

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
        qz = (R[1, 2] + R[2, 1]) / S; 
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
    all_3dpoints = []
    #for data_i in range(len(point_cloud_data)):
        #point_cloud_data[data_i]['3dpoints'] = point_cloud_data[data_i]['3dpoints'].astype(int64)
        #data = point_cloud_data[data_i]

    empty_data = []
        
    for data_i in range(len(point_cloud_data)):
        if len(point_cloud_data[data_i]['3dpoints']) == 0:
            empty_data.append(data_i)
            continue
    
    num_deleted = 0
    for i in empty_data:
        del point_cloud_data[i - num_deleted]
        del poses[i - num_deleted]
        n_images -= 1
        num_deleted += 1

    for data_i in range(len(point_cloud_data)):
        tmp_list = point_cloud_data[data_i]['3dpoints'][np.all(np.isfinite(point_cloud_data[data_i]['3dpoints']), axis=1)]
        all_3dpoints.extend(tmp_list)
        point_cloud_data[data_i]['3dpoints'] = tmp_list

    list_3dpoints = np.unique(all_3dpoints, axis=-1)
    list_colors = []
    J_rows = len(all_3dpoints)*2
    J_ccols = n_images*7
    J_pcols = len(list_3dpoints)*3
    print(J_rows)
    print(J_ccols)
    print(J_pcols)
    J_c = scipy.sparse.dok_matrix((J_rows, J_ccols), dtype=np.float64)
    J_p = scipy.sparse.dok_matrix((J_rows, J_pcols), dtype=np.float64)
    e = np.zeros((J_rows, 1))

    row = 0
    X_num = 0

    print("Creating matrices")
    for X in list_3dpoints:
        j = 0
        found_color = False
        for data in point_cloud_data:
            points = np.asarray(data['3dpoints'] == X).nonzero()
            if len(points[0]) > 0:
                point = points[0][0]
                camera_w = list(range((j*7), ((j+1)*7)))
                point_w = list(range((X_num*3), (X_num+1)*3))
                J_c[np.ix_([row, row+1], camera_w)] = jacobian_get_camera(K, poses[j], X)
                J_p[np.ix_([row, row+1], point_w)] = jacobian_get_point(K, poses[j], X)
                e_tmp = data['2dpoints'][point, :] - reproject_point(K, poses[j], X)
                e[row] = e_tmp[0]
                e[row+1] = e_tmp[1]
                if not found_color:
                    list_colors.append(data['colors'][point, :])
                    found_color = True
                row += 2
            j += 1
        X_num += 1

    print("Beginning matrix operations...")
    J_c = scipy.sparse.csc_matrix(J_c)
    J_p = scipy.sparse.csc_matrix(J_p)
    print("Sparse matrix conversion complete (dok to csr).")
    B = scipy.sparse.dia_matrix(J_c.T.dot(J_c))
    C = scipy.sparse.dia_matrix(J_p.T.dot(J_p))
    E = J_c.T.dot(J_p)
    print("Hessian decomposition complete.")
    g_c = J_c.T.dot(e)
    g_p = J_p.T.dot(e)
    print("Gradients complete.")
    if np.sum(C > 1e-10) == J_pcols:
        print("Optimization 01 possible. Executing...")
        EC_1 = E.dot(scipy.sparse.linalg.inv(C))
        S_C = B - EC_1.dot(E.T)
        v = -g_c + EC_1.dot(g_p)
        p_c = scipy.sparse.linalg.lsqr(S_C, v).todense()
        p_p = scipy.sparse.linalg.inv(C).dot(-g_p - E.T.dot(p_c)).todense()
    elif np.sum(B > 1e-10) == J_pcols:
        print("Optimization 02 possible. Executing...")
        EB_1 = E.T.dot(scipy.sparse.linalg.inv(B))
        S_B = C - EB_1.dot(E)
        v = -g_c + EB_1.dot(g_p)
        p_c = scipy.sparse.linalg.lsqr(S_B, v).todense()
        p_p = scipy.sparse.linalg.inv(B).dot(-g_p - E.dot(p_c)).todense()
    else:
        print("No optimization possible (matrices are singular).")
        print("Using default operation.")
        J = scipy.sparse.hstack([J_c, J_p])
        H = J.T.dot(J)
        b = J.T.dot(e)
        [p, info] = scipy.sparse.linalg.minres(H, b)
        if info == 0:
            print("Convergence successful")
        else:
            print("Convergence unsuccessful")
        p_c = p[:n_images*7]
        p_p = p[n_images*7:]
    print("Bundle adjustment complete.")

    return [p_c, p_p.reshape((-1, 3)), np.matrix(list_3dpoints), np.matrix(list_colors)]
