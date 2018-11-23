import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = points.dot(v.T)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + (1 - cos_theta) * dot.dot(v)

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras*9].reshape((n_cameras, 9))
    points_3d = params[n_cameras*9:].reshape((n_points, 3))
    points_proj = project(points_3d, camera_params[camera_indices])
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    #for j in point_indices:
        for s in range(3):
            A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

def adjust(camera_params, points_3d, n_cameras, n_points, camera_indices, points_2d):
    flat_cam_params = camera_params.ravel()
    flat_points3d = points_3d.ravel()

    print(flat_cam_params.shape)
    print(flat_points3d.shape)
    x0 = np.hstack((flat_cam_params, flat_points3d))
    print(x0[0])
    print(x0.shape)
    exit()
    opt_result = least_squares(fun, x0, verbose=2, x_scale='jac', tr_solver='lsmr',
                        ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, points_2d))

    x = opt_result.x
    print(x.shape)
    exit()
    return
