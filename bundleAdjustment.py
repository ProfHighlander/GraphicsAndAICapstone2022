import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from matplotlib import pyplot as plt


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    #Same as scipy tutorial
    """
    Prepares the sparsity matrix for bundle adjustment
    n_cameras: Number of cameras/images currently resected
    n_points: number of distinct 3D points that have been triangulated
    camera_indices: List. Value at ith position is index of camera that sees ith 2D point
    point_indices: List. Value at ith position is index of 3D point that sees ith 2D point
    """
    m = camera_indices.size * 2
    n = n_cameras * 12 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(12): #camera is the 3x3 Rotation Matrix and 3x1 translation vector
        A[2 * i, camera_indices * 12 + s] = 1
        A[2 * i + 1, camera_indices * 12 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 12 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 12 + point_indices * 3 + s] = 1

    return A

def project(points, camera_params, K):
    #Redefining how this works to have each camera record its own K matrix is not necessary since do_BAV2 translates all the image keypoints into their "normed" form
    """
    Projects 3D points onto camera coordinates
    points: N x 3 List of 3D point coordinates
    camera_params: N x 12 List of 12D camera parameters (r1, ... r9, t1, t2, t3)
    K: Intrinsics matrix
    """
    points_proj = []

    for idx in range(len(camera_params)):
        R = camera_params[idx][:9].reshape(3,3) #More intuitive than the scipy tutorial version
        rvec, _ = cv2.Rodrigues(R)
        t = camera_params[idx][9:]
        pt = points[idx]
        pt = np.expand_dims(pt, axis=0)
        pt, _ = cv2.projectPoints(pt, rvec, t, K, distCoeffs=np.array([])) #Assuming no distortion
        pt = np.squeeze(np.array(pt))
        points_proj.append(pt)

    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals for Bundle Adjustment.
    params: List of all reprojection error calculation parameters. First n_cameras*12 parameters are (r1, ..., r9, t1, t2, t3), one for each resected camera.
    Remaining n_points*3 paramaters are (x, y, z) coord of each triangulated point
    
    n_cameras: Integer. # of resected cameras
    n_points: Integer. # of triangulated points
    camera_indices: List of indices of cameras viewing each 2D observation
    point_indices: List of indices of 3D points corresponding to each 2D observation
    2D pixel coordinates of each observation by a camera of a 3D point
    K: Intrinsics matrix
    """
    camera_params = params[:n_cameras * 12].reshape((n_cameras, 12)) #reshape the cameras
    points_3d = params[n_cameras * 12:].reshape((n_points, 3)) #Reshape the 3dpts
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K) #Get the reprojected points
    return (points_proj - points_2d).ravel()#get the difference

def do_BA(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, K, ftol):
    """
    Perform Bundle Adjustment on all currently resected images and all triangulated 3D points. Return updated
    values for camera poses and 3D point coordinates.
    points3d_with_views: List of Point3D_with_views objects.
    R_mats: Dict mapping index of resected cameras to their Rotation matrix
    t_vecs: Dict mapping index of resected cameras to their translation vector
    resected_imgs: List of indices of resected images
    keypoints: List of lists of cv2 Keypoint objects. keypoints[i] is list for image i.
    ftol: Tolerance for change in total reprojection error. Used so scipy.optimize.least_squares knows
    when to stop adjusting
    """
    point_indices = []
    points_2d = []
    camera_indices = []
    points_3d = []
    camera_params = []
    BA_cam_idxs = {} # maps from true cam indices to 'normalized' (i.e 11, 23, 31 maps to -> 0, 1, 2)
    cam_count = 0

    for r in resected_imgs: #Build camera_params
        BA_cam_idxs[r] = cam_count
        camera_params.append(np.hstack((R_mats[r].ravel(), t_vecs[r].ravel()))) #Only considering R and t for each camera as K is the same for all images
        cam_count += 1

    for pt3d_idx in range(len(points3d_with_views)): #Build points_3d and points_2d
        points_3d.append(points3d_with_views[pt3d_idx].point3d)
        for cam_idx, kpt_idx in points3d_with_views[pt3d_idx].source_2dpt_idxs.items():
            if cam_idx not in resected_imgs: continue
            point_indices.append(pt3d_idx)
            camera_indices.append(BA_cam_idxs[cam_idx])#append normalized cam idx
            points_2d.append(keypoints[cam_idx][kpt_idx].pt)
    if len(points_3d[0]) == 3: points_3d = np.expand_dims(points_3d, axis=0)

    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    camera_indices = np.array(camera_indices)
    points_3d = np.squeeze(points_3d)
    camera_params = np.array(camera_params) #scipy depends on numpy, so arrays will be more useful than lists

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel())) #Make the initial guess
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices) 

    res = least_squares(fun, x0, jac_sparsity=A, verbose=0, x_scale='jac', loss='linear', ftol=ftol, xtol=1e-12, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K))#Same as scipy, difference in fun definition

    #plt.title("Graph {0}".format(len(resected_imgs)))
    #plt.plot(res.fun)
    #plt.show()
    
    adjusted_camera_params = res.x[:n_cameras * 12].reshape(n_cameras, 12) #unpack the results
    adjusted_points_3d = res.x[n_cameras * 12:].reshape(n_points, 3)
    adjusted_R_mats = {}
    adjusted_t_vecs = {}
    for true_idx, norm_idx in BA_cam_idxs.items():
        adjusted_R_mats[true_idx] = adjusted_camera_params[norm_idx][:9].reshape(3,3)
        adjusted_t_vecs[true_idx] = adjusted_camera_params[norm_idx][9:].reshape(3,1)
    R_mats = adjusted_R_mats
    t_vecs = adjusted_t_vecs
    for pt3d_idx in range(len(points3d_with_views)):
        points3d_with_views[pt3d_idx].point3d = np.expand_dims(adjusted_points_3d[pt3d_idx], axis=0)

    return points3d_with_views, R_mats, t_vecs

def do_BAV2(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, Ks, ftol):
    #Untested
    '''Revision of do_BA to alow for multiple distinct camera intrinsic matrices to be used'''
    point_indices = []
    points_2d = []
    camera_indices = []
    points_3d = []
    camera_params = []
    BA_cam_idxs = {} # maps from true cam indices to 'normalized' (i.e 11, 23, 31 maps to -> 0, 1, 2)
    cam_count = 0

    normedKpts = []
    #Create a counterpart to keypoints that are all based on the normed camera
    for i in range(len(keypoints)):
        normedKpts.append(cv2.undistortPoints(keypoints[i],Ks[i]))

    

    for r in resected_imgs:
        BA_cam_idxs[r] = cam_count
        camera_params.append(np.hstack((R_mats[r].ravel(), t_vecs[r].ravel()))) #Only considering R and t for each camera as K is the same for all images
        cam_count += 1

    for pt3d_idx in range(len(points3d_with_views)):
        points_3d.append(points3d_with_views[pt3d_idx].point3d)
        for cam_idx, kpt_idx in points3d_with_views[pt3d_idx].source_2dpt_idxs.items(): #What I attempted to do in milestone3
            if cam_idx not in resected_imgs: continue
            point_indices.append(pt3d_idx)
            camera_indices.append(BA_cam_idxs[cam_idx])#append normalized cam idx
            points_2d.append(normedKpts[cam_idx][kpt_idx].pt)
    if len(points_3d[0]) == 3: points_3d = np.expand_dims(points_3d, axis=0)

    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    camera_indices = np.array(camera_indices)
    points_3d = np.squeeze(points_3d)
    camera_params = np.array(camera_params)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices) #Similar, but slightly different from the scipy tutorial

    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', loss='linear', ftol=ftol, xtol=1e-12, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, np.eye(3,3)))#Same as scipy, difference in fun definition

    adjusted_camera_params = res.x[:n_cameras * 12].reshape(n_cameras, 12)
    adjusted_points_3d = res.x[n_cameras * 12:].reshape(n_points, 3)
    adjusted_R_mats = {}
    adjusted_t_vecs = {}
    for true_idx, norm_idx in BA_cam_idxs.items():
        adjusted_R_mats[true_idx] = adjusted_camera_params[norm_idx][:9].reshape(3,3)
        adjusted_t_vecs[true_idx] = adjusted_camera_params[norm_idx][9:].reshape(3,1)
    R_mats = adjusted_R_mats
    t_vecs = adjusted_t_vecs
    for pt3d_idx in range(len(points3d_with_views)):
        points3d_with_views[pt3d_idx].point3d = np.expand_dims(adjusted_points_3d[pt3d_idx], axis=0)

    return points3d_with_views, R_mats, t_vecs
