import numpy as np
import random
import cv2


class Point3D_with_views:
    #Each 3D point needs to record which 2d points were used to find it
    def __init__(self, point3d, source_2dpt_idxs):
        self.point3d = point3d
        self.source_2dpt_idxs = source_2dpt_idxs

def best_img_pair(img_adjacency, matches, keypoints, K, top_x_perc=0.2):
    """
    Returns the image pair that has the greatest difference in rotation given that it is in the
    top xth percentile for number of matches between images. 
    img_adjacency: Matrix with value at indices i and j = 1 if images have matches, else 0
    matches: List of lists of lists where matches[i][j][k] is the kth cv2.Dmatch object for images i and j
    keypoints: List of lists of cv2 Keypoint objects. keypoints[i] is list for image i.
    K: Matrix. Intrinsic parameters of camera
    top_x_perc: Float. Threshold for considering image pairs to init reconstruction.
    """
    num_matches = []

    for i in range(img_adjacency.shape[0]):
        for j in range(img_adjacency.shape[1]):
            if img_adjacency[i][j] == 1:
                num_matches.append(len(matches[i][j]))

    num_matches = sorted(num_matches, reverse=True) #Find top x% of matches
    min_match_idx = int(len(num_matches)*top_x_perc)
    min_matches = num_matches[min_match_idx]
    print(min_matches, min_match_idx)
    best_R = 0
    best_pair = None

    #For all img pairs in top xth %ile of matches, find pair with greatest rotation between images.
    for i in range(img_adjacency.shape[0]):
        for j in range(img_adjacency.shape[1]):

            if img_adjacency[i][j] == 1:

                if len(matches[i][j]) > min_matches:
                    kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs = get_aligned_kpts(i, j, keypoints, matches)
                    E, _ = cv2.findEssentialMat(kpts_i, kpts_j, K, cv2.FM_RANSAC, 0.999, 1.0)
                    points, R1, t1, mask = cv2.recoverPose(E, kpts_i, kpts_j, K)

                    rvec, _ = cv2.Rodrigues(R1)
                    rot_angle = abs(rvec[0]) +abs(rvec[1]) + abs(rvec[2])# sum rotation angles for each dimension

                    if (rot_angle > best_R or best_pair == None) :
                        best_R = rot_angle
                        best_pair = (i,j)

    return best_pair


def best_pairV2(img_adjacency, matches, keypoints, Ks, top_x_perc=0.2):
    '''Implementation of best_pair that allows for multiple intrinsic matrices to be used'''
    num_matches = [] #Find number of matches per pair
    for i in range(img_adjacency.shape[0]): 
        for j in range(img_adjacency.shape[1]):
            if img_adjacency[i][j] == 1:
                num_matches.append(len(matches[i][j]))
    num_matches = sorted(num_matches, reverse = True)
    min_match_idx = int(len(num_matches)*top_x_perc)
    min_matches = num_matches[min_match_idx]

    best_R = 0
    best_pair = None

    for i in range(img_adjacency.shape[0]):
        for j in range(img_adjacency.shape[1]):
            if img_adjacency[i][j] == 1:
                 
                if len(matches[i][j]) > min_matches:
                    #Undistort kpts_i and kpts_j -> convert the keypoints to their locations in a camera with eye(3,3) as the intrinsic matrix
                    kpts_i, kpts_j, kpts_i_idx, kpts_j_idx = get_aligned_kpts(i,j,keypoints,matches)
                    kpts_i = cv2.undistortPoints(kpts_i,Ks[i],dist[i]) #Should allow for the E matrix to be found even if we don't know Ks[i] == Ks[j]
                    kpts_j = cv2.undistortPoints(kpts_j,Ks[j],dist[j])

                    E, _ = cv2.findEssentialMat(kpts_i, kpts_j, np.eye(3,3),cv2.FM_RANSAC, 0.999,1.0)
                    points, R1, t1, mask = cv2.recoverPose(E,kpts_i, kpts_j, np.eye(3,3))

                    rvec, _ = cv2.Rodrigues(R1)
                    rot_angle = abs(rvec[0]) + abs(rvec[1]) + abs(rvec[2])
                    if (rot_angle > best_R or best_pair == None):
                        best_R = rot_angle
                        best_pair = (i,j)
    return best_pair
                    
                    
                       

def get_aligned_kpts(i, j, keypoints, matches, mask=None):
    """
    Returns aligned list of keypoints so that kpts_i[x] and kpts_j[x] refer to the same match. Also, returns a list of the indicies from keypoints[i] for kpts_i 
    i: Integer. Index of first image
    j: Integer. Index of second image
    keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
    matches: List of lists of lists where matches[i][j][k] is the kth cv2.Dmatch object for images i and j
    mask: Array. Equal to 0 at indices where information on that keypoint is not needed, else 1
    """
    if mask is None:
        mask = np.ones(len(matches[i][j])) #if no mask is given, all matches used.

    kpts_i, kpts_i_idxs, kpts_j, kpts_j_idxs = [], [], [], []
    for k in range(len(matches[i][j])):
        if mask[k] == 0: continue #Skip the keypoint pair if masked out
        kpts_i.append(keypoints[i][matches[i][j][k].queryIdx].pt)
        kpts_i_idxs.append(matches[i][j][k].queryIdx)
        kpts_j.append(keypoints[j][matches[i][j][k].trainIdx].pt)
        kpts_j_idxs.append(matches[i][j][k].trainIdx)
    kpts_i = np.array(kpts_i)
    kpts_j = np.array(kpts_j)
    kpts_i = np.expand_dims(kpts_i, axis=1) #this seems to be required for cv2.undistortPoints and cv2.trangulatePoints to work
    kpts_j = np.expand_dims(kpts_j, axis=1)

    return kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs

def triangulate_points_and_reproject(R_l, t_l, R_r, t_r, K, points3d_with_views, img_idx1, img_idx2, kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs, reproject=True):
    """
    Triangulate points given 2D correspondences as well as camera parameters.
    R_l: Rotation matrix for left image
    t_l: Translation vector for left image
    R_r: Rotation matrix for right image
    t_r: Translation vector for right image
    K: Intrinsics matrix
    points3d_with_views: List of Point3D_with_views objects. Will have new points appended to it
    img_idx1: Index of left image
    img_idx2: Index of right image
    kpts_i: List of index-aligned keypoint coordinates in left image
    kpts_j: List of index-aligned keypoint coordinates in right image
    kpts_i_idxs: Indexes of keypoints for left image
    kpts_j_idxs: Indexes of keypoints for right image
    reproject: Boolean. True if reprojection errors desired
    """

    print(f"Triangulating: {len(kpts_i)} points.")
    P_l = np.dot(K, np.hstack((R_l, t_l)))
    P_r = np.dot(K, np.hstack((R_r, t_r)))

    kpts_i = np.squeeze(kpts_i)
    kpts_i = kpts_i.transpose()
    kpts_i = kpts_i.reshape(2,-1)
    kpts_j = np.squeeze(kpts_j)
    kpts_j = kpts_j.transpose()
    kpts_j = kpts_j.reshape(2,-1)

    #reconstruct the 3dpts in homogenous coordinates
    point_4d_hom = cv2.triangulatePoints(P_l, P_r, kpts_i, kpts_j)
    points_3D = cv2.convertPointsFromHomogeneous(point_4d_hom.transpose())
    for i in range(kpts_i.shape[1]):
        source_2dpt_idxs = {img_idx1:kpts_i_idxs[i], img_idx2:kpts_j_idxs[i]}
        pt = Point3D_with_views(points_3D[i], source_2dpt_idxs)
        points3d_with_views.append(pt)

    if reproject: #find the avg reprojection error
        kpts_i = kpts_i.transpose()
        kpts_j = kpts_j.transpose()
        rvec_l, _ = cv2.Rodrigues(R_l)
        rvec_r, _ = cv2.Rodrigues(R_r)
        projPoints_l, _ = cv2.projectPoints(points_3D, rvec_l, t_l, K, distCoeffs=np.array([]))
        projPoints_r, _ = cv2.projectPoints(points_3D, rvec_r, t_r, K, distCoeffs=np.array([]))
        delta_l , delta_r = [], []
        for i in range(len(projPoints_l)):
            delta_l.append(abs(projPoints_l[i][0][0] - kpts_i[i][0]))
            delta_l.append(abs(projPoints_l[i][0][1] - kpts_i[i][1]))
            delta_r.append(abs(projPoints_r[i][0][0] - kpts_j[i][0]))
            delta_r.append(abs(projPoints_r[i][0][1] - kpts_j[i][1]))
        avg_error_l = sum(delta_l)/len(delta_l)
        avg_error_r = sum(delta_r)/len(delta_r)
        print(f"Average reprojection error for just-triangulated points on image {img_idx1} is:", avg_error_l, "pixels.")
        print(f"Average reprojection error for just-triangulated points on image {img_idx2} is:", avg_error_r, "pixels.")
        errors = list(zip(delta_l, delta_r))
        return points3d_with_views, errors, avg_error_l, avg_error_r

    return points3d_with_views

#Need to alter - may or may not be correct
#Do kpts_i and kpts_j need to be in terms of the normed camera? if so Ks[i] -> np.eye(3,3)
def triangulate_points_and_reprojectV2(R_l, t_l, R_r, t_r, Ks, points3d_with_views, img_idx1, img_idx2, kpts_i, kpts_j,kpts_i_idxs, kpts_j_idxs, reproject=True):
    '''Proposed implementation of triangulate_points_and reproject() to take multiple intrinsic Matrices K'''
    print(f"Triangulating: {len(kpts_i)} points.")
    P_l = np.dot(Ks[img_idx1], np.hstack((R_l, t_l)))
    P_r = np.dot(Ks[img_idx2], np.hstack((R_r, t_r)))

    kpts_i = np.squeeze(kpts_i)
    kpts_i = kpts_i.transpose()
    kpts_i = kpts_i.reshape(2,-1)
    kpts_j = np.squeeze(kpts_j)
    kpts_j = kpts_j.transpose()
    kpts_j = kpts_j.reshape(2,-1)

    point_4d_hom = cv2.triangulatePoints(P_l, P_r, kpts_i, kpts_j)
    points_3D = cv2.convertPointsFromHomogeneous(point_4d_hom.transpose())
    for i in range(kpts_i.shape[1]):
        source_2dpt_idxs = {img_idx1:kpts_i_idxs[i], img_idx2:kpts_j_idxs[i]}
        pt = Point3D_with_views(points_3D[i], source_2dpt_idxs)
        points3d_with_views.append(pt)

    if reproject:
        kpts_i = kpts_i.transpose()
        kpts_j = kpts_j.transpose()
        rvec_l, _ = cv2.Rodrigues(R_l)
        rvec_r, _ = cv2.Rodrigues(R_r)
        projPoints_l, _ = cv2.projectPoints(points_3D, rvec_l, t_l, Ks[img_idx1], distCoeffs=np.array([])) #May need to switch to eye(3,3) if points_3d are in terms of normed camera 
        projPoints_r, _ = cv2.projectPoints(points_3D, rvec_r, t_r, Ks[img_idx2], distCoeffs=np.array([]))
        delta_l , delta_r = [], []
        for i in range(len(projPoints_l)):
            delta_l.append(abs(projPoints_l[i][0][0] - kpts_i[i][0]))
            delta_l.append(abs(projPoints_l[i][0][1] - kpts_i[i][1]))
            delta_r.append(abs(projPoints_r[i][0][0] - kpts_j[i][0]))
            delta_r.append(abs(projPoints_r[i][0][1] - kpts_j[i][1]))
        avg_error_l = sum(delta_l)/len(delta_l)
        avg_error_r = sum(delta_r)/len(delta_r)
        print(f"Average reprojection error for just-triangulated points on image {img_idx1} is:", avg_error_l, "pixels.") #High reprojection errors can denote major misunderstanding
        print(f"Average reprojection error for just-triangulated points on image {img_idx2} is:", avg_error_r, "pixels.")
        errors = list(zip(delta_l, delta_r))
        return points3d_with_views, errors, avg_error_l, avg_error_r

    return points3d_with_views

def initialize_reconstruction(keypoints, matches, K, img_idx1, img_idx2):
    """
    Solve for pose of initial image pair and triangulate points seen by them.
    keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
    matches: List of lists of lists where matches[i][j][k] is the kth cv2 Dmatch object for images i and j
    K: Intrinsics matrix
    img_idx1: Index of left image
    img_idx2: Index of right image
    """

    
    kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs = get_aligned_kpts(img_idx1, img_idx2, keypoints, matches)
    E, _ = cv2.findEssentialMat(kpts_i, kpts_j, K, cv2.FM_RANSAC, 0.999, 1.0) #Find the essential matrix
    points, R1, t1, mask = cv2.recoverPose(E, kpts_i, kpts_j, K)
    assert abs(np.linalg.det(R1)) - 1 < 1e-7

    R0 = np.eye(3, 3)
    t0 = np.zeros((3, 1))

    points3d_with_views = []
    points3d_with_views = triangulate_points_and_reproject(
        R0, t0, R1, t1, K, points3d_with_views, img_idx1, img_idx2, kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs, reproject=False)

    return R0, t0, R1, t1, points3d_with_views


def initialize_reconstructionV2(keypoints, matches, Ks, img_idx1, img_idx2):
    '''Implementation of initialize_reconstructionV2 to accomodate multiple intrinsic matrices'''

    kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs = get_aligned_kpts(img_idx1, img_idx2, keypoints, matches)

    kpts_i_norm = cv2.undistortPoints(kpts_i,Ks[i],dist[i]) #Should allow for the E matrix to be found even if we don't know Ks[i] == Ks[j]
    kpts_j_norm = cv2.undistortPoints(kpts_j,Ks[j],dist[j])

    E, _ = cv2.findEssentialMat(kpts_i_norm, kpts_j_norm, np.eye(3,3),cv2.FM_RANSAC, 0.999,1.0)
    points, R1, t1, mask = cv2.recoverPose(E,kpts_i_norm, kpts_j_norm, np.eye(3,3))    

    assert abs(np.linalg.det(R1)) - 1 < 1e-7

    R0 = np.eye(3, 3)
    t0 = np.zeros((3, 1))

    points3d_with_views = []
    points3d_with_views = triangulate_points_and_reprojectV2(
        R0, t0, R1, t1, Ks, points3d_with_views, img_idx1, img_idx2, kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs, reproject=False)

    return R0, t0, R1, t1, points3d_with_views



def get_idxs_in_correct_order(idx1, idx2):
    """First idx must be smaller than second when using upper-triangular arrays (matches, keypoints)"""
    if idx1 < idx2: return idx1, idx2
    else: return idx2, idx1

def images_adjacent(i, j, img_adjacency):
    """Return true if both images view the same scene (have enough matches)."""
    if img_adjacency[i][j] == 1 or img_adjacency[j][i] == 1:
        return True
    else:
        return False

def has_resected_pair(unresected_idx, resected_imgs, img_adjacency):
    """Return true if unresected_idx image has matches to >= 1 currently resected image(s) """
    for idx in resected_imgs:
        if img_adjacency[unresected_idx][idx] == 1 or img_adjacency[idx][unresected_idx] == 1:
            return True
    return False

def has_unresected_pair(resected_idx, unresected_imgs, img_adjacency):
    """Return true if resected_idx image has matches to >= 1 currently unresected image(s) """
    for idx in unresected_imgs:
        if img_adjacency[resected_idx][idx] == 1 or img_adjacency[idx][resected_idx] == 1:
            return True
    return False


def next_img_pair_to_grow_reconstruction(n_imgs, init_pair, resected_imgs, unresected_imgs, img_adjacency, skip):
    """
    Given initial image pair, resect images between the initial ones, then extend reconstruction in both directions.
    :param n_imgs: Number of images to be used in reconstruction
    :param init_pair: tuple of indicies of images used to initialize reconstruction
    :param resected_imgs: List of indices of resected images
    :param unresected_imgs: List of indices of unresected images
    :param img_adjacency: Matrix with value at indices i and j = 1 if images have matches, else 0
    """

    if len(unresected_imgs) == 0: raise ValueError('Should not check next image to resect if all have been resected already!')
    straddle = False
    if init_pair[1] - init_pair[0] > n_imgs/2 : straddle = True #initial pair straddles "end" of the circle (ie if init pair is idxs (0, 49) for 50 images)

    init_arc = init_pair[1] - init_pair[0] + 1 # Number of images between and including initial pair
    #print(straddle)

    #fill in images between initial pair (if possible)
    if len(resected_imgs) < init_arc:
        if straddle == False: idx = resected_imgs[-2] + 1
        else: idx = resected_imgs[-1] + 1
        while True:
            if idx not in resected_imgs:
                prepend = True
                unresected_idx = idx
                resected_idx = random.choice(resected_imgs)
                return resected_idx, unresected_idx, prepend
            idx = idx + 1 % n_imgs

    unresected_idx = -1
    extensions = len(resected_imgs) - init_arc # How many images have been resected after the initial arc

    i = 0
    while ((init_pair[1] + i) % n_imgs) in resected_imgs:
        i+=1
    
        
    extV3 = [0,i-1]

    while((init_pair[0] - extV3[0]) % n_imgs) in resected_imgs:
        extV3[0] +=1

    extV3[0] = extV3[0]-1

    #print(extV3)
    
    #extV2 = [init_pair[0]-min(resected_imgs), max(resected_imgs)-init_pair[1]] #not necessarily true if images 0 and num_imgs are linked first

    while(unresected_idx == -1):
        #print(extV3) #Unbalanced if diff(extV3[0], extV3[1]) > 1
        #print("Extensions:",extensions)

        #Can we be unbalanced if we are straddling?
        if straddle == True: #smaller init_idx should be increased and larger decreased
            #print("Set A")
            if extensions % 2 == 0:
                unresected_idx = (init_pair[0] + int(extensions/2) + 1) % n_imgs
                resected_idx = (unresected_idx - 1) % n_imgs
            else:
                unresected_idx = (init_pair[1] - int(extensions/2) - 1) % n_imgs
                resected_idx = (unresected_idx + 1) % n_imgs
                
        elif abs(extV3[0]-extV3[1]) <= 1: #Need to guarentee that resected is in resected and unresected is in unresected - some edge cases around 0
            #print("Set B")
            if extensions % 2 == 0: #Expand L -> use max resected
                #print("L")
                unresected_idx = (init_pair[1] + int(extensions/2) + 1) % n_imgs
                resected_idx = (unresected_idx - 1) % n_imgs
                extV3[1] = extV3[1]+1
                #resected_idx = max(resected_imgs)
            else: #Expand R ->use min resected
                #print("R")
                unresected_idx = (init_pair[0] - int(extensions/2) - 1) % n_imgs
                resected_idx = (unresected_idx + 1) % n_imgs
                #resected_idx = min(resected_imgs)
                extV3[0] = extV3[0]+1
        else:
            if max(resected_imgs) == n_imgs -1 and min(resected_imgs) == 0:
                #print("Keep")
                if extensions % 2 == 0: #Expand L -> use max resected
                    #print("L")
                    unresected_idx = (init_pair[1] + int(extensions/2) + 1) % n_imgs
                    resected_idx = (unresected_idx - 1) % n_imgs
                    extV3[1] = extV3[1]+1
                    #resected_idx = max(resected_imgs)
                else: #Expand R ->use min resected
                    #print("R")
                    unresected_idx = (init_pair[0] - int(extensions/2) - 1) % n_imgs
                    resected_idx = (unresected_idx + 1) % n_imgs
                    #resected_idx = min(resected_imgs)
                    extV3[0] = extV3[0]+1
            elif max(resected_imgs) == n_imgs -1:
                #print("Max Reached First")
                resected_idx = min(resected_imgs)
                unresected_idx = max(unresected_imgs)
            elif min(resected_imgs) == 0:
                #print("Min reached First")
                resected_idx = max(resected_imgs)
                unresected_idx=min(unresected_imgs)
                
        #print("unresected_idx:",unresected_idx)
        #print("resected_idx:",resected_idx)
        #print("Skip:",skip)
        if unresected_imgs == skip and (extensions) != len(resected_imgs) - init_arc:
            skip = []

        #print("New Skip:",skip)
        
        if unresected_idx in skip or not(unresected_idx in unresected_imgs):
            unresected_idx = -1
            extensions += 1

    prepend = False
    return resected_idx, unresected_idx, prepend

def check_and_get_unresected_point(resected_kpt_idx, match, resected_idx, unresected_idx):
    """
    Check if a 3D point seen by the given resected image is involved in a match to the unresected image
    and is therefore usable for Pnp.
    resected_kpt_idx: Index of keypoint in keypoints list for resected image
    match: cv2.Dmatch object
    resected_idx: Index of the resected image
    unresected_idx: Index of the unresected image
    """
    if resected_idx < unresected_idx:
        if resected_kpt_idx == match.queryIdx:
            unresected_kpt_idx = match.trainIdx
            success = True
            return unresected_kpt_idx, success
        else:
            return None, False
    elif unresected_idx < resected_idx:
        if resected_kpt_idx == match.trainIdx:
            unresected_kpt_idx = match.queryIdx
            success = True
            return unresected_kpt_idx, success
        else:
            return None, False

def get_correspondences_for_pnp(resected_idx, unresected_idx, pts3d, matches, keypoints):
    """
    Returns index aligned lists of 3D and 2D points to be used for Pnp. For each 3D point check if it is seen
    by the resected image, if so check if there is a match for it between the resected and unresected image.
    If so that point will be used in Pnp. Also keeps track of matches that do not have associated 3D points,
    and therefore need to be triangulated.
    resected_idx: Index of resected image to be used in Pnp
    unresected_idx Index of unresected image to be used in Pnp
    pts3d: List of Point3D_with_views objects
    matches: List of lists of lists where matches[i][j][k] is the kth cv2.Dmatch object for images i and j
    keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
    """
    idx1, idx2 = get_idxs_in_correct_order(resected_idx, unresected_idx)
    triangulation_status = np.ones(len(matches[idx1][idx2])) # if triangulation_status[x] = 1, then matches[x] used for triangulation
    pts3d_for_pnp = []
    pts2d_for_pnp = []
    for pt3d in pts3d:
        if resected_idx not in pt3d.source_2dpt_idxs: continue
        resected_kpt_idx = pt3d.source_2dpt_idxs[resected_idx]
        for k in range(len(matches[idx1][idx2])):
            unresected_kpt_idx, success = check_and_get_unresected_point(resected_kpt_idx, matches[idx1][idx2][k], resected_idx, unresected_idx)
            if not success: continue
            pt3d.source_2dpt_idxs[unresected_idx] = unresected_kpt_idx #Add new 2d/3d correspondences to 3D point object
            pts3d_for_pnp.append(pt3d.point3d)
            pts2d_for_pnp.append(keypoints[unresected_idx][unresected_kpt_idx].pt)
            triangulation_status[k] = 0

    return pts3d, pts3d_for_pnp, pts2d_for_pnp, triangulation_status

def do_pnp(pts3d_for_pnp, pts2d_for_pnp, K, iterations=200, reprojThresh=5):
    """
    Performs Pnp with Ransac implemented manually. The camera pose which has the most inliers (points which
    when reprojected are sufficiently close to their keypoint coordinate) is deemed best and is returned.
    pts3d_for_pnp: list of index aligned 3D coordinates
    pts2d_for_pnp: list of index aligned 2D coordinates
    K: Intrinsics matrix
    iterations: Number of Ransac iterations
    reprojThresh: Max reprojection error for point to be considered an inlier
    """
    list_pts3d_for_pnp = pts3d_for_pnp
    list_pts2d_for_pnp = pts2d_for_pnp
    pts3d_for_pnp = np.squeeze(np.array(pts3d_for_pnp))
    pts2d_for_pnp = np.expand_dims(np.squeeze(np.array(pts2d_for_pnp)), axis=1)
    num_pts = len(pts3d_for_pnp)

    highest_inliers = 0
    for i in range(iterations):
        pt_idxs = np.random.choice(num_pts, 6, replace=False)
        pts3 = np.array([pts3d_for_pnp[pt_idxs[i]] for i in range(len(pt_idxs))])
        pts2 = np.array([pts2d_for_pnp[pt_idxs[i]] for i in range(len(pt_idxs))])
        _, rvec, tvec = cv2.solvePnP(pts3, pts2, K, distCoeffs=np.array([]), flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rvec)
        pnp_errors, projpts, avg_err, perc_inliers = test_reproj_pnp_points(list_pts3d_for_pnp, list_pts2d_for_pnp, R, tvec, K, rep_thresh=reprojThresh)
        if highest_inliers < perc_inliers:
            highest_inliers = perc_inliers
            best_R = R
            best_tvec = tvec
    R = best_R
    tvec = best_tvec
    print('rvec:', rvec,'\n\ntvec:', tvec)

    return R, tvec

#May not be convertable - are the 3dpts still in the normed camera?
#Assume so and Ks2d is the known intrinsic matrix for the pts2d
def do_pnpV2(pts3d_for_pnp, pts2d_for_pnp, Ks2d, iterations=200, reprojThresh=5):
    '''Implementation of do_pnpV2 for distinct intrinsic matrices'''

    pts2d_for_pnp = cv2.undistortPoints(pts2d_for_pnp, Ks2d) 

    list_pts3d_for_pnp = pts3d_for_pnp
    list_pts2d_for_pnp = pts2d_for_pnp
    pts3d_for_pnp = np.squeeze(np.array(pts3d_for_pnp))
    pts2d_for_pnp = np.expand_dims(np.squeeze(np.array(pts2d_for_pnp)), axis=1)
    num_pts = len(pts3d_for_pnp)

    highest_inliers = 0
    for i in range(iterations):
        pt_idxs = np.random.choice(num_pts, 6, replace=False)
        pts3 = np.array([pts3d_for_pnp[pt_idxs[i]] for i in range(len(pt_idxs))])
        pts2 = np.array([pts2d_for_pnp[pt_idxs[i]] for i in range(len(pt_idxs))])
        _, rvec, tvec = cv2.solvePnP(pts3, pts2, np.eye(3,3), distCoeffs=np.array([]), flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rvec)
        pnp_errors, projpts, avg_err, perc_inliers = test_reproj_pnp_points(list_pts3d_for_pnp, list_pts2d_for_pnp, R, tvec, np.eye(3,3), rep_thresh=reprojThresh)
        if highest_inliers < perc_inliers:
            highest_inliers = perc_inliers
            best_R = R
            best_tvec = tvec
    R = best_R
    tvec = best_tvec
    print('rvec:', rvec,'\n\ntvec:', tvec)

    return R, tvec

def prep_for_reproj(img_idx, points3d_with_views, keypoints):
    """
    Returns aligned vectors of 2D and 3D points to be used for reprojection
    img_idx: Index of image for which reprojection errors are desired
    points3d_with_views: List of Point3D_with_views objects. Will have new points appended to it
    keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
    """
    points_3d = []
    points_2d = []
    pt3d_idxs = []
    i = 0
    for pt3d in points3d_with_views:
        if img_idx in pt3d.source_2dpt_idxs.keys():
            pt3d_idxs.append(i)
            points_3d.append(pt3d.point3d)
            kpt_idx = pt3d.source_2dpt_idxs[img_idx]
            points_2d.append(keypoints[img_idx][kpt_idx].pt)
        i += 1

    return np.array(points_3d), np.array(points_2d), pt3d_idxs

#Forces 2dpt to be from the view of the normed camera
def prep_for_reprojV2(img_idx, points3d_with_views, keypoints, Ks):
    '''Implementation for prep_for_reproj with distinct intrinsic matrices'''
    kptsIdx = cv2.undistortPoints(keypoints[img_idx],Ks[img_idx])
    points_3d = []
    points_2d = []
    pt3d_idxs = []
    i = 0
    for pt3d in points3d_with_views:
        if img_idx in pt3d.source_2dpt_idxs.keys():
            pt3d_idxs.append(i)
            points_3d.append(pt3d.point3d)
            kpt_idx = pt3d.source_2dpt_idxs[img_idx]
            points_2d.append(kptsIdx[kpt_idx].pt)
        i += 1

    return np.array(points_3d), np.array(points_2d), pt3d_idxs

def calculate_reproj_errors(projPoints, points_2d):
    """
    Calculate reprojection errors (L1) between projected points and ground truth (keypoint coordinates)
    projPoints: list of index aligned  projected points
    points_2d: list of index aligned corresponding keypoint coordinates
    """
    assert len(projPoints) == len(points_2d)
    delta = []
    for i in range(len(projPoints)):
        delta.append(abs(projPoints[i] - points_2d[i]))

    average_delta = sum(delta)/len(delta) # 2-vector, average error for x and y coord
    average_delta = (average_delta[0] + average_delta[1])/2 # average error overall

    return average_delta, delta

def get_reproj_errors(img_idx, points3d_with_views, R, t, K, keypoints, distCoeffs=np.array([])):
    """
    Project all 3D points seen in image[img_idx] onto it, return reprojection errors and average error
    img_idx: Index of image for which reprojection errors are desired
    points3d_with_views: List of Point3D_with_views objects. Will have new points appended to it
    R: Rotation matrix
    t: Translation vector
    K: Intrinsics matrix
    keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
    """
    points_3d, points_2d, pt3d_idxs = prep_for_reproj(img_idx, points3d_with_views, keypoints)
    rvec, _ = cv2.Rodrigues(R)
    projPoints, _ = cv2.projectPoints(points_3d, rvec, t, K, distCoeffs=distCoeffs)
    projPoints = np.squeeze(projPoints)
    avg_error, errors = calculate_reproj_errors(projPoints, points_2d)

    return points_3d, points_2d, avg_error, errors

#Using Ks to ensure that each 2dpt is with respect to the normed camera liek the 3dpts
def get_reproj_errorsV2(img_idx, points3d_with_views, R, t, Ks, keypoints, distCoeffs=np.array([])):
    '''implementation for get_reproj_errors with multiple intrinsics matrices'''
    points_3d, points_2d, pt3d_idxs = prep_for_reprojV2(img_idx, points3d_with_views, keypoints, Ks) #Likely needs to be changed to use Ks
    rvec, _ = cv2.Rodrigues(R)
    projPoints, _ = cv2.projectPoints(points_3d, rvec, t, np.eye(3,3), distCoeffs=distCoeffs)#Changed to use eye(3,3)
    projPoints = np.squeeze(projPoints)
    avg_error, errors = calculate_reproj_errors(projPoints, points_2d)

    return points_3d, points_2d, avg_error, errors

def test_reproj_pnp_points(pts3d_for_pnp, pts2d_for_pnp, R_new, t_new, K, rep_thresh=5):
    """
    Reprojects points fed into Pnp back onto camera whose R and t were just obtained via Pnp.
    pts3d_for_pnp: List of axis aligned 3D points
    pts2d_for_pnp: List of axis aligned 2D points
    R_new: Rotation matrix of newly resected image
    t_new: Translation vector of newly resected image
    rep_thresh: Number of pixels reprojected points must be within to qualify as inliers
    """
    
    errors = []
    projpts = []
    inliers = []
    for i in range(len(pts3d_for_pnp)):
        Xw = pts3d_for_pnp[i][0]
        Xr = np.dot(R_new, Xw).reshape(3,1)
        Xc = Xr + t_new
        x = np.dot(K, Xc)
        x /= x[2]
        errors.append([np.float64(x[0] - pts2d_for_pnp[i][0]), np.float64(x[1] - pts2d_for_pnp[i][1])])
        projpts.append(x)
        if abs(errors[-1][0]) > rep_thresh or abs(errors[-1][1]) > rep_thresh: inliers.append(0)
        else: inliers.append(1)
    a = 0
    for e in errors:
        a = a + abs(e[0]) + abs(e[1])
    avg_err = a/(2*len(errors))
    perc_inliers = sum(inliers)/len(inliers)

    return errors, projpts, avg_err, perc_inliers

#May not be necessary if we assume that K = np.eye for every matrix - Not quite
def test_reproj_pnp_pointsV2(pts3d_for_pnp, pts2d_for_pnp, R_new, t_new, K2d, rep_thresh=5):

    pts2d_for_pnp = cv2.undistort(pts2d_for_pnp,K2d)

    errors = []
    projpts = []
    inliers = []
    for i in range(len(pts3d_for_pnp)):
        Xw = pts3d_for_pnp[i][0]
        Xr = np.dot(R_new, Xw).reshape(3,1)
        Xc = Xr + t_new
        x = np.dot(K, Xc)
        x /= x[2]
        errors.append([np.float64(x[0] - pts2d_for_pnp[i][0]), np.float64(x[1] - pts2d_for_pnp[i][1])])
        projpts.append(x)
        if abs(errors[-1][0]) > rep_thresh or abs(errors[-1][1]) > rep_thresh: inliers.append(0)
        else: inliers.append(1)
    a = 0
    for e in errors:
        a = a + abs(e[0]) + abs(e[1])
    avg_err = a/(2*len(errors))
    perc_inliers = sum(inliers)/len(inliers)

    return errors, projpts, avg_err, perc_inliers
