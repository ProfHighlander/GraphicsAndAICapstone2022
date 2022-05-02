
import cv2
#import pyntcloud #External Library for working with point clouds, replace with open3d
import random #Library for randomness
import numpy as np 


import bundleAdjustment as b 
import matching as m         
import reconstruction as r 

from scipy import stats

import open3d

def verify_Images(imgList):
    cluster = []
    for i in range(len(imgList)):
        imgI = cv2.imread(imgList[i]).shape
        inserted = False
        for j in cluster:
            if imgI[0] == j[0][0] and imgI[1] == j[0][1]:
                j.append(imgI)
                inserted=True
                break
        if not inserted:
            cluster.append([imgI])

    if len(cluster) == 1:
        #All images have the same size
        return 0, 0

    else:
        return 1, len(cluster)
            


def structure_from_motion(imgList, camID):
#--Matching/Outlier Removal--

    #n_imgs = 46 #46 if imgset = 'templering', 49 if imgset = 'Viking'
    #changes to accept any input set with a cameraID
    n_imgs = len(imgList)
    
    #images, keypoints, descriptors, K = m.find_features(n_imgs, imgset='templering') #Alter to take result of another function to get images and calibration matrices
    print(camID)
    images, keypoints,descriptors, K = m.find_featuresV2(imgList, camID[3:])
    # images, keypoints, descriptors, Ks = m.find_featuresMulti(imgList, cameraID)
    
    
    matcher = cv2.BFMatcher(cv2.NORM_L1)
    matches = m.find_matches(matcher, keypoints, descriptors)
    print('num_matches before outlier removal:', m.num_matches(matches))

    matches = m.remove_outliers(matches, keypoints)
    print("After outlier removal:")
    m.print_num_img_pairs(matches)

    img_adjacency, list_of_img_pairs = m.create_img_adjacency_matrix(n_imgs, matches)
    #print(list_of_img_pairs)
    #print(img_adjacency)
    

    #--Reconstruction Initialization--
    
    best_pair = r.best_img_pair(img_adjacency, matches, keypoints, K, top_x_perc=0.25)
    #best_pair = r.best_img_pairV2(img_adjacency, matches, keypoints, Ks, top_x_perc=0.2) #ready?
    
    R0, t0, R1, t1, points3d_with_views = r.initialize_reconstruction(keypoints, matches, K, best_pair[0], best_pair[1]) #Implementation of multiple K's
    #R0, t0, R1, t1, points3d_with_views = r.initialize_reconstruction(keypoints, matches, Ks, best_pair[0], best_pair[1]) #ready?

    R_mats = {best_pair[0]: R0, best_pair[1]: R1}
    t_vecs = {best_pair[0]: t0, best_pair[1]: t1}

    resected_imgs = [best_pair[0], best_pair[1]] 
    unresected_imgs = [i for i in range(len(images)) if i not in resected_imgs] 
    print('initial image pair:', resected_imgs)
    avg_err = 0
    

    #--Builds Reconstruction Beyond first pair
    
    BA_chkpts = [3,4,5,6, n_imgs]
    i = 6
    j = 1
    while i < n_imgs:
        i = int(6*(1.34**j))
        BA_chkpts.append(i)
        j += 1

    print(BA_chkpts)
    skippool = []
    while len(unresected_imgs) > 0:
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        resected_idx, unresected_idx, prepend = r.next_img_pair_to_grow_reconstruction(n_imgs, best_pair, resected_imgs, unresected_imgs, img_adjacency, skippool)
        points3d_with_views, pts3d_for_pnp, pts2d_for_pnp, triangulation_status = r.get_correspondences_for_pnp(resected_idx, unresected_idx, points3d_with_views, matches, keypoints)
        #print(len(pts2d_for_pnp))
        #print(len(pts3d_for_pnp))
        if len(pts3d_for_pnp) < 12:
            print(f"{len(pts3d_for_pnp)} is too few correspondences for pnp. Skipping imgs resected:{resected_idx} and unresected:{unresected_idx}")
            print(f"Currently resected imgs: {resected_imgs}, unresected: {unresected_imgs}")
            #unresected_imgs.remove(unresected_idx)
            skippool.append(unresected_idx)
            if len(skippool) == len(unresected_imgs) and len(skippool) == 1:
                #break
                print("Significant issues in the data set")
            continue

        skippool = []

        R_res = R_mats[resected_idx]
        t_res = t_vecs[resected_idx]
        print(f"Unresected image: {unresected_idx}, resected: {resected_idx}")
        
        R_new, t_new = r.do_pnp(pts3d_for_pnp, pts2d_for_pnp, K)
        #R_new, t_new = r.do_pnpV2(pts3d_for_pnp, pts2d_for_pnp, Ks) #ready??
        
        R_mats[unresected_idx] = R_new
        t_vecs[unresected_idx] = t_new
        print("U", unresected_imgs)
        print("R", resected_imgs)
        if prepend == True: resected_imgs.insert(0, unresected_idx)
        else: resected_imgs.append(unresected_idx)
        unresected_imgs.remove(unresected_idx)
        
        pnp_errors, projpts, avg_err, perc_inliers = r.test_reproj_pnp_points(pts3d_for_pnp, pts2d_for_pnp, R_new, t_new, K)
        #pnp_errors, projpts, avg_err, perc_inliers = r.test_reproj_pnp_pointsV2(pts3d_for_pnp, pts2d_for_pnp, R_new, t_new, Ks) - Questions
        
        print(f"Average error of reprojecting points used to resect image {unresected_idx} back onto it is: {avg_err}")
        print(f"Fraction of Pnp inliers: {perc_inliers} num pts used in Pnp: {len(pnp_errors)}")

        #Triangulation - Any new data points
        if resected_idx < unresected_idx:
            kpts1, kpts2, kpts1_idxs, kpts2_idxs = r.get_aligned_kpts(resected_idx, unresected_idx, keypoints, matches, mask=triangulation_status)
            if np.sum(triangulation_status) > 0: #at least 1 point needs to be triangulated
                points3d_with_views, tri_errors, avg_tri_err_l, avg_tri_err_r = r.triangulate_points_and_reproject(R_res, t_res, R_new, t_new, K, points3d_with_views, resected_idx, unresected_idx, kpts1, kpts2, kpts1_idxs, kpts2_idxs, reproject=True)
                #points3d_with_views, tri_errors, avg_tri_err_l, avg_tri_err_r = r.triangulate_points_and_reprojectV2(R_res, t_res, R_new, t_new, Ks, points3d_with_views, resected_idx, unresected_idx, kpts1, kpts2, kpts1_idxs, kpts2_idxs, reproject=True)
                #Ready?
        else:
            kpts1, kpts2, kpts1_idxs, kpts2_idxs = r.get_aligned_kpts(unresected_idx, resected_idx, keypoints, matches, mask=triangulation_status)
            if np.sum(triangulation_status) > 0: #at least 1 point needs to be triangulated
                points3d_with_views, tri_errors, avg_tri_err_l, avg_tri_err_r = r.triangulate_points_and_reproject(R_new, t_new, R_res, t_res, K, points3d_with_views, unresected_idx, resected_idx, kpts1, kpts2, kpts1_idxs, kpts2_idxs, reproject=True)
                #points3d_with_views, tri_errors, avg_tri_err_l, avg_tri_err_r = r.triangulate_points_and_reprojectV2(R_new, t_new, R_res, t_res, Ks, points3d_with_views, unresected_idx, resected_idx, kpts1, kpts2, kpts1_idxs, kpts2_idxs, reproject=True)
                #ready?

        #Bundle Adjustment - Too time-expensive to perform for every extension
        #If close enough, use low cost version
        if 0.8 < perc_inliers < 0.95 or 5 < avg_tri_err_l < 10 or 5 < avg_tri_err_r < 10: 
            #If % of inlers from Pnp is too low or triangulation error on either image is too high, bundle adjust
            points3d_with_views, R_mats, t_vecs = b.do_BA(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, K, ftol=1e0)
            #points3d_with_views, R_mats, t_vecs = b.do_BAV2(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, Ks, ftol=1e0) #questionably ready

        #If high reprojection error, use extensive version
        if len(resected_imgs) in BA_chkpts or len(unresected_imgs) == 0 or perc_inliers <= 0.8 or avg_tri_err_l >= 10 or avg_tri_err_r >= 10:
            #If % of inlers from Pnp is very low or triangulation error on either image is very high, bundle adjust with stricter tolerance
            points3d_with_views, R_mats, t_vecs = b.do_BA(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, K, ftol=1e-1)
            #points3d_with_views, R_mats, t_vecs = b.do_BAV2(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, Ks, ftol=1e-1) #Questionably ready
        
        av = 0
        for im in resected_imgs:
            p3d, p2d, avg_error, errors = r.get_reproj_errors(im, points3d_with_views, R_mats[im], t_vecs[im], K, keypoints, distCoeffs=np.array([]))
            #p3d, p2d, avg_error, errors = r.get_reproj_errorsV2(im, points3d_with_views, R_mats[im], t_vecs[im], Ks, keypoints, distCoeffs=np.array([]))
            
            print(f'Average reprojection error on image {im} is {avg_error} pixels')
            av += avg_error
        av = av/len(resected_imgs)
        print(f'Average reprojection error across all {len(resected_imgs)} resected images is {av} pixels')

    #--Visualize Point Cloud-- Need to filter out the outlier points


    x, y, z = [], [], []
    
    count = 0

    for pt3 in points3d_with_views:
        if abs(pt3.point3d[0][0]) + abs(pt3.point3d[0][1]) + abs(pt3.point3d[0][2]) < 100:
            x.append(pt3.point3d[0][0])
            y.append(pt3.point3d[0][1])
            z.append(pt3.point3d[0][2])
            
            count +=1
    avgX = np.average(np.asarray(x))
    avgY = np.average(np.asarray(y))
    avgZ = np.average(np.asarray(z))
    print("AVGPOINT:",avgX,avgY,avgZ)
    
    ab =[]
    for i in range(len(x)):
        ab.append(abs(avgX - x[i]) + abs(avgY - y[i]) + abs(avgZ - z[i]))
    
    pts3d = list(zip(x,y,z))
##    zeta = np.abs(stats.zscore(np.asarray(ab))) #Could get rid of any with zeta[i] > 2
##    i = 0
##    while i < len(zeta):
##        if zeta[i] > 2:
##            #print("Removed:", pts3d[i],":",zeta[i],":",ab[i])
##            pts3d.remove(pts3d[i])
##            zeta = np.delete(zeta, i)
##        else:
##            i = i + 1
##            
##    
##    print(count ,":", len(pts3d))
    pts3d = np.asmatrix(pts3d) #Return this
    return(pts3d)

