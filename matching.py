import cv2
import numpy as np

import math
from PIL import ExifTags
from PIL import Image

def find_features(n_imgs, imgset):
    """
    Testing version
    Reads in n_imgs images from a directory and returns lists of keypoints and
    descriptors each of dimensions (n_imgs x n_kpts).
    n_imgs: Number of images to read in and process.
    imgset: String. Name of image set to read in and get matches/keypoints for
    """
    images = []
    keypoints = []
    descriptors = []
    sift = cv2.SIFT_create()
    for i in range(n_imgs):
        if imgset == 'templering':
            img = cv2.imread(f'./{i:02d}.png', cv2.IMREAD_GRAYSCALE)
            K = np.matrix('1520.40 0.00 302.32; 0.00 1525.90 246.87; 0.00 0.00 1.00')
        

        elif imgset == 'Hollenbeck':
            print(f'./DSCN045{i:01d}.JPG')
            img = cv2.imread(f'./DSCN045{i:01d}.JPG',cv2.IMREAD_GRAYSCALE)
            K = np.matrix('1066.34 0 817.53;0 1068.31 579.83; 0 0 1')
            
        else: raise ValueError('Need to pass in valid imgset name!')
        images.append(img)
        kp, des = sift.detectAndCompute(images[-1], None)
        keypoints.append(kp)
        descriptors.append(des)
    return images, keypoints, descriptors, K

def find_featuresV2(imgList, camID):
    '''
    Reads in the camera data and list of images and returns a list of images, list of keypoints, list of descriptors, and list of intrinsic matrices.
    :param imgList: List of image paths
    :param camID: Identifier Information for the Camera Used
    '''
    images = []
    keypoints = []
    desc = []
    K = []
    #print("FF:",camID)
    sift = cv2.SIFT_create()
    for i in range(len(imgList)):
        img = cv2.imread(imgList[i], cv2.IMREAD_GRAYSCALE)
        if not(img is None):
            K = calibrate(imgList[i], camID, img.shape[:2])
            images.append(img)
            kp, des = sift.detectAndCompute(images[-1],None)
            keypoints.append(kp)
            desc.append(des)
    return images, keypoints, desc, K

def find_featuresMulti(imgList, camID):
    '''Implementation of find_featuresV2 that attempts handle multiple K'''
    images = []
    keypoints = []
    desc = []
    Ks = []
    sift = cv2.SIFT_create()
    for i in range(len(imgList)):
        img = cv2.imread(imgList[i], cv2.IMREAD_GRAYSCALE)
        if not(img is None):
            Ks.append(calibrate(imgList[i], camID, img.shape[:2]))
            images.append(img)
            kp, des = sift.detectandCompute(images[-1],None)
            keypoints.append(kp)
            desc.append(des)
    return images, keypoints, desc, Ks

def calibrate(img, camID, imgSize):
    '''
    Calibration Method from M3 - Get camera intrinsics from file, compare to image data and adjust the matrix appropriately
    img: Path for image to be calibrated
    camID: [initK (3x3 Matrix), initFocal (int), initSize (tuple)]
    imgShape: height and width of original image
    '''
    #Assuming the camID is the intrinsic Data
    exif = Image.open(img)._getexif()
    #print(img)
    #print(exif)
    if(exif is None) or not(37386 in exif.keys()):
        print("No Exif Data Available")
        return camID[0]
    else:
        newK = np.copy(camID[0])
        currFocal = exif[37386]
        if not(camID[1] == currFocal):
            ratio = currFocal/camID[1]
            newK[0,1] = newK[0,1] * ratio
            newK[1,1] = newK[1,1] * ratio
        if not(camID[2][0] == imgSize[0] and camID[2][1] == imgSize[1]):
            ratioy = imgSize[0]/camID[2][0]
            ratiox = imgSize[1]/camID[2][1]
            newK[0,:] *= ratiox
            newK[1,:] *= ratioy
        return newK
                      
    

def find_matches(matcher, keypoints, descriptors, lowes_ratio=0.7):
    """
    Performs kNN matching with k=2 and Lowes' ratio test to return a list of dimensions
    n_imgs x n_imgs where matches[i][j] is the list of cv2.DMatch objects for images i and j
    matcher: cv2.BFMatcher
    keypoints: List of lists of cv2 keypoints. keypoints[i] is list for image i.
    descriptors: List of lists of SIFT descriptors. descriptors[i] is list for image i.
    """
    matches = []
    n_imgs = len(keypoints)
    for i in range(n_imgs):
        matches.append([])
        for j in range(n_imgs):
            if j <= i: matches[i].append(None)
            else:
                match = []
                m = matcher.knnMatch(descriptors[i], descriptors[j], k=2)
                for k in range(len(m)):
                    try:
                        if m[k][0].distance < lowes_ratio*m[k][1].distance:
                            match.append(m[k][0])
                    except:
                        continue
                matches[i].append(match)
    return matches


def remove_outliers(matches, keypoints):
    """
    Calculate fundamental matrix between 2 images to remove incorrect matches.
    Return matches with outlier removed. Rejects matches between images if there are < 20
    matches: List of lists of lists where matches[i][j][k] is the kth cv2 Dmatch object for images i and j
    keypoints: List of lists of cv2 keypoint objects. keypoints[i] is list for image i.
    """

    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if j <= i: continue
            if len(matches[i][j]) < 20:
                matches[i][j] = []
                continue
            kpts_i = []
            kpts_j = []
            for k in range(len(matches[i][j])):
                kpts_i.append(keypoints[i][matches[i][j][k].queryIdx].pt)
                kpts_j.append(keypoints[j][matches[i][j][k].trainIdx].pt)
            kpts_i = np.int32(kpts_i)
            kpts_j = np.int32(kpts_j)
            F, mask = cv2.findFundamentalMat(kpts_i, kpts_j, cv2.FM_RANSAC, ransacReprojThreshold=3)
            if np.linalg.det(F) > 1e-7: raise ValueError(f"Bad F_mat between images: {i}, {j}. Determinant: {np.linalg.det(F)}")
            matches[i][j] = np.array(matches[i][j])
            if mask is None:
                matches[i][j] = []
                continue
            matches[i][j] = matches[i][j][mask.ravel() == 1]
            matches[i][j] = list(matches[i][j])

            if len(matches[i][j]) < 20:
                matches[i][j] = []
                continue

    return matches

def num_matches(matches):
    """Count matches before/after outlier removal """
    n_matches = 0
    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if j <= i: continue
            n_matches += len(matches[i][j])

    return n_matches

def print_num_img_pairs(matches):
    '''Prints the current number of accepted matches'''
    num_img_pairs = 0
    num_pairs = 0
    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if j <= i: continue
            num_pairs += 1
            if len(matches[i][j]) > 0: num_img_pairs += 1

    print(f"Number of img pairs is {num_img_pairs} out of possible {num_pairs}")

def create_img_adjacency_matrix(n_imgs, matches):
    """
    Returns a n_imgs x n_imgs matrix where if img_adjacency[i][j] = 1, the images[i] and images[j]
    have a sufficient number of matches (<20 during remove-outliers), and are regarded as viewing the same scene.
    n_imgs: Integer. Total number of images to be used in reconstruction
    matches: List of lists of lists where matches[i][j][k] is the kth cv2 Dmatch object for images i and j
    """
    num_img_pairs = 0
    num_pairs = 0
    pairs = []
    img_adjacency = np.zeros((n_imgs, n_imgs))
    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if j <= i: continue
            num_pairs += 1
            if len(matches[i][j]) > 0:
                num_img_pairs += 1
                pairs.append((i,j))
                img_adjacency[i][j] = 1

    list_of_img_pairs = pairs
    return img_adjacency, list_of_img_pairs
