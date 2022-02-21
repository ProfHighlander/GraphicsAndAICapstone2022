import numpy as np
import cv2 
from matplotlib import pyplot as plt



def main():

#Greyscale removes colors from causing false negatives

    img_ = cv2.imread('test1.JPG')

    print(img_.shape)
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    img = cv2.imread('test3(size).JPG')
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   # img__ = cv2.imread('test5(distinct).jpg')
    #img3 = cv2.cvtColor(img__,cv2.COLOR_BGR2GRAY)


#Create Keypoints and Descriptors for each image

   #Switch to orb - SIFT finds more keypoints

    orb = cv2.SIFT_create()
# find the keypoints and descriptors with ORB
    kp1 = orb.detect(img1,None)
    print(len(kp1)) #Identical images will match every keypoint
    kp1, des1 = orb.compute(img1,kp1)
    kp2 = orb.detect(img2,None)
    kp2, des2 = orb.compute(img2,kp2)

    marked=cv2.drawKeypoints(img1 ,kp1 ,img_,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(marked),plt.show()

    marked2=cv2.drawKeypoints(img2 ,kp2 ,img ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(marked2),plt.show()
   # cv2.imwrite('test1-with-keypoints.jpg', marked)


#Finds the best k matches for each descriptor - result is a list of lists: each sublist consists of k elements (k is optional)
    bf = cv2.BFMatcher()

   
    #matches = bf.knnMatch(des1,des2, k=2)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance) #If using bf.match()

    #good = []
    #for m,n in matches:
    #   if m.distance < 0.8*n.distance:
    #       good.append([m])
    #matches = np.asarray(good)

    #print(len(matches))


    img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches, img_, flags=2) #If using bf.match
   # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches, img_, flags=2)


    #cv2.imwrite('test2-3comparisonV2.jpg',img3)
    plt.imshow(img3),plt.show()
    #image scale has a large effect on finding matches
    #order matters a lot for knn
    #can we scale the images 
    



    # create BFMatcher object


# Match descriptors.





### Apply ratio test - David Lowe's suggestion is to use 0.8 instead of 0.5
##    good = []
##    for m in matches:
##        if m[0].distance < 0.5*m[1].distance:
##            good.append(m)
##    matches = np.asarray(good)
##
##    print(len(matches))
##
###Find the homography - The map to linearly transform the points in image 1 into the points for image 2
###FindHomography uses RANSAC to estimate and remove the worse matches
    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        matchesMask = masked.ravel().tolist()
        print(H)

        h,w,n = img_.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)

        img3 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        #plt.imshow(img3),plt.show()
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

        #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,img_,**draw_params)

        #plt.imshow(img3, 'gray'),plt.show()
        
    else:
        raise AssertionError("Can’t find enough keypoints.")

#Theres a homography, now how to verify it

main()
