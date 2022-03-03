from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d


from PIL import ExifTags
from PIL import Image

import visualizations
import meshCreation

#Need Open3D to use the write_point_cloud() method



def getImageMatches(img1,img2):
    ''' Performs SIFT feature matching
        img1 - trainingImage
        img2 - quereyImage'''
    
    sift = cv.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    #matches = bf.knnMatch(desc1, desc2, k =2) #KNN matching
    
    matches = bf.match(desc1, desc2)
    matches = sorted(matches,key = lambda x: x.distance)
    
    return kp1,desc1,kp2,desc2,matches

def alignMatches(kp1, desc1, kp2, desc2, matches):
    '''kp1 - keypoints from image 1
        desc1 - descriptors from image 1
        kp2 - keypoints from image 2
        desc2 - descriptors from image 2
        matches - SIFT matched pairs'''
    
    matches = sorted(matches, key = lambda x:x.distance)
    img1idx = np.array([m.queryIdx for m in matches])
    img2idx = np.array([m.trainIdx for m in matches])
    
    kp1_ = (np.array(kp1))[img1idx]
    kp2_ = (np.array(kp2))[img2idx]
    #print(type(kp1_))
    #print(kp1_[0])

    img1pts = np.array([kp.pt for kp in kp1_])
    img2pts = np.array([kp.pt for kp in kp2_])

    return img1pts,img2pts,img1idx,img2idx

def computeEpiline(points, index, FMatrix):
    ''' points - Keypoints from the image
        index - The position of the image in the pair
        FMatrix - Fundamental Matrix'''
    if points.shape[1] == 2:
        points = cv.convertPointsToHomogeneous(points)[:,0,:]

    if index ==1:
        lines = FMatrix.dot(points.T)

    elif index==2:
        lines = FMatrix.T.dot(points.T)

    return lines.T


def extractPoses(EsMatrix):
    ''' EsMatrix - Essential Matrix '''

    
    u,d,v = np.linalg.svd(EsMatrix)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    Rs,Cs = np.zeros((4,3,3)),np.zeros((4,3))

    t = u[:,-1]
    #print(t)
    R1 = u.dot(W.dot(v))
    R2 = u.dot(W.T.dot(v))

    if np.linalg.det(R1) < 0:
        R1 = R1*-1

    if np.linalg.det(R2) < 0:
        R2 = R2*-1

    return R1, R2,t
    

def getTriangulatedPts(img1pts, img2pts, K, R, t, Rbase=None, tbase=None):
    ''' img1pts - Keypoints from img1
        img2pts - Keypoints from img2
        K - Intrinsic Camera Matrix
        R - Rotation Matrix
        t - Translation Vector
        Rbase - Rotation Matrix for img1
        tbase - Translation Vector for img1'''
    
    if Rbase is None: 
        Rbase = np.eye(3,3) 
    if tbase is None: 
        tbase = np.zeros((3,1))

    img1ptsHom = cv.convertPointsToHomogeneous(img1pts)[:,0,:]
    img2ptsHom = cv.convertPointsToHomogeneous(img2pts)[:,0,:]

    img1ptsNorm = (np.linalg.inv(K).dot(img1ptsHom.T)).T
    img2ptsNorm = (np.linalg.inv(K).dot(img2ptsHom.T)).T

    img1ptsNorm = cv.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
    img2ptsNorm = cv.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]

    t=t.reshape(3,1) #Questionable Fix
    #print (Rbase.shape, tbase.shape, R.shape, t.shape)
    
    
    pts4d = cv.triangulatePoints(np.eye(3,4),np.hstack((R,t)),img1ptsNorm.T,img2ptsNorm.T)
    pts3d = cv.convertPointsFromHomogeneous(pts4d.T)[:,0,:]
    
    return pts3d


def computeReprojection(trian, R, t, K):
    ''' trian - Matrix?
        R - Rotation Matrix
        t - Translation Matrix
        K - Camera Intrinsic Matrix
    '''
    outh = K.dot(R.dot(trian.T)+t)
    out = cv.convertPointsFromHomogeneous(outh.T)[:,0,:]
    return out


def find2D3DMatches(desc1, img1idx, desc2, img2idx,desc3, kp3, mask, pts3d):
    ''' 
    '''
    desc1_3D = desc1[img1idx][mask]
    desc2_3D = desc2[img2idx][mask]

    matcher = cv.BFMatcher(crossCheck=True)
    matches = matcher.match(desc3,np.concatenate((desc1_3D,desc2_3D),axis=0))
    print("Matches: ",len(matches))

    img3idx = np.array([m.queryIdx for m in matches])
    kp3_ = (np.array(kp3))[img3idx]
    img3pts = np.array([kp.pt for kp in kp3_])

    pts3didx = np.array([m.trainIdx for m in matches])
    pts3didx[pts3didx >= pts3d.shape[0]] = pts3didx[pts3didx>=pts3d.shape[0]] - pts3d.shape[0]

    pts3d_ = pts3d[pts3didx]
    print("pts3d_: ",pts3d_.shape)
    print("img3pts: ",img3pts.shape)

    return img3pts, pts3d_



def calibration(img1, img1path):
    '''returns what can be determined from img1 about the intrinsic camera matrix'''

    #No shear, 2d translation can be determined from img1.size
    #scaling can be determined from exif parameters or estimated
    h1,w1,c = img1.shape
    #print(h1,w1)
    #print(w1/h1)
    y0 = h1/2
    x0 = w1/2
    #print("Calibration Testing")
    #print(h1, w1)

    img = Image.open(img1path)
    img1_exif = img._getexif()
    if not(img1_exif is None):
            for tag,value in exifDataRaw.items():
                decoded = ExifTags.TAGS.get(tag,tag)
                print(decoded, value)
    else:
        print("No EXIF data found, using esitmate")
    
    #Estimation
    #Still within a good range of the given version
    #Handwave Version - weighted average of the dimensions
    ratio = x0/y0
    #print(ratio)
    val = (h1 + ratio*w1)/(ratio+1)
    #print(val)

    
    cali = np.array([[val,0,x0],[0,val,y0],[0,0,1]])
    #K = np.array([[2759.48,0,1520.69],[0,2764.16,1006.81],[0,0,1]])
    #Focal AngX = 122.2 deg Focal AngY = 139.97 deg

    return cali


def readFileSet(fileList):
    out = []
    sizeW = []
    sizeH = []
    fileList = open(fileList, "r")
    fList = fileList.read().split("\n")
    
    #print("List:", fList)
    for file in fList:
        #print(file[len(file)-4:])
        if(file[len(file)-4:] == ".png" or file[len(file)-4:] == ".jpg"): #Other types exist, just working with the two for now
            try:
                img = cv.imread(file)
                if(type(img) != np.ndarray): raise FileNotFoundError
                h,w,c = img.shape
                numPix = h*w

                inserted = False

                for i in range(len(out)):
                    if numPix > sizeW[i]*sizeH[i]:
                        inserted = True
                        out.insert(i, file)
                        sizeW.insert(i,w)
                        sizeH.insert(i,h)
                        break;

                if not inserted:
                    out.append(file)
                    sizeW.append(w)
                    sizeH.append(h)
                        
            
            except FileNotFoundError:
                print("{0} is not an image".format(file))
                continue #Just skip the file
        else:
            print("Invalid Extension")

    return out

def findHomography(img1, img2, matches):
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        matchesMask = masked.ravel().tolist()
        return H, matchesMask

#Esitmate focals from Homography
#FL = (d/2)/tan(a/2)
#FL = focal length, d = sensor diagonal, a= view diagonal
    #Height of view from identifying a fixed sized object, like a door?
    #Doors are typically 3ft wide and 80in tall - Regulations for Comercial buildings
    #Can be up to 32-48in wide and 60-96in tall
    #Then use the estimate to get the height and width of the frame (necessary for a)
    #Sensor Diagonal From CropFactor comparing to 35mm

def main():
    #Reading two images for reference__________________
    #img1 = cv.imread('./TestSet2/test5-2.jpg')
    #img2 = cv.imread('./TestSet2/test7-2.jpg')

    #Infering that Image1, is the nearly the same as the world coord system
    #img1 = cv.imread("test1.jpg")
    #img2 = cv.imread("test3(size).jpg")
    #path1 = "0004.jpg"
    #img1 = cv.imread("0004.jpg")
    #img2 = cv.imread("0005.jpg")

    #At this point, we should assume that every image in the file is of the same subject
    
    file = ""
    srcfile = None
    while len(file) == 0:
        file = input("Enter file with image names: ")

        try:
            srcfile = open(file, "r")
            break
        except FileNotFoundError:
            print("File: {0} could not be found".format(file))
            file = ""

    imgSet = readFileSet(file) #Returns the set of usable image paths, ordered by size

    if len(imgSet) < 2:
        print("Insufficient Number of Images")

    else:
        print(imgSet)
        #Do Process
        #Take First two and remove them to build a base image
        path1 = imgSet[0]
        path2 = imgSet[1]
        imgSet = imgSet[2:]

        img1 = cv.imread(path1)
        img2 = cv.imread(path2)

##    #Both images must be the same size for later processing
        h1,w1,c = img1.shape
        h2,w2,c = img2.shape
        h = max(h1,h2)
        w = max(w1,w2)

##    #Potential Solution to the size issue
##    #Can scaling one up to match the other in width, then stretching the other to match in height work better?
        #Will need to test

        #Simple Resizing
        #print("simple resize = both images", h, w)
        img1 = cv.resize(img1,(w,h), 0,0,cv.INTER_NEAREST)
        img2 = cv.resize(img2,(w,h),0,0,cv.INTER_NEAREST)

##        #More Complicated - It works slightly better, but costs accuracy on the base image
##        if h == h1: 
##            w2 = w2 * h1/h2
##            #h2 = h1
##        else:
##            w1 = w1 * h2/h1
##            #h1 = h2
##        print(h1,h2)
##        print(w1,w2)
##
##        w = int( max(w1,w2))
##        print("complex resize = both images", h,w)
##        img1 = cv.resize(img1,(w,h),0,0,cv.INTER_NEAREST)
##        img2 = cv.resize(img2,(w,h),0,0,cv.INTER_NEAREST)

        #Set Coloring to RGB rather than initial BGR
        img1 = img1[:,:,::-1]
        img2 = img2[:,:,::-1]

##    #Comparison Tool
##        fig,ax=plt.subplots(ncols=2,figsize=(9,4)) 
##        ax[0].imshow(img1)
##        ax[1].imshow(img2)
##        plt.show()

        kp1,desc1,kp2,desc2,matches=getImageMatches(img1,img2)

        img1pts, img2pts, idx1, idx2 = alignMatches(kp1,desc1, kp2, desc2, matches)
        
##    #RANSAC on two images
        MatF, mask = cv.findFundamentalMat(img1pts, img2pts, method = cv.FM_RANSAC)
        mask = mask.astype(bool).flatten()
        
##    #Needs to be adjustable to meet any camera specification
##    #getExif data then calculate if possible
        K = calibration(img1, path1)

        E = K.T.dot(MatF.dot(K))

    
##    #Rotation of Camera1, Rotation of Camera2, translation between the two
        R1, R2, t = extractPoses(E)

        t = t.reshape(3,1)

##    #Triangulation
        pts3d = getTriangulatedPts(img1pts[mask],img2pts[mask],K,R2,t)

##    #Camera Pose Disambiguation - Which R and t make the most sense
        _, Rn, tn, maskn = cv.recoverPose(E, img1pts[mask],img2pts[mask], K)

        pts3d = getTriangulatedPts(img1pts[mask], img2pts[mask],K,Rn,tn)

##    #PointCloud output
####    pcd = open3d.geometry.PointCloud()
####    pcd.points = open3d.utility.Vector3dVector(pts3d)
####    pcd.paint_uniform_color([1,1,0])
####    open3d.io.write_point_cloud("view2.ply",pcd)

##    #Up to now, we have been assuming that img1 is the world coordinate system

        #Strangeness of the testSet1 (Hollenbeck Hall) is due to multiple different image sizes
##        img1ptsRepro = computeReprojection(pts3d, np.eye(3,3),np.zeros((3,1)),K)
##        img2ptsRepro = computeReprojection(pts3d, Rn, tn, K)
##        x = img1pts[mask].shape[0] if img1pts[mask].shape[0] <= 50 else 50
##        visualizations.drawCorrespondences(img1,img1pts[mask],img1ptsRepro,plt, drawOnly = x)
##        visualizations.drawCorrespondences(img2,img2pts[mask],img2ptsRepro,plt, drawOnly = x)

##    #inputing a third(and later) views
##
        desc = [desc1,desc2]
        #print("Desc:",desc)
        kpts = [kp1, kp2]
        #print("KPTS:",kpts)
        Rs = [np.eye(3),Rn]
        #print("Rs:",Rs)
        ts = [np.zeros((3,1)), tn]
        #print("ts:",ts)

        #print("idxs:",idxs)


        print(pts3d.shape)
        print("Completed 2-View SFM")



        finalPts = []
        for imgNew in imgSet:
            img3 = cv.imread(imgNew)
            img3 = cv.resize(img3,(w,h),0,0,cv.INTER_NEAREST)
            img3 = img3[:,:,::-1]
            sift = cv.SIFT_create()
            kp3,desc3= sift.detectAndCompute(img3,None)

            print(desc[0].shape)
            print(mask.shape)
            print(desc[len(desc)-1].shape)
##            #Needs revised version to account for any number of previous iterations
            ##img3pts, pts3dpts, idx3 = find2D3DMatchesMod(desc, idxs, desc3, kp3, mask, pts3d)
            img3pts, pts3dpts = find2D3DMatches(desc1,idx1, desc2, idx2, desc3,kp3,mask,pts3d)

            retval, Rvec, tnew, mask3gt = cv.solvePnPRansac(pts3dpts[:,np.newaxis],img3pts[:,:np.newaxis],K,None,confidence=.99,flags=cv.SOLVEPNP_DLS)
            Rnew,_ = cv.Rodrigues(Rvec)
            tnew = tnew[:,0]

            tnew = tnew.reshape(3,1)


            #Confirmed Functionality for 3 views - Camera Visualization
##            fig = plt.figure(figsize=(9,6))
##            ax = fig.add_subplot(111, projection = '3d')
##            ax.set_xlabel('X')
##            ax.set_ylabel('Y')
##            ax.set_zlabel('Z')
##            print(Rs[0], ts[0])
##            visualizations.plotCamera(Rs[0],ts[0][:,0],ax,0.5,0.5, faceColor = 'blue')
##            visualizations.plotCamera(Rs[1],ts[1][:,0],ax,0.5,0.5, faceColor = 'green')
##            visualizations.plotCamera(Rnew, tnew[:,0], ax, 0.5,0.5, 'red')
##            plt.show()
            
            
            #Retriangulation
            ROld = Rs[0]
            TOld = ts[0]
            kpOld = kpts[0]
            descOld = desc[0]
            kpNew = kp3
            descNew = desc3


            
            accPts = []

            for j in range(len(Rs)):
                #print("Feature Matching...")
                matcher= cv.BFMatcher(crossCheck = True)
                matches = matcher.match(desc[j],desc3)
                matches = sorted(matches, key = lambda x:x.distance)
                imgOldPts, imgNewPts, _,_ = alignMatches(kpts[j],desc[j], kpNew, descNew, matches)

                #print("Pruning Matches...")
                MatF1, mask1 = cv.findFundamentalMat(imgOldPts, imgNewPts, method = cv.FM_RANSAC)
                mask1 = mask1.flatten().astype(bool)
                imgOldPts = imgOldPts[mask1]
                imgNewPts = imgNewPts[mask1]


                #print("Triangulating...")
                newPts = getTriangulatedPts(imgOldPts, imgNewPts, K, Rnew, tnew[:,np.newaxis],Rs[j],ts[j])


                finalPts.append(newPts)
                #print(accPts)
            #accPts.append(pts3d)


            #print(accPts)
            #pts3d = np.concatenate((accPts),axis=0)
            #print(pts3d.shape)
    finalPts.append(pts3d)
    print(np.concatenate((finalPts),axis=0).shape)
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.concatenate((finalPts),axis=0))

    pcd.paint_uniform_color([1,1,0])
    open3d.io.write_point_cloud("view8.ply", pcd)

    #Downsampling may be necessary (use open3d 0.15.1 documentation)
    #dpcd = pcd.voxel_down_sample(voxel_size = 0.05) #Voxels serve as buckets of points, size is in mm

    print("Calc. Normals")
    meshCreation.applyNormals(pcd)
    #Can now access the normals using pcd.normals
    print("Generating Mesh")
    
    open3d.io.write_triangle_mesh("mesh8a.ply",meshCreation.poissonMesh(pcd))
##    #print(pcd)
##
##    #open3d.visualization.draw_geometries([pcd]) #Not usable on this device
main()
