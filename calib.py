#Finds the calibration matrix for a camera
#All images must be taken at the same focal length/f-number and image size

import cv2 as cv
import numpy as np
from PIL import ExifTags
from PIL import Image
import os

import fileSave as sf

def verifyCalib(imgSet):
    #Filter out images that do not have the chessboard pattern - note the idxs
    #Returns true if all images are the same size, focalLen, make and Model, false otherwise
    imgSetOut =[]
    
    baseFocal = 0
    baseW = 0
    baseH = 0
    baseMake = ""
    baseModel = ""
    none = False

    for i in range(len(imgSet)):
        exif = Image.open(imgSet[i])._getexif()
        x = cv.imread(imgSet[i])
        if x is None:
            return 1, i,0 #Image Not available
        if exif is None:
            if baseH == 0:
                none = True
                baseH = x.shape[0]
                baseW = x.shape[1]
                baseMake = "N\A"
                baseModel = "N\A"
            else:
                if not none:
                    return 2, i,0 #Inconsistent Metadata Availability
                else:
                    x = cv.imread(imgSet[i])
                    if x is None:
                        return 1, i,0 #Image Not available
                    if x.shape[0] != baseH or x.shape[1] != baseW:
                        return 3, i,0 #Inconsistent Metadata values
        else:
            f = exif[37386]
            ma = "N/A" if len(exif[271])==0 else exif[271]
            mo = "N/A" if len(exif[272])==0 else exif[272]
            if baseH == 0:
                baseFocal = f
                baseMake = ma
                baseModel = mo
                baseH = x.shape[0]
                baseW = x.shape[1]
            else:
                if f != baseFocal or ma != baseMake or mo != baseModel or x.shape[0] != baseH or x.shape[1] != baseW:
                    return 3,i, 0

        grey = cv.cvtColor(x, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(grey, (7,6), None)
        if ret:
            imgSetOut.append(imgSet[i])
    print(imgSetOut)
    if len(imgSetOut) < 10:
        return 4, 0, imgSetOut #Lenght of calibration set insufficient - allow override

    
                
    return 0, (baseMake, baseModel, baseFocal, baseH, baseW), imgSetOut #Success
                


def calibrate(calibSet):

    #Finding the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #For a 7x6 set of corners
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    images = []

##    make = "ERR"
##    model = "ERR"
##    focal = -1
##    imgS = (0,0)

##    x,y = verifyCalib(calibSet)
##    if x != 0:
##        return 1
##    else:
##        make = y[0]
##        model = y[1]
##        focal = y[2]
##        imgS = (y[3],y[4])

    for fname in calibSet:
        #print(fname)
        img = cv.imread(fname)
        imgShape = img.shape[0:2]
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(grey, (7,6), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(grey, corners, (11,11), (-1,-1), criteria)

            imgpoints.append(corners)

            #cv.drawChessboardCorners(img, (7,6), corners2, ret) #Visualization

            #cv.imshow('Corners'+fname, img)
            #cv.waitKey(0) #Closes window after about 1 sec

            #cv.destroyAllWindows()

    #Calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, grey.shape[::-1], None, None)
    #print(ret)

    #print("MTX",mtx)


    #print(rvecs)
    #file = "saveState.txt" #Allow to be set by the user?


##    sf.addToFile(file,name,make,model,mtx,focal,imgS)
    return mtx
