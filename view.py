import numpy as np
import cv2 as cv

#Add distortion and focal parameters

class View:
    def __init__(self):
        self.keypoints = []
        self.desc = []
        self.R = [[0,0,0],[0,0,0],[0,0,0]]
        self.t = [[0],[0],[0]]
        self.path = ""
        self.matchSet = []
        self.K = [[0,0,0],[0,0,0],[0,0,0]]
        self.distort = []
        self.focal = 0

    def __init__(self, p, k, d):
        self.keypoints = k
        self.desc = d
        self.R = [[0,0,0],[0,0,0],[0,0,0]]
        self.t = [[0],[0],[0]]
        self.path = p
        self.matchSet = []
        self.K = [[0,0,0],[0,0,0],[0,0,0]]
        self.focal = 0
        self.distort = []

    def getKeypoints(self):
        return self.keypoints

    def getDescriptors(self):
        return self.desc

    def getPath(self):
        return self.path

    def getRotation(self):
        return self.R

    def getTranslation(self):
        return self.t

    def getMatchSet(self):
        return self.matchSet

    def getMatches(self, partnerId):
        for i in range(len(self.matchSet)):
            if partnerId == self.matchSet[i][1]:
                return self.matchSet[i]
        return None

    def getK(self):
        return self.K

    def getDistort(self):
        return self.distort

    def getFocal(self):
        return self.focal

    def setKeypoints(self, keypointsNew):
        self.keypoints = keypointsNew

    def setDescriptors(self,descNew):
        self.desc = descNew

    def setPath(self, p):
        self.path = p

    def setRotation(self, Rnew):
        self.R = Rnew

    def setTranslation(self, tnew):
        self.t = tnew

    def setK (self, newK):
        self.K = newK

    def setDistort(self, d):
        self.distort = d

    def setFocal(self, f):
        self.focal = f

    def appendMatchSet(self, partnerId, matches):
        #Check if partner already has a registered set of matches
        exists = False
        for i in range(len(self.matchSet)):
            if partnerId == self.matchSet[i][1]:
                exists = True
                self.matchSet[i][0] = matches
        if not exists:
            self.matchSet.append([matches,partnerId])

        return exists #Whether or not there was something replaced


    #We already know what the kp and desc are for self and partner



    def triangulate(self, partner, K, mask, Rbase=None, tbase=None):
        '''Using partner's R and t
            Mask - result of align matches'''

        if Rbase is None: 
            Rbase = np.eye(3,3) 
        if tbase is None: 
            tbase = np.zeros((3,1))

        t = partner.getTranslation()
        R = partner.getRotation()

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
        
        

        




        

    
    
        
