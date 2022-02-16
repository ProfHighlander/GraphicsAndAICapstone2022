#C Conrad
import cv2
import numpy as np
from matplotlib import pyplot as plt


#Objective: Cluster a file of image filenames into groups based on image subject
    #Batch Computation

#Larger Test Set Needed

#Class Definition

#Cluster - Group of images on the same subject
    #Constructor - Default and With one image
    #Iteration through the cluster next(), hasNext(), interation variable
    #Storage - *FileName, *KeyPoints1, *Descriptors,
    #Keypoints are just the keypoints and descriptors for each image, so it only has to be calc. once

#No Private attributes in Python, but can identify it in the API using _attributeName
#Can implement iteration using __next__() and __iter__() 
#Self is only for convention

class Cluster:
    def __init__(self):
        self._imageList = [] #Strings of filename
        self._KPList = []    #Lists of Keypoints (1 List per Image)
        self._DesList = []   #Lists of Descriptors (1 List per Image)
        self._size = 0       #Number of Images(len() of any of the lists)

        self._index = 0      #For the iteration function


#Getters
    def getImageList(self):
        return self._imageList

    def getImage(self, i):  #Exception when this occurs?
        return None if i>self._size-1 else self._imageList[i]
    
    def getKPList(self):
        return self._KPList

    def getKeyPoints(self, i):
        return None if i>self._size-1 else self._KPList[i]
    
    def getDesList(self):
        return self._DesList

    def getDescriptors(self, i):
        return None if i>self._size-1 else self._DesList[i]

    def size(self):
        return self._size

#No Setters Should be Used

#Mutators

    def append(self, img, kp, des):
        self._imageList.append(img)
        self._KPList.append(kp)
        self._DesList.append(des)
        self._size = self._size+1

        return True

    def remove(self, i):
        if(i <= self._size):
           
           self._imageList.remove(i)
           self._KPList.remove(i)
           self._DesList.remove(i)

           self._size = self._size-1
           return True
        else:
            return False        

#Iterator - Access the elements in some (insertion) order
        def __iter__(self):
            return self

        def __next__(self):
            if self._index == self._size-1:
                raise StopIteration
            self._index = self._index + 1
            return self._imageList[self._index], self._KPList[self._index], self._DesList[self._index]

#ToString() -- Not Functioning Properly
  #      def __str__(self):
  #          if(self._size == 0):
  #              return "Cluster: Size : 0"
 #           return "Cluster: " + self._imageList[0] + " Size: " +self._size
        
    



def main():
#Main
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

    if(srcfile != None):
        fileList = srcfile.read().split("\n")
        sift = cv2.SIFT_create() #Using SIFT for matching bonus, can revert to ORB if patent is an issue
        #create cluster array
        clusterList = []
        for image in fileList :
            print("** {0} **".format(image))
            #Can just index the last 4 characters srcfile[len(srcfile)-5:]==".png" or srcfile[len(srcfile)-5:]==".jpg"
            if(image[len(image)-4:]==".png" or image[len(image)-4:] == ".jpg"):
                print(image)
                newImg = 0
                newkp = 0
                newDes = 0
                try:
                    img = cv2.imread(image)
                    print(type(img))
                    if(type(img) != np.ndarray): raise FileNotFoundError
                    newImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Will only need to calc once, shows in greenscale, but not an issue
                    newkp = sift.detect(newImg,None)
                    newkp, newDes = sift.compute(newImg,newkp)


                    
                except FileNotFoundError:
                    print("File: {0} could not be found".format(image))
                    continue

                



                if(len(clusterList) == 0):
                    newC = Cluster()
                    newC.append(image,newkp,newDes) #TBD
                    clusterList.append(newC)
                    continue                        #Jump to the next image                  

                
                clustInsert = 0
                for clust in clusterList :
                    failout = False
                    for i in range(clust.size()) :
                        if failout:
                            break
                        
                        print(i) #Which member of the cluster are we attempting to match with
                    
                        #Access the keypoints and descriptors for the testImg
                        testImg = cv2.cvtColor(cv2.imread(clust.getImage(i)), cv2.COLOR_BGR2GRAY) 
                        testkp = clust.getKeyPoints(i)   
                        testDes = clust.getDescriptors(i)      
                        
                        #marked=cv2.drawKeypoints(testImg ,testkp ,testImg ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        #plt.imshow(marked),plt.show()

                        bf = cv2.BFMatcher()
                        
                        #Allows it to be switched if necessary
                        #matches = bf.knnMatch(newDes,testDes, k=2)
                        matches2 = bf.knnMatch(testDes, newDes, k=2) 
                        

                        #good = []
                        #for m,n in matches:
                        #    if m.distance < 0.8*n.distance:
                        #        good.append([m])
                        #matches = np.asarray(good)

                        good = []
                        for m,n in matches2:
                            if m.distance < 0.8*n.distance:
                                good.append([m])
                        matches2 = np.asarray(good)



                        #Testing and Demonstration
                       # print("new->test: {0} \ntest->new:{1}".format(len(matches),len(matches2))) 

                       # img3 = cv2.drawMatchesKnn(newImg,newkp,testImg,testkp,matches, testImg, flags=2)
                       # plt.imshow(img3),plt.show()

                        
                       # img4 = cv2.drawMatchesKnn(testImg,testkp,newImg,newkp,matches2, testImg, flags=2)
                       # plt.imshow(img4),plt.show()

                        #which to use? Max Min, test->new or new->test
                        #new->test : All images added when failonce
                        
                        #test->new : Clustered Correctly when failonce is used


                        if len(matches2[:,0]) < 4:
                            failout = True

                            
                    if not failout:
                        clust.append(image,newkp,newDes)
                        clustInsert = clustInsert + 1

                        
                if clustInsert == 0:
                    newC = Cluster()
                    newC.append(image,newkp,newDes) #TBD
                    clusterList.append(newC)
                    continue    
        
        print(len(clusterList)) #Noticeable pauses when calc the keypoints
        for i in clusterList :
            print(i)
            for j in i.getImageList() :
                print(j)


    
#Ask for file to insert images from

#While the has another line
    #Verify that it is an accessible file 
    #Find the cluster(s) it belongs to
        #Find keypoints and descriptors for new image
        #Locate them for the old ones
        #Find the matches using knnMatch
        #Reduce to sufficient matches using ratio test
        #Are there enough matches at this point?
            #Add to Cluster
            #Increment the number of successful matches
            #Else move onto the next cluster
        #If at end of cluster, check if every image has a match to the new one
            #So, add it to the cluster and increment the number of clusters
        
        #Check the next cluster - Must be some way to use dynamic programming to speed up the program
        
    #If none, create a new cluster for this image

#Testing - For each cluster, print all fileNames of the cluster*
#Visual comparison


main()
