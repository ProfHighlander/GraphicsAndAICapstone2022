from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d

#Visualization Tool - Will be removed to testFile
def plotCamera(R,t,ax, scale=.5,depth = .5, faceColor='grey'):
    C = -t #Cam center

    axes = np.zeros((3,6))
    axes[0,1],axes[1,3],axes[2,5]=1,1,1

    axes = R.T.dot(axes)+C[:,np.newaxis]

    ax.plot3D(xs=axes[0,:2],ys=axes[1,:2], zs=axes[2,:2], c= 'r')
    ax.plot3D(xs=axes[0,2:4],ys=axes[1,2:4], zs=axes[2,2:4], c= 'g')
    ax.plot3D(xs=axes[0,4:],ys=axes[1,4:], zs=axes[2,4:], c= 'b')

    pt1 = np.array([[0,0,0]]).T
    pt2 = np.array([[scale,-scale,depth]]).T
    pt3 = np.array([[scale,scale,depth]]).T
    pt4 = np.array([[-scale,-scale,depth]]).T
    pt5 = np.array([[-scale,scale,depth]]).T
    pts = np.concatenate((pt1,pt2,pt3,pt4,pt5),axis=-1)

    pts = R.T.dot(pts)+C[:,np.newaxis]
    ax.scatter3D(xs=pts[0,:],ys=pts[1,:],zs=pts[2,:], c = 'k')

    verts = [[pts[:,0],pts[:,1],pts[:,2]],[pts[:,0],pts[:,2],pts[:,-1]],
             [pts[:,0],pts[:,-1],pts[:,-2]],[pts[:,0],pts[:,-2],pts[:,-1]]]

    ax.add_collection3d(Poly3DCollection(verts,facecolors=faceColor,linewidths=1,edgecolors='k',alpha=0.25))

#Visualization Tool, will be removed to testFile
def drawCorrespondences(img, ptsTrue, ptsRepro, ax, drawOnly = 50):
    ''' Comparison of reprojected points and actual keypoints on the image
    '''
    ax.imshow(img)

    randidx = np.random.choice(ptsTrue.shape[0],size=(drawOnly,),replace = False)
    ptsTrue_, ptsRepro_ = ptsTrue[randidx], ptsRepro[randidx]
    colors = np.random.rand(drawOnly,3)

    
    ax.scatter(ptsTrue_[:,0],ptsTrue_[:,1],marker='x',c=colors)
    ax.scatter(ptsRepro_[:,0],ptsRepro_[:,1],marker='o',c=colors)
    plt.show()

def visCameraAmbiguity(R1, R2, t):
    ''' Plots the comparision of the various combinations of R and t for two cameras
'''
    count = 0
    colors = ["red","orange","yellow","green"]
    for R_ in [R1, R2]:
        for t_ in [t,-t]:
            fig = plt.figure(figsize=(9,6))

            ax = fig.add_subplot(111,projection = "3d")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plotCamera(np.eye(3,3),np.zeros((3,)),ax, 0.5,0.5,"blue") #The Standard
            plotCamera(R_, t_,ax,0.5,0.5,colors[count])                 #The Unknown

            plt.show()
            count = count + 1;

def visCameraCorrespondences(R1, R2, t, imgpts1, imgpts2, mask, K):
    configSet = [None,None,None,None]
    colors = ["red","orange","yellow","green"]

    configSet[0] = (R1,t,getTriangulatedPts(img1pts[mask],img2pts[mask],K, R1,t))
    configSet[1] = (R1,-t,getTriangulatedPts(img1pts[mask],img2pts[mask],K, R1,-t))
    configSet[2] = (R2,t,getTriangulatedPts(img1pts[mask],img2pts[mask],K, R2,t))
    configSet[3] = (R2,-t,getTriangulatedPts(img1pts[mask],img2pts[mask],K, R2,-t))

    count = 0
    for cs in configSet:
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        print(cs[1])

        plotCamera(np.eye(3,3),np.zeros((3,)),ax,scale=5,depth=5,faceColor='blue')
        plotCamera(cs[0],cs[1][:,0],ax,scale=5,depth=5,faceColor = colors[count])
        count = count +1

        pts3d = cs[-1]
        ax.scatter3D(pts3d[:,0],pts3d[:,1],pts3d[:,2])

        ax.set_xlim(left=-50,right=50)
        ax.set_ylim(bottom=-50,top=50)
        ax.set_zlim(bottom=-50,top=50)

        plt.show()

def visCameraPositions3(Rn, tn, Rnew, tnew):
    ''' Rn, tn - Rotation and Translation Matrices for the second image
        Rnew, tnew - Rotation and Translation Matrices for the third image
'''
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    tnew = tnew.reshape(3,1)

    plotCamera(np.eye(3,3),np.zeros((3,)),ax,0.5,0.5, faceColor = 'blue')
    
    plotCamera(Rn,tn[:,0],ax,0.5,0.5, faceColor = 'green')
    #plt.show()
    
    plotCamera(Rnew,tnew[:,0],ax,0.5,0.5,faceColor = 'red')

    plt.show()
    


