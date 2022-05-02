#Main
#GUI 
from tkinter import *
from tkinter import messagebox as mbx
import tkinter.ttk as ttk
from time import strftime
from tkinter.filedialog import askopenfilenames
from tkinter.filedialog import asksaveasfilename
from tkinter.filedialog import askopenfilename

#Matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

#Opencv
import numpy as np
import open3d

#Homebrew
import calib 
import fileSave as fs 
import SFM



def insertImgs(imgList):
        #Ask the user for which images they want to add
        file = askopenfilenames( filetypes =[("All","*.*"),('JPEG', '*.JPG'), ("JPEG",'*.jpg'),("PNG",".png")])
        if file is not None:
            for i in file:
                    imgList.append(i)
        
def saveModel(xyzModel):
        #Can add mesh file types later on, fix the mesh module
        if(type(xyzModel[0]) != np.ndarray):
                mbx.showerror("No Model", "There is currently no model loaded.")
                return
        
        files = [("All","*.*"),("Polygon", "*.ply"), ("Point Cloud Data","*.pcd"),("XYZ","*.xyz")]
        file = asksaveasfilename(filetypes = files, defaultextension=files)
        
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(xyzModel[0])
        open3d.io.write_point_cloud(file, pcd)



def printList(lis, title, select = False,rem = False, dest = None): #Now make it augmentable for both cameras and imgs
   #Print the given list in a form dependent on the subject
    wind = Tk()
    wind.title(title)
    wind.resizable(width=1, height=1)

    label = Label(wind, text = title)
    label.pack(side = "top")
    
    frame = Frame(wind)
    frame.pack(fill=BOTH, expand = True,padx = 10, pady = 10)

    tv = ttk.Treeview(frame)
    if(title == "Cameras"):
            
            tv['columns']=('Name', 'Make', 'Model')
            tv.column('#0', width=0, stretch=NO)
            tv.column('Name', anchor=CENTER, width=80)
            tv.column('Make', anchor=CENTER, width=80)
            tv.column('Model', anchor=CENTER, width=80)
            tv.heading('#0', text='', anchor=CENTER)
            tv.heading('Name', text='Name', anchor=CENTER)
            tv.heading('Make', text='Make', anchor=CENTER)
            tv.heading('Model', text='Model', anchor=CENTER)
            for i in range(len(lis)):
                    #print(lis[i])
                    tv.insert("","end",text = i,values=lis[i][0:3])

    else:
        tv['columns']=("File")
        tv.column('#0', width=0, stretch=NO)
        tv.column('File', anchor=CENTER, width=180)
        for i in range(len(lis)):

                #print(lis[i])
                x = lis[i].rfind("/")
                x = x if x > -1 else 0
                tv.insert("", 'end', text =i,values =(lis[i][x:]))#Orig values = (lis[i][0], lis[i][1], lis[i][2])
        
    vscrl = ttk.Scrollbar(frame,orient='vertical',command = tv.yview)
    vscrl.pack(side="left")

    hscrl = ttk.Scrollbar(frame,orient='horizontal',command = tv.xview)
    hscrl.pack(side = 'bottom')

    tv.configure(yscrollcommand=vscrl.set)
    tv.configure(xscrollcommand=hscrl.set)


    def loadDest(dest): #This loads it correctly, just need to associate the rest of the information

            itm = tv.item(tv.focus())["text"]
            dest[0] = lis[itm]
            #print(dest[0])
            wind.destroy()
            return
            
    def removeItem():
            dest = tv.item(tv.focus())["text"]
            lis.remove(lis[dest])
            wind.destroy()
            return

    btFrame = Frame(wind)
    btFrame.pack(side = "bottom")

    if select:
        btn = Button(btFrame, text = 'Select',command = lambda:loadDest(dest))
        btn.pack(side = 'left')

    elif rem:
        btn = Button(btFrame, text = "Remove",command = removeItem)
        btn.pack(side = 'left')

    tv.pack(fill=BOTH,expand= True)       

    btn2 = Button(btFrame,text="Close",command = wind.destroy)
    
    btn2.pack(side = 'right')
    wind.mainloop()
    return None


def selectCam(camFile, currCam): #***Major Issue - Try currCam as dest
        #List all options, then let the user click then confirm the option they want
        select1=[-1]
        cams=fs.findAll(camFile)
        
        if len(cams)==0:
                mbx.showerror("No Calibrations", "There are currently no calibrated cameras saved, go to the Calibrate Camera menu to add one.")
                return
                
        #printList([cams[i][0:3] for i in range(len(cams))], "Cameras", select = True,rem = False,dest=select1)
        printList(cams, "Cameras", select = True,rem = False,dest=currCam)
        #print(select1)
        #cam = cams[select1[0]]
        #currCam[0] = [cam[0],cam[1],cam[2],cam[3],cam[4],cam[5]]
        #print(currCam)



def listClear(lis,title):
        #Clears some list
        x = mbx.askyesno("ClearList?", "Do you want to clear the current {0}?".format(title))
        if x:
                lis.clear()


def calibrateCamera(imgSet, outputFile, currCam):
        #Verify Calib
        if len(imgSet) == 0:
                mbx.showerror("Insufficient Images", "The current image set has no images.")
                return
        mess, idx, out = calib.verifyCalib(imgSet)
        name_var = StringVar()
        def submit():

                currCam[0] = [name_entry.get(),idx[0],idx[1],mtx,idx[2],(idx[3],idx[4])] #Successfully sets the current camera
                fs.addToFile(outputFile,currCam[0][0],idx[0],idx[1],mtx,idx[2],(idx[3],idx[4]))
                wind.destroy()
                
        
        if mess == 4:
                #Insufficient Number of Images
                mbx.showerror("Insufficient Images", "The current image set has {0} useable images. We need at least 10 to calibrate the camera.".format(len(out)))
                return
        elif mess != 0:
                #Something is wrong about a specific img (could list all of them)
                issues = ["NumImages", "Unavailable Images", "Inconsistent EXIF Availability","Inconsistent EXIF Values"]
                mbx.showerror("Calibration Verification Failed", "The current image set has {0}. We noticed the issue on {1}".format(issues[mess], imgSet[idx]))
                return
        else:
                #In later stages, could ensure that the user doesn't repeat nicknames
                mtx = calib.calibrate(out)

                wind = Tk()
                wind.title("Register Camera")
                wind.geometry("200x50")

                name_label = Label(wind, text = 'Name:', font=('calibre',10, 'bold'))

                name_entry = ttk.Entry(wind, font=('calibre',10,'normal'))
                name_entry.focus()

                sub_btn=Button(wind,text = 'Submit', command = submit)
  

                name_label.grid(row=0,column=0)
                name_entry.grid(row=0,column=1)

                sub_btn.grid(row=1,column=1)

                wind.mainloop()

def fullRun(imgList, currCam):

        #print(currCam) #How to get this to update correctly
        #print(imgList)
        print("Verifying")
        x, y =SFM.verify_Images(imgList)
        print(x,y)
        if currCam[0] == -1:
                mbx.showerror("No Calibrated Camera Selected", "Select a calibrated camera to use.")
                return []

        elif x != 0: #Something is wrong with the images
                mbx.showerror("Image Inconsistency", "Some images are not the same size. We noticed {0} distinct groups.".format(y))
                return []
                
        else:
                xyz = SFM.structure_from_motion(imgList, currCam[0])
                return xyz
                



def main():

#Update the matplotlib enviroment
    def update_Fig(newX,newY,newZ):
            ax.cla()
            ax.set_box_aspect([1,1,1])

            avgX = np.asarray(np.average(newX))
            avgY = np.asarray(np.average(newY))
            avgZ = np.asarray(np.average(newZ))
            #ax.set_anchor((avgX.item(),avgY.item()))


            print("Center:",(avgX,avgY,avgZ)) #Center on the average point
            ax.scatter3D(avgX.item(),avgY.item(),avgZ.item(), marker='^', color = 'green')
            ax.scatter3D(newX,newY,newZ, s=1)

            ax.autoscale(enable=False,axis='both')
            ax.set_xbound(avgX.item()-1, avgX.item()+1)
            ax.set_ybound(avgY.item()-1, avgY.item()+1)
            ax.set_zbound(avgZ.item()-1, avgZ.item()+1)
            canvas.draw()
            #print(ax.get_xlim())
            #print(ax.get_xbound())
            #print(ax.get_aspect())


#Open a model
    def openModel():
        files = [("All","*.ply *.pcd *.xyz"),("Polygon", "*.ply"), ("Point Cloud Data","*.pcd"),("XYZ","*.xyz")]
        file = askopenfilename(filetypes = files)
        
        if len(file) != 0:
                currModel[0] = np.asarray(open3d.io.read_point_cloud(file).points)
                update_Fig(currModel[0][:,0],currModel[0][:,1],currModel[0][:,2])


    def runM(currModel):
            xyz = fullRun(imgList, currCam)
            
            if len(xyz) != 0:
                    currModel[0] = np.asarray(xyz)
                    
                    update_Fig(currModel[0][:,0],currModel[0][:,1],currModel[0][:,2])
 
    def clearAll(imgList, currModel, currCam, ax):
        save = mbx.askyesnocancel("Save All", "Would you like to save before starting a new file?")
        print(save)
        if save:
                saveModel(currModel)
        if not save is None:
                imgList.clear()

                currModel = [-1]

                currCam = [-1]

                ax.cla()
                canvas.draw()
                
#Initialize GUI root

    root = Tk()
    root.title("MENU")
    root.geometry("960x540")

    frame = Frame(root)
    frame.pack(fill=BOTH, expand = True)


    currCam = [-1] #Need to get it to update from list selection correctly
    imgList = []
    save = "saveState2.txt" #Can Let User Set the file name later
    currModel = [-1]

    menubar = Menu(root)
    file = Menu(menubar,tearoff = 0)
    menubar.add_cascade(label = 'File', menu = file)


#File Menu
    file.add_command(label ='New File', command = lambda: clearAll(imgList, currModel, currCam, ax)) #Clear camera, imgs, and model information (after asking if the user would like to save
    file.add_command(label ='Open...', command = openModel)
    file.add_command(label ='Save', command = lambda: saveModel(currModel))
    file.add_separator()
    file.add_command(label ='Exit', command = root.destroy)

#Run Model Menu
    run = Menu(menubar,tearoff = 0)
    menubar.add_cascade(label = "Run Model", menu = run)
    run.add_command(label = "Run Model", command = lambda: runM(currModel)) #Run from SFM file - Last one to wire
    run.add_command(label = "Select Camera",command= lambda :selectCam(save, currCam)) #Print camera list - This is the probelm
    run.add_command(label = "Calibrate Camera",command=lambda: calibrateCamera(imgList, save, currCam)) #Run from calib and saveFile file

#Images Menu -- All works
    imgs = Menu(menubar,tearoff = 0)
    menubar.add_cascade(label = "Images", menu = imgs)
    imgs.add_command(label = "Insert Images", command = lambda:insertImgs(imgList))
    imgs.add_command(label = "Current Images",command = lambda :printList(imgList, "Images",select=False,rem=False)) #Print img list
    imgs.add_command(label = "Remove Image",command = lambda :printList(imgList, "Images",rem=True))
    imgs.add_command(label = "Clear Images",command = lambda : listClear(imgList, "Image List")) #Clear img list

        

#Initialize the matplotlib 3d plot
    fig = Figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(left=-5,right=5)
    ax.set_ylim(bottom=-5,top=5)
    ax.set_zlim(bottom=-5,top=5)

#Convenience of Matplotlib tools
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
    toolbar.update()

    toolbar.pack(side=BOTTOM, fill = X)
    canvas.get_tk_widget().pack(side = TOP, fill = BOTH, expand = 1)
    
    root.config(menu = menubar)

    root.mainloop()
    #print(currModel[0])

main()
