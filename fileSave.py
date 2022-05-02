import numpy as np

def readFile(file):
    '''Returns a list of strings from file'''
    fileR = open(file, "r")
    x = fileR.read().split("\n")
    fileR.close()
    i = 0
    while i < len(x):
        if len(x[i]) == 0:
            x.remove(x[i])
        else:
            i = i+1
    #print("FILE:",x)
    return x

def removeFromFile(file, data, count):
    #Data as the result of convert
    out = data[0] + "," + data[1] + "," + data[2] + "," + str(data[3].ravel()).replace("\n","")[1:-1] +"," + str(data[4]) +"," +str(data[5])
    #print("Q:",out)
    fileW = open(file,'r')
    x = fileW.read().split("\n")
    fileW.close()
    fileW = open(file,'w')
    #print("X:",x)
    

    str1 = ""
    for i in x:
        if  count > 0 and hamming(i,out) == 0:
            count -=1
            continue
        str1 = str1+i+"\n"

    #print("OUT:",str1)
    fileW.write(str1[:-1])
    fileW.close()
    return count
    
    


def hamming(train, query):
    return sum(c1 != c2 for c1,c2 in zip(train, query))
    
def findMakeModel(file, make, model):
    '''Returns a list of cameras that match the make, model description given'''

    lis = readFile(file)
    
    MINCOUNT = 1
    out = []
    reject = []
    idx = -1
    ham = -1
    for i in range(len(lis)):
        lisI = lis[i].split(",")
        if make.lower() == lisI[1].lower() and model.lower() == lisI[2].lower():
            out.append(lisI)

        else:
            reject.append(lisI)
            if idx == -1:
                idx = 0
                ham = sum(hamming(make.lower(), lisI[1].lower()), hamming(model.lower(), lisI[2].lower()))
            else:
                h2 = sum(hamming(make.lower(), lisI[1].lower()), hamming(model.lower(), lisI[2].lower()))
                if h2<ham:
                    ham = h2
                    idx = len(reject)-1

    if len(out) < MINCOUNT:
        out.append(reject[idx])

    return [convert(x) for x in out]

def findNickName(file, nickname):
    lis = readFile(file)
    
    MINCOUNT = 1
    out = []
    reject = []
    idx = -1
    ham = -1
    for i in range(len(lis)):
        #print(lis[i])
        lisI = lis[i].split(",")
        if nickname.lower() == lisI[0].lower():
            out.append(i)

        else:
            reject.append(lisI)
            if idx == -1:
                idx = 0
                ham = hamming(nickname.lower(),lisI[0].lower())
            else:
                h2 = hamming(nickname.lower(),lisI[0].lower())
                if h2<ham:
                    ham = h2
                    idx = len(reject)-1

    if len(out) < MINCOUNT:
        out.append(idx)
        print("Closest:",reject[idx])
    x = [convert(lis[i]) for i in out]
    #print(x)
    return x
def findAll(file):
    lis = readFile(file)
    #print(len(lis))
    #print(lis)
    if(len(lis) != 0):
        x = [convert(lis[i]) for i in range(len(lis))]
    else:
        x = [] 
    return x

def convert(str1):
    #Converts str1 from file format to list of values
    #Could simplify this further using tKinter
    lis = str1.split(",")

    lis3 = lis[3][1:-1].split(" ")
    lis3Out = ""
    #print(lis3)
    
    lis3 = list(filter("".__ne__,lis3))
    for i in range(3):
        lis3Out = lis3Out + lis3[3*i]
        for j in range(2):
            lis3Out = lis3Out + " " + lis3[3*i+j+1]
        lis3Out = lis3Out + ";"

    #print(lis3Out)
    lis[3] = np.matrix(lis3Out[:-1])

    lis[4] = float(lis[4])
    #print(lis[5:7])
    lis[5] = (int(lis[5][1:]), int(lis[6][:-1]))
    lis = lis[:6]
    #print(lis)
    return lis
                         

def addToFile(file, name, make, model, intrinsic, focal, initSize):
    #Ignoring Distort for now
    fileA = open(file, "a")
    out = name + "," + make + "," + model + "," + str(intrinsic.ravel()).replace("\n"," ") +"," + str(focal) +"," +str(initSize)+ "\n"
    fileA.write(out)
    fileA.close()
    return 0


                    
        
