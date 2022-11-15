#MD

'''
Line Detector so we can use lines and points from image to spot parking spots
there is a Space class that will save all the data 
a line Class that save line points and other
and we use openCv to detect lines 
        -----------------
once the lines and points are collected we should compute them to get parking
spaces that we could shape with the time ( cars parking in )
        ---------

'''

from audioop import cross
from cmath import cos
import copy
import math
import time
import cv2
from cv2 import CALIB_USE_QR
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import keyboard
import ast
import random
from scipy import stats

class Parking():
    def __init__(self,lines) -> None:
        self.lines=[]
        self.graphStrcut={}
        self.intersection={}
        for idc,l in enumerate(lines) :
            leninter=len([intersect(l,l2) for l2 in lines if intersect(l,l2)!=None])
            self.lines.append(Line(coords=l,ID=idc,inter=leninter))
        self.lines.sort(key=lambda k: k.lensections,reverse=True)
        dumplines = self.lines.copy()
        for we,l in enumerate(self.lines):
            if l not in dumplines:continue
            print(l)

            interpt=[intersect(l.coord,l2.coord) for l2 in self.lines ]
            interid=[i for i in range(len(interpt)) if interpt[i]!=None ]
            if len(interid)==0:continue
            
            for d in interid :  
                try:

                    dumplines.remove(self.lines[d]) 
                    dumplines.remove(self.lines[we]) 
                except:pass
            self.graphStrcut[we]=interid
            self.intersection[we]={}
            for d in interid:self.intersection[we][d]=interpt[d]
        
            
            


        
class Line():
    def __init__(self,coords,ID,inter) -> None:
        self.coord=coords
        self.LineID=ID
        self.intersections=[]
        self.lensections=inter
        self.linelen=round(math.sqrt((coords[0][0]-coords[1][0])**2+(coords[1][0]-coords[1][1])**2),3)

    def __repr__(self) -> str:
        return f'Line {self.LineID} cross :{self.lensections} long : {self.linelen} px '
    def printIntersections(self):
        print(f'>====Line{self.LineID} , {self.lensections} Intersections :')

        if self.lensections==0:
                print('>=No intersection')
        else:
            for dg,i in enumerate(self.intersections):
                if i==None:continue
                print(f'=line {dg} : {i}')
                



def intersect(l1,l2):
    x1,y1 = l1[0]
    x2,y2 = l1[1]
    x3,y3 = l2[0]
    x4,y4 = l2[1]
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)

	

######################################### MAIN #############################################


def detectLines(file):
    #store the y=mx+b lines
    global space
    bigLines=[]
    #we read the frame/picture
    img = cv2.imread(file)
    img=cv2.resize(img,[640,480])
    #filter
    gr=cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    #filter to lower details
    blur_gray = cv2.GaussianBlur(src=gr, ksize=(5, 5), sigmaX=0)
    #threshold used for canny edges detection
    thresholds=[20, 200,]
    #state str
    state=f'{file},threshold: {thresholds[0]}-{thresholds[1]}'
    #clear plt for next picture
    plt.clf()   
    #set new plt title
    plt.title(state)

    #canny edge detection filter 
    edges = cv2.Canny(image=blur_gray, threshold1=thresholds[0], threshold2=thresholds[1], apertureSize=3)
    #print state of actual image
    print(state)
    #lsd = cv2.createLineSegmentDetector(0)   --  never worked with this one --
    #lines = lsd.detect(edges)[0] #Position 0 of the returned tuple are the detected lines

    #Detect lines in the image with HoughLinesP !! the parameters are not optimal !!
    #cv.HoughLinesP(	image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]	) ->	lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 50)  # !  Lines= Detected lines
    #get the size of the image
    i,j= img.shape[:2]
    #Canny filter gives us a picture of active(255)/not active(0) pixels in the picture
    #so we can iterate in the picture to spot the active pixels 
    lescord=[(x,y) for x in range(i) for y in range(j) if edges[x,y]==255]
    # we create a matrix with the size of the picture (i x j) init to 0
    space=np.full((i+10,j+10),0)
    #we mark our active pixels as 1 in the space matrix
    for x,y in lescord:
        space[x,y]=1
    #we put our picture in matplotlib board

    plt.imshow(img)
    #list where we save the line equations 
    bigLines=[]
    #(new idea) list of intersection points
    comp=[]
    #we iterate detected lines
    splittedLines=[]
    for ff,line1 in enumerate(lines):
        #line1 => [[line]] so we convert to line1 => [[x0,y0],[x1,y1]]
        line1=[line1[0][:2],line1[0][2:]]
        #calculate the slope of the line:    det = (y1-y0)/(x1-x0)
        #   if same x absis slope is vertical :)  det=0 
        if line1[1][0]!=line1[0][0]:
            det = (line1[1][1]-line1[0][1])/(line1[1][0]-line1[0][0])
        else:det=0
        #update title with plotting line number
        plt.title(f'Low Detail image')
        #create list of number from 0 to picture width (x)
        x = np.linspace(0,j,j)
        #compute y images ( height)
        y= det*(x-line1[1][0])+line1[1][1]
        # absis origin point y0=m*(x1-0)+y1    
        y0= det*(0-line1[1][0])+line1[1][1]
        #new x y list to filter points
        nx=[]
        dx=[]
        ny=[]
        dy=[]
        #we check our line by iterating the  points
############################### DETECTED LINES FILTERING ###############################
        for s in range(len(x)):
            #skip if point is not on picture size range (0-i for y) (0-j for x but already satisfafied)
            if  (int(y[s])>=i or int(y[s])<0) or (int(x[s])>=j or x[s]<0) :continue
            #Skip if point is not active according to space matrix
            #We check points arount the current not active points (if line not shaped corectly)
            if space[int(y[s]),int(x[s])]==0:
                for u in range(-2,2):
                    for v in range(-2,2):
                        try:t= space[int(y[s])+v,int(x[s])+u]
                        except: t=0
                        if t>0:
                            #increment space index to manifest more trust for the point (since the point is on active pixel and is part of a line)
                            try:space[int(y[s])+v,int(x[s])+u]+=1
                            except:continue
                            # add x y to new lists
                            nx.append(int(x[s])+u)
                            ny.append(int(y[s])+v)
                dx.append(x[s])
                dy.append(y[s])
            #if its active point and in pciture size 
            else:
                #increment space index to manifest more trust for the point (since the point is on active pixel and is part of a line)
                space[int(y[s]),int(x[s])]+=1
                # add x y to new lists
                nx.append(x[s])
                ny.append(y[s])
                dx.append(x[s])
                dy.append(y[s])           
        #list containing our x and y and  the slope our y0 and representation of 4 points of the line
        if len(dx)<2 : continue
        try:
         equa=[nx,ny,det,y0,[[dx[0],dy[0]],[dx[-1],dy[-1]]]]
        # equa=[dx,dy,det,y0,[[dx[0],dy[0]],[dx[-1],dy[-1]]]]
        except:pass
        #if fist iteration add line
        if len(bigLines)==0:
            bigLines.append(equa)
            continue
        #we suppose a new line
        newLine=True
        #we iterate trought saved lines
        #current line
        currentLine=equa[-1]
        l1=(currentLine[1][0]-currentLine[0][0],currentLine[1][1]-currentLine[0][1])
        #compared lines
        for bv,compareLine in enumerate(bigLines):
            #loop line
            compareLine=compareLine[-1]
            l2=(compareLine[1][0]-compareLine[0][0],compareLine[1][1]-compareLine[0][1])
            costeta=(np.dot(l1,l2))/(math.sqrt(l2[0]**2+l2[1]**2)*(math.sqrt(l1[0]**2+l1[1]**2)))
            diffs=[abs(currentLine[i][j]-compareLine[i][j]) for i in range(2) for j in range(2)]
            diffxy=30
            if costeta<0.3:
                stateee='Perpendicular'
            if all(x<diffxy for x in diffs) and (costeta>0.78):
                newLine=False
                break
        if newLine:
            bigLines.append(equa)    
################################################################################################################           
######## Line Splitting ########
    for llf in bigLines:
     filteredLines=[]
     cleanX=list(set(llf[0]))
     def myfunc(x):
            return llf[2] * x + llf[3]
     cleanY=list(map(myfunc, cleanX))
     LinebyPoints=[(cleanX[xx],cleanY[xx]) for xx in range(len(cleanX)) if 0<cleanX[xx] and 0<cleanY[xx]]
     LinebyPoints.sort(key=lambda k: [k[1], k[0]])
     ita=0
     for i in range(len(LinebyPoints)-1):
        VectDist=math.dist(LinebyPoints[i],LinebyPoints[i+1])
        if VectDist>10:
            try:
                filteredLines.append([LinebyPoints[ita],LinebyPoints[i]])
                ita=i+1
            except:pass#a voir
     try:
      filteredLines.append([LinebyPoints[ita],LinebyPoints[i]])
      splittedLines=splittedLines+filteredLines
     except:pass

    return splittedLines

################################

'''
######################################
    

        plt.scatter([x[0] for x in inter],[x[1] for x in inter],color='r')
    for line in LineObject:
        line.printIntersections()
    
#####################################

    
'''





         
     #plt.plot([llf[-1][0][0],llf[-1][1][0]],[llf[-1][0][1],llf[-1][1][1]],color='g')
     #plt.plot([llf[0][0],llf[0][-1]],[llf[1][0],llf[1][-1]],color='g',linewidth=5)
    

    

# Images

# Inference

# Results



#all files to test ( must be pictores of parking space with min number of cars empty if possible for now)
files = ['input.jpg','inputTest.png','input1.webp','int.webp','outin.webp','parking2.jpg',]
#files to test
#random.shuffle(files)
#if you want random demo order
iy=1
for po,file in enumerate(files):
    img = cv2.imread(file)
    img=cv2.resize(img,[640,480])
    li=detectLines(file)
    
 
    parkingPlace=Parking(li)
    ld=sorted(parkingPlace.lines,key=lambda k: k.linelen,reverse=True)
    for lg in ld :
        print(lg)
        
     

    for l in parkingPlace.graphStrcut:
        #if l>3:continue
        plt.plot([parkingPlace.lines[l].coord[0][0],parkingPlace.lines[l].coord[-1][0]],[parkingPlace.lines[l].coord[0][1],parkingPlace.lines[l].coord[-1][1]],color='r',linewidth=3)

        for ll in parkingPlace.graphStrcut[l]:
            
            plt.plot([parkingPlace.lines[ll].coord[0][0],parkingPlace.lines[ll].coord[-1][0]],[parkingPlace.lines[ll].coord[0][1],parkingPlace.lines[ll].coord[-1][1]],color='g')
            if l in parkingPlace.intersection and ll in parkingPlace.intersection[l]:
                coordsInt=parkingPlace.intersection[l][ll]
                plt.scatter(coordsInt[0],coordsInt[1],color='b')
        #plt.pause(2)

    #plt.imshow(edges)

    plt.show()
    #input()
