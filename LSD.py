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
import math
import keyboard
import ast
import random

#func to get intersection btwn to lines line1((x0 y0) x1 y1) line2((x0 y0) (x1 y1))
def line_intersection(line1, line2):
    #acces to space matrix (which is the matrix that tells us if a pixel in the picture is worth to compute or no )
    global space

    #intersection detect : source  (https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    #avoid infinite division
    if div == 0:
       return False

    d = (det(*line1), det(*line2))
    # end intersection detect
    #intersect coords:
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    #check if intersection point is in our picture size range
    if 0<y<len(space) and 0<x<len(space[0]):
     #prints the intersection point and its state in the space matrix
     #print(f'({int(x),int(y)}) : {space[int(y),int(x)]}')
     #if our matrix space at a postion x,y is 0 (space(x,y)=0 : this point is not  woth to compute because  its not an active point)
     if space[int(y),int(x)]!=0:
        #if active point return intersection point
        return x, y
        #if active point return intersection point
     else:return False
     
#IF POINT not in pciture => False
    else : return False

######################################### MAIN #############################################


def detectLines(file):
    #store the y=mx+b lines

    bigLines=[]

    #we read the frame/picture
    img = cv2.imread(file)
    #filter
    gr=cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    #filter to lower details
    blur_gray = cv2.GaussianBlur(src=gr, ksize=(5, 5), sigmaX=0)
    #threshold used for canny edges detection
    thresholds=[30, 200,]
    #read picture/frame
    img = cv2.imread(file)   
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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 50)  # !  Lines= Detected lines
    #get the size of the image
    i,j= img.shape[:2]
    #Canny filter gives us a picture of active(255)/not active(0) pixels in the picture
    #so we can iterate in the picture to spot the active pixels 
    lescord=[(x,y) for x in range(i) for y in range(j) if edges[x,y]==255]
    # we create a matrix with the size of the picture (i x j) init to 0
    space=np.full((i,j),0)
    #we mark our active pixels as 1 in the space matrix
    for x,y in lescord:
        space[x,y]=1
    #we put our picture in matplotlib board
    plt.imshow(edges)
    #list where we save the line equations 
    bigLines=[]
    #(new idea) list of intersection points
    comp=[]
    #we iterate detected lines
    for ff,line1 in enumerate(lines):
        #line1 => [[line]] so we convert to line1 => [[x0,y0],[x1,y1]]
        line1=[line1[0][:2],line1[0][2:]]
        #calculate the slope of the line:    det = (y1-y0)/(x1-x0)
        #   if same x absis slope is vertical :)  det=0 
        if line1[1][0]!=line1[0][0]:
            det = (line1[1][1]-line1[0][1])/(line1[1][0]-line1[0][0])
        else:det=0
        #update title with plotting line number
        #plt.title(f'[{po+1}/{len(files)}]{state} Line {ff+1}/{len(lines)}')
        plt.title(f'Low Detail image')
        #Plot detected line Instruction :
        #plt.plot([line1[0][0],line1[1][0]],[line1[0][1],line1[1][1]],color='r',linewidth=1)
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
        for s in range(len(x)):
            #skip if point is not on picture size range (0-i for y) (0-j for x but already satisfafied)
            if  (int(y[s])>=i or int(y[s])<0) or (int(x[s])>=j or x[s]<0) :continue
            #Skip if point is not active according to space matrix
            if space[int(y[s]),int(x[s])]==0:
                

                for u in range(5):
                    for v in range(5):
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
        equa=[nx,ny,det,y0,[[dx[0],dy[0]],[dx[-1],dy[-1]]]]
        #if fist iteration add line
        if len(bigLines)==0:
            bigLines.append(equa)
            continue
        #we suppose a new line
        newLine=True
        #we iterate trought saved lines
        #A
        ll2=equa[-1]
        l1=(ll2[1][0]-ll2[0][0],ll2[1][1]-ll2[0][1])
        for bv,ll in enumerate(bigLines):
            #loop line
            ll=ll[-1]
            l2=(ll[1][0]-ll[0][0],ll[1][1]-ll[0][1])
            costeta=(np.dot(l1,l2))/(math.sqrt(l2[0]**2+l2[1]**2)*(math.sqrt(l1[0]**2+l1[1]**2)))
            c=np.cross(l1,l2)
            stateee=''
            if np.arccos(costeta)*180/math.pi<15:
                stateee='Parallal'
            if 70<np.arccos(costeta)*180/math.pi<97:
                stateee='Perpendicular'

            diffs=[abs(ll2[i][j]-ll[i][j]) for i in range(2) for j in range(2)]
        
            print(f'>>>>>Line {ff}--{bv}    \n      COS={costeta:.2}\n      angle: {(np.arccos(costeta)*180/math.pi)}\n     crossP: {c:.4} ')
            diffxy=30
            print(f'        points  <10   :{all(x<diffxy for x in diffs)}')
            print(f'*******End Line {ff}--{bv}   ******** ')
            if all(x<diffxy for x in diffs) and (costeta>0.78):
                newLine=False
                print(f'>Same line : Line {ff}--{bv}')

                break
        if newLine:
            bigLines.append(equa)    
           
    print(f'>  Result :  {len(bigLines)} Lines')
    for kj,las in enumerate(bigLines):
       
        print(f'> Line {kj}/{len(bigLines)-1}  : {len(las[0])}')
        plt.scatter(las[0],las[1],linewidths=0.5)
        plt.pause(0.1)
    plt.title('space')
    plt.show()


#all files to test ( must be pictores of parking space with min number of cars empty if possible for now)
files = ['parking.jpg','parking2.jpg','input.jpg','outin.webp','int.webp','inputTest.png','input1.webp']
#files to test
random.shuffle(files)
#if you want random demo order

for po,file in enumerate(files):
    detectLines(file)