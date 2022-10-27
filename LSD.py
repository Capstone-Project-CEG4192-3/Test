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
#store the y=mx+b lines
bigLines=[]

#func to get intersection btwn to lines line1((x0 y0) x1 y1) line2((x0 y0) (x1 y1))
def line_intersection(line1, line2):
    #acces to space matrix (which is the matrix that tells us if a pixel in the picture is worth to compute or no )
    global space

    #intersection detect (https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
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

#all files
files = ['parking.jpg','parking2.jpg','p3.jpg','p4.jpg','input.jpg','outin.webp','int.webp','inputTest.png','input1.webp']
#files to test
demo=['parking2.jpg','inputTest.png','input.jpg','input1.webp','outin.webp',]
random.shuffle(files)
#if you want random demo order
random.shuffle(demo)




for po,file in enumerate(demo):
    #we read the frame/picture
    file='outin.webp'
    img = cv2.imread(file)
    #filter
    gr=cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    #filter to lower details
    blur_gray = cv2.GaussianBlur(src=gr, ksize=(5, 5), sigmaX=0)
    #threshold used for canny edges detection
    thresholds=[40, 180,]
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
        else:det=9
        #update title with plotting line number
        plt.title(f'[{po+1}/{len(files)}]{state} Line {ff+1}/{len(lines)}')
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
        ny=[]
        #we check our line by iterating the  points
        for s in range(len(x)):
            #skip if point is not on picture size range (0-i for y) (0-j for x but already satisfafied)
            if  (int(y[s])>=i or y[s]<0) or (int(x[s])>=j or x[s]<0) :continue
            #Skip if point is not active according to space matrix
            if space[int(y[s]),int(x[s])]==0:continue
            #if its active point and in pciture size 
            else:
                #increment space index to manifest more trust for the point (since the point is on active pixel and is part of a line)
                space[int(y[s]),int(x[s])]+=1
                # add x y to new lists
                nx.append(x[s])
                ny.append(y[s])
                #list containing our x and y and  the slope our y0 and representation of 4 points of the line
        equa=[x,y,det,y0,[[nx[0],ny[0]],[nx[-1],ny[-1]]]]
        #add line to our lines list
        bigLines.append(equa)
        #check intersection between saved lines (or we can use a function to detect if two lines are similar and save just one of the two)
        comp=comp+[line_intersection(lst1[-1],lst2[-1]) for lst1 in bigLines for lst2 in bigLines if line_intersection(lst1[-1],lst2[-1])!=False]
        #draw img
        img2 = cv2.imread(file)   
        #draw filtred line in opencv
        cv2.line(img2, [int(nx[0]),int(ny[0])], [int(nx[-1]),int(ny[-1])], (0, 0, 255), 2)
        plt.pause(0.8)
        cv2.imshow("LSD",img2 )
          
        print(f'> Line  {ff+1}/{len(lines)} : y0: {equa[-2]} slope: {det}  equa : y={det:.2}x+{det+y0:.2}')
        print(f'>Intersection points before set  : ',(len(comp)))
        if len(comp)>=1:
            comp=[ (int(xx),int(y)) for xx,y in comp]
            comp=set(comp)
            comp=list(comp)
            print(f'>Intersection points after set  : ',(len(comp)))
        #sort list of intersection according to : 
        #distance from origin
        comp.sort(key=lambda x:math.sqrt(x[0]**2+x[1]**2))
        #ratio distance/angle
        #comp.sort(key=lambda x:math.sqrt(x[0]**2+x[1]**2)/(math.atan(x[1]/x[0])))
        #actual point angle with origin
        ang1=0
        #previous point angle with origin
        ang2=0
        #actual distance btwn point and origin 
        ccs1=0
        #previous distance btwn point and origin 
        ccs2=0
        #previous point
        prevP=[0,0]
        #savedpoints
        accumPointList=[]
        #if there is intersection points
        if len(comp)>=1:
            #vvv is index and ccs is actual point in the list of intersection points
            for vvv ,ccs in enumerate(comp):
                #we get distance and angle
                ccs1=math.sqrt(ccs[0]**2+ccs[1]**2)
                ang1=math.atan(ccs[1]/ccs[0])*180/math.pi
                #print actual point infos
                print(f'> Intersect Point  {vvv+1}/{len(comp)} :  ',ccs,f'   -- distance: {ccs1}  -- angle : {ang1}')
                #IF Statement to tell if actual point is the same as previois point as the list of points is sorted (testing it to see if it works)
                #we check distance btwn points + angle difference and x and y differences 
                #we can test cross product if 2 points are on same postion matbe cross product close to 0
                print(f'>cross : {np.cross(ccs,prevP)}')
                if abs(ccs1-ccs2)<20 and  abs(ang1-ang2)<10 and abs(ccs[0]-prevP[0])<10 and abs(ccs[1]-prevP[1])<10 :
                    #if two close points add then to the list accumulator 
                    accumPointList.append(ccs)
                    #if the actual point and previous point are not close we draw the actual saved points as one point since they are all similar 
                else:
                    #check if we have a lot of points that spot the same point position (generaly >2 or >5)
                    if len(accumPointList)>2:
                        #SOME Np results to compare
                        average=np.average(accumPointList,axis=0)
                        mediane=np.median(accumPointList,axis=0)
                        mean=np.mean(accumPointList,axis=0)
                        #using median for now
                        graal=mediane
                        print('>$$$$$$$ Average   :', average)
                        print('>$$$$$$$ Mediane   :', mediane)
                        print('>$$$$$$$ Mean   :', mean)
                        print('>$$$$$$$ Used   :', graal)
                        print('>$$$$$$$ SPACEMatrix at this position :' ,{space[int(graal[1]),int(graal[0])]})
                        #maybe we can use this alsp : 
                        print(f'> %%%%   %%%% Result  {len(accumPointList)}/{len(comp)} Points pointing to => {graal}')
                        #we draw the point in matplotlib
                        plt.scatter(int(graal[0]),int(graal[1]))
                        #reset acumulator list for a new point
                        accumPointList=[]
                        #split line for next Point to draw
                        print('-----------------------------------')
                        #plt.pause(0.5)
                #we save actual point info
                prevP=ccs
                ccs2=math.sqrt(ccs[0]**2+ccs[1]**2)
                ang2=math.atan(ccs[1]/ccs[0])*180/math.pi

        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
       