from concurrent.futures import ThreadPoolExecutor
import random
import time
from scipy import stats
import torch
import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math
from sklearn.linear_model import LinearRegression


class vect:
    def __init__(self,coord) -> None:
       self.coord=coord
       self.count=1
       self.state='MOVE'#STATIC
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
fff=open('coco.txt')
aclass=fff.readlines()
aclass=[str(x.replace('\n','')) for x in aclass]


     
    


def run(inpt):
    # used to record the time when we processed last frame
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame
    new_frame_time = 0
    img_array = []

    cap = cv2.VideoCapture(inpt)
    ff=0
    while True:
        vectLine=[]

        ff+=1
        try:
            imgSource = cap.read()[1]
            img = cv2.resize(imgSource, (640,480))
            img_array.append((img,ff))
        except Exception as e:
            print(e)

        if ff%11==1:
            results = model(img)
            res=list(results.xyxy[0])
            #x1y1x2y2ac
            resi=[]
            if len(res)==0:continue
            if ff==1:
             savedTensors=[[t,0,i] for i,t in enumerate(res)]
            if ff!=1:
                if len(res)!=len(oldres):
                    print(f' Detection DIFF {len(res)-len(oldres)}')
                    a=[(i,j,torch.allclose(re,x[0],atol=50)) for i,x in enumerate(savedTensors) for j,re in enumerate(res)]
                    PreviousCarTrack=[i for i in a if True== i[-1]]
                    PreviousCarTracki=[i[1] for i in PreviousCarTrack]
                    NewCarTrack=[i for i in range(len(res)) if i not in PreviousCarTracki]

                    
                    for ii,jj,rr in PreviousCarTrack :
                        savedTensors[ii][1]+=1
                        savedTensors[ii][0]=res[jj]
                  


                    print(f'>Previous Cars ==== {len(PreviousCarTracki)}')
                    for car in PreviousCarTrack:
                        print(f'>>CAR   {car} :')
                        #input()
                    
                    print(f'>New Cars ==== {len(NewCarTrack)} ')
                    for car in NewCarTrack:
                        print(f'>>NEW CAR    : Seen : {car}')

                    
                



                        
            resi=[x[0] for x in savedTensors]
            #resi=res
            lowline=[(float(x[0]),float(x[1])) for x in resi ]
            midLine=[(float(x[0]),float((x[3]+x[1])/2)) for x in resi ]
            upline=[(float(x[0]),float(x[3])) for x in resi ]
            upline.sort(key=lambda k : k[0])
            res.sort(key=lambda k : float(k[0]))
            midLine.sort(key=lambda k : k[0])
            lowline.sort(key=lambda k : k[0])
            for x in resi:img =cv2.arrowedLine(img, (int(x[0]),int(x[3])), (int(x[2]),int(x[3])), (0,255,0), 3) 
            for l in range(0, len(midLine)-1):
                index=l+1
                pt1=midLine[l]
                pt2=midLine[index]
                vect1=(pt2[0]-pt1[0],pt2[1]-pt1[1])
                if abs(vect1[1])>10:
                    continue
                else: vectLine.append([int(x) for x in pt2])
                cv2.arrowedLine(img, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), (255,0,0), 8) 
            plots=[vectLine,upline ,lowline]
        if ff%11==1:
            ux=[x[0] for x in vectLine]
            uy=[x[1] for x in vectLine]
            slope, intercept, r, p, std_err = stats.linregress(ux, uy)
            def myfunc(x):
                return slope * x + intercept
            mymodel = list(map(myfunc, ux))
            x1=0
            x2=int(img.shape[1])-1
         
            y1=int(myfunc(x1))
            y2=int(myfunc(x2))
        for t in savedTensors:
                        print(t)
                        t=t[0]
                        #input()
                        x1=int(t[0])
                        x2=int(t[1])
                        x3=int(t[2])
                        x4=int(t[3])
                        
                        cv2.rectangle(img,(x1,x2),(x3,x4),(255,0,0),2)
        cv2.line(img, (0,0),(x2,y2),  (0,255,2), 4) 
        cv2.line(img, (x1,y1), (0,0), (0,255,2), 4) 
        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 4) 
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
    
        # converting the fps into integer
        fps = int(fps)
    
        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)
    
        # putting the FPS count on the frame
        cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
      

        oldres=res
    
                
        cv2.imshow("VIDEO",img)
        #cv2.imshow("VIDEO",np.squeeze (results.render()))
        k = cv2.waitKey(10)
        if k == ord('q'):
            ff=-1
            return

         
        

inputs=['in4.mp4',]
#inputs=['input.mp4','input2.mp4','inputWrong.mp4','inputSmallCam.mp4',]
img_array = []

for inpt in inputs:
    run(inpt)
print(len(img_array))

