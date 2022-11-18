import torch
import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math



class vect:
    def __init__(self,coord) -> None:
       self.coord=coord
       self.count=1
       self.state='MOVE'#STATIC
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
fff=open('coco.txt')
aclass=fff.readlines()
aclass=[str(x.replace('\n','')) for x in aclass]


def procImg():
    #global imgSource
    global res
    global results

    global ff
    ff1=ff
    plt.figure(figsize=(20, 10))
    i=0
    while True:
     i+=1
     if ff==-1 :break
     if ff!=ff1:
      results = model(imgSource)
      res=list(results.xyxy[0])
      strinfo=f'{ff1} - {ff}'
      #print(f'ok  {strinfo}')
      #print(len(res))
      plt.clf()
      try:
        plt.imshow(imgSource)
      except:break
      #im=Image.fromarray(imgSource)
      if len(res)==0:continue
      xmin,ymin,xmax,ymax,conf,classid =[float(ii) for ii in list(res[0])]
      lowline=[(float(x[0]),float(x[1])) for x in res ]
      midLine=[(float(x[0]),float((x[3]+x[1])/2)) for x in res ]
      upline=[(float(x[0]),float(x[3])) for x in res ]
      upline.sort(key=lambda k : k[0])
      res.sort(key=lambda k : float(k[0]))
      midLine.sort(key=lambda k : k[0])
      lowline.sort(key=lambda k : k[0])
      [plt.arrow(int(x[0]),int(x[3]),int(x[2]-x[0]),int(x[1]-x[1]),color='tab:olive',head_width = 12,head_length=12) for x in res]
      vects=[vect((int(x[0]),int(x[3]),int(x[2]-x[0]),0),) for x in res]
      print('===========')
      #input()
      for l in range(0, len(midLine)-1):
        index=l+1
        pt1=midLine[l]
        pt2=midLine[index]
        vect1=(pt2[0]-pt1[0],pt2[1]-pt1[1])
        if abs(vect1[1])>10:
            continue
        plt.arrow(pt1[0],pt1[1],vect1[0],vect1[1],color='r',head_width = 12,head_length=12)
        plots=[upline ,lowline]
        for tab in plots:
          ux=[x[0] for x in tab]
          uy=[x[1] for x in tab]
          plt.plot(ux,uy,color='g',linewidth=2)
     
      
      

      plt.savefig(f'i.jpg')
      
    

      #plt.show()
      #plt.savefig(f'detectionFrameResults/take{strinfo}.jpg',bbox_inches='tight')

      #input()
      #im.save(f'detectionFrameResults/take{strinfo}.jpg')
      #im.show()



      ff1=ff
    
    return



def run(inpt):
    global img,img_array
    img_array = []
    global imgSource
    global res
    global results
    cap = cv2.VideoCapture(inpt)
    started=False
    global ff
    ff=0
    while True:
        try:
         imgSource = cap.read()[1]
         img = cv2.resize(imgSource, (640,480))
         img_array.append(img)
         
        except:
            ff=-1
            h.join()
            return
            break
        ff+=1
        
        if not  started:
         results = model(img)
        if img is None:
            ff=-1
            break
        if not  started:
            if __name__ == '__main__':
             h= threading.Thread(target=procImg)
             h.start()
            started=True
        

        
        cv2.imshow("VIDEO",img)
        #cv2.imshow("VIDEO",np.squeeze (results.render()))
        k = cv2.waitKey(10)
        if k == ord('q'):
            h.join()
            ff=-1
            return
        continue

inputs=['in4.mp4',]
#inputs=['input.mp4','input2.mp4','inputWrong.mp4','inputSmallCam.mp4',]
img_array = []

for inpt in inputs:
    run(inpt)
print(len(img_array))
procImg()