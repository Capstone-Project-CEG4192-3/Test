import cv2
import numpy as np

import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = r'C:\Users\Lenovo\Desktop\capstone\parking2.jpg'  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)
print(results.pandas().xyxy[0])
import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("yolov5n.onnx")
file = open("coco.txt","r")
classes = file.read().split('\n')
#print(classes)

def run(inpt):

        img = cv2.imread(imgs,)
        
        blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
        net.setInput(blob)
        detections = net.forward()[0]
    

        # cx,cy , w,h, confidence, 80 class_scores
        # class_ids, confidences, boxes

        classes_ids = []
        confidences = []
        boxes = []
        rows = detections.shape[0]

        img_width, img_height = img.shape[1], img.shape[0]
        x_scale = img_width/640
        y_scale = img_height/640

        for i in range(len(results.xyxy[0])):
            row = results.xyxy[0][i]
            confidence = row[4]
            
            if confidence > 0.5:
                  
                classes_score = row[5:]
                
                ind = np.argmax(classes_score)
               # input()

                if classes_score[ind] > 0.5:
                    classes_ids.append(ind)
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx- w/2)*x_scale)
                    y1 = int((cy-h/2)*y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1,y1,width,height])
                    print(box)
                    boxes.append(box)

        #indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)

        for i,jj in enumerate(boxes):
            x1,y1,w,h = boxes[i]
            label = classes[classes_ids[i]]
            conf = confidences[i]
            text = label + "{:.2f}".format(conf)
            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),2)
            cv2.putText(img, text, (x1+w,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),1)
        cv2.imshow("VIDEO",img)
        k = cv2.waitKey(10)
        if k == ord('q'):
            return
inputs=['input.mp4','input2.mp4']
for inpt in inputs:
    run(inpt)

