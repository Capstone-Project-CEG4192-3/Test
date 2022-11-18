import cv2
import numpy as np



net = cv2.dnn.readNetFromONNX(r"C:\Users\Lenovo\Desktop\capstone\yolov5n.onnx")
file = open(r"C:\Users\Lenovo\Desktop\capstone\coco.txt","r")
classes = file.read().split('\n')

def run(imgname):
    img = cv2.imread(r'C:\Users\Lenovo\Desktop\capstone\outin.webp')
    img = cv2.imread(imgname)

    img = cv2.resize(img, (1000,600))
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

    for i in range(rows):
        
        row = detections[i]
        confidence = row[4]
     
        if confidence > 0.5:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.5:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx- w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)

    for i in indices:
        x1,y1,w,h = boxes[i]
        f=open('ip5coords.txt','a')
        f.write(f'{x1,y1,w,h}\n')
        
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),2)
        cv2.putText(img, text, (x1,y1+h-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,255,255),2)
    cv2.imshow('Dectecting objects (Cars)',img)
    cv2.waitKey(0)
    
    # cv2.destroyAllWindows() simply destroys all the windows we created.
    cv2.destroyAllWindows()

files = ['ip5.jpg','parking2.jpg','outin.webp','ip3.jpg',]
#files to test
#random.shuffle(files)
#if you want random demo order

for po,file in enumerate(files):
    run(file)