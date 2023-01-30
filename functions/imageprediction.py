# from fcntl import DN_ACCESS
# from locale import dcgettext
# from os import sched_getscheduler
# from telnetlib import XASCII
import cv2 as cv 
import numpy as np
import time
# import YOLOv4
# from yolov4.tf import YOLOv4

class Yolov4():
    def __call__(self,img):
        CONFIDENCE_THRESHOLD = 0.9
        NMS_THRESHOLD = 0.4
        COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        class_names = []
        with open("yolov3.txt", "r") as f:
            class_names = [cname.strip() for cname in f.readlines()]
        net = cv.dnn.readNet("yolov4-obj_final.weights", "yolov4.cfg")
        # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
        model = cv.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
        start = time.time()
        classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()
        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[classid], score)
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            # cv.rectangle(img, (round(x),round(y)), (round(x+w),round(y+h)), color, 2)
            # cv.putText(img, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)    
        # cv.putText(img, 2 , (0, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return img

            
class Yolo():

    def get_output_layers(self,net):   
        layer_names = net.getLayerNames()   
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    # def draw_prediction(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):
    #     label = str(classes[class_id])
    #     color = COLORS[class_id]
    #     cv.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    #     cv.putText(img, label, (x-10,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def draw_prediction(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv.putText(img, label, (x-10,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def __call__(self,img):
        Width = img.shape[1]
        Height = img.shape[0]
        scale = 0.00392
        classes = None
        with open('yolov3.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        net = cv.dnn.readNetFromDarknet('yolov4-obj.cfg', 'yolov4-obj_final.weights')
        # net = cv.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4-obj_final.weights')
        blob = cv.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(self.get_output_layers(net))
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        abox = []
        for i in indices:
            box = boxes[i]
            print(box)
            abox.append(box)
            print(abox)
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_prediction(img, class_ids[i], confidences[i],
                                     round(x), round(y), round(x+w), round(y+h),classes,COLORS)
        return img,abox

class TrackImage():
    def __init__(self):
        self.objects = []
        self.object_pos = []
        self.object_vel =[]
        self.object_acc = []

    def center(self,position):
        pos = []
        pos.append(position[0]+position[2]/2)
        pos.append(position[1]+position[3]/2)
        return pos
    
    def images(self,boxes):
        centroids = []
        for i in range(len(boxes)):
            centroids.append(self.center(boxes[i]))
        return centroids
    
    def register_acc(self):
        for i in range(len(self.object_vel)):
            if len(self.object_vel[i]) > 1:
                self.object_acc[i] = np.concatenate((self.object_acc[i],[(self.object_vel[i][-1]-self.object_vel[i][-2])]),axis = 0)
    
    def register_velocity(self):
        for i in range(len(self.objects)):
            if len(self.objects[i]) > 1:
                self.object_vel[i] = np.concatenate((self.object_vel[i],[(self.objects[i][-1]-self.objects[i][-2])]),axis = 0)
        return self.object_vel
    
    def register_object(self,imgs,thres):
        centers = self.images(imgs)
        flag = False
        done = []
        for i in range(len(centers)):
            flag = False
            for j in range(len(self.objects)):
                if max(np.absolute(np.array(centers[i]) - np.array(self.objects[j][-1]))) < thres :
                    self.objects[j] = np.concatenate((self.objects[j],[centers[i]]),axis = 0)
                    flag = True
            if flag==False:
                self.objects.append([centers[i]])
                self.object_vel.append([[0,0]])
                self.object_acc.append([[0,0]])
            self.register_velocity()
            self.register_acc()
        return self.objects

    
class imagepred():
    
    def __init__(self):
        self.yolo = Yolo()
        self.yolov4 = Yolov4()
        self.trackimage = TrackImage()
        
    def show(self,img):
        cv.imshow("object detection", img)
        cv.waitKey()  
        cv.imwrite("object-detection.jpg", img)
        cv.destroyAllWindows()
    
    def single(self,image):
        img = self.yolo(image)
        self.show(img)

    def __call__(self,img,str,end,time):
        boxes = []
        for i in range(str,end):
            pos = []
            if i % time == 0:
                img1,box,classes = self.yolo(img[i])
                boxes.append(box)
                print(box)
                # print(img1)
                self.show(img1)
                self.trackimage.register_object(pos,0.8)
        return boxes
            
        

