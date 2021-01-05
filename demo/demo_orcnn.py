import cv2 as cv
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

AGE_MODEL_PATH = "../model/age/test12072/best_model.pt"
GENDER_MODEL_PATH = "../model/gender/test1209/model_gen.pt"
pathAdience = "../model/Adience/"
pathAFAD = "../model/AFAD/"
frame_color = (10,10,205)
########################
### Face
########################
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    #print(detections.shape[2])
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), frame_color, int(round(frameHeight/150)), 1)
    return frameOpencvDnn, bboxes


faceProto = pathAdience+"opencv_face_detector.pbtxt"
faceModel = pathAdience+"opencv_face_detector_uint8.pb"
faceNet = cv.dnn.readNet(faceModel, faceProto)

######################
### Gender
######################
from GenCNN import CNN_Model_
genderList = ['Male', 'Female']

def GenderPredict(image_np):
    image_PIL = Image.fromarray(image_np)
    custom_transform = transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.CenterCrop((227,227)),
                                     transforms.ToTensor()
                                     ])
    image = custom_transform(image_PIL)
    DEVICE = torch.device('cpu')
    image = image.to(DEVICE)
    
    model = CNN_Model_()
    model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=DEVICE))
    model.eval()
    image = image.unsqueeze(0)

    with torch.set_grad_enabled(False):
        output = model(image)
        probas = output.data.max(dim = 1, keepdim = True)[1]
        #predict_levels = probas > 0.5
        #predicted_label = torch.sum(predict_levels, dim=1)
        #predicted_label = torch.sum(probas, dim=1)
        #print('Predicted gender:', probas)#predicted_label.item())
        #genderPredict = int(predicted_label.item())
        genderLabel = genderList[probas]
        #print(genderLabel)
        return genderLabel

##############################
### Age
##############################
from ResNet34 import resnet34_

torch.backends.cudnn.deterministic = True #
GRAYSCALE = False #
#dataset=afad,start=15,finish=41
ADD_CLASS = 15
AGE_NUM = 58


#input<numpy.ndarray>/output age number
def AgePredict(image_np):
    image_PIL = Image.fromarray(image_np)
    custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.CenterCrop((120, 120)),
                                       transforms.ToTensor()])
    image = custom_transform(image_PIL)
    DEVICE = torch.device('cpu')
    image = image.to(DEVICE)
    
    model = resnet34_(AGE_NUM , GRAYSCALE)
    model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    image = image.unsqueeze(0)

    with torch.set_grad_enabled(False):
        logits, probas = model(image)
        predict_levels = probas > 0.5
        predicted_label = torch.sum(predict_levels, dim=1)
        #print('Predicted age:', predicted_label.item() + ADD_CLASS)
        agePredict = predicted_label.item() + ADD_CLASS
        return agePredict


###############################
### webcam
###############################
'''import numpy as np
photo = 'D:/DeepLearning/GAR/1202/AFAD-Full/19/111/29305-0.jpg'
img = Image.open(photo)
img = np.array(img)
gender = GenderPredict(img)
age= AgePredict(img)
print('a/g:',age,gender)'''


cap = cv.VideoCapture('D:\DeepLearning\GAR\1223\video\台灣捷運站_3F.mp4')
fourcc = cv.VideoWriter_fourcc(*'XVID')#fourcc為視訊編解碼器
padding = 20
start_time = time.time()
while cv.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        ##G
        gender = GenderPredict(face)

        ##A
        age=AgePredict(face)

        label = "{},{}".format(gender, age)
        cv.rectangle(frameFace, (bbox[0]-2, bbox[3]), (bbox[2]+2, bbox[3]+10), frame_color, -1)
        cv.putText(frameFace, label, (bbox[0], bbox[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv.LINE_AA)
        cv.imshow("Age Gender Demo", frameFace)

    print("time : {:.3f}".format(time.time() - t))
    if cv.waitKey(20) & 0xFF == 27:
        break 
    
cap.release() 
cv.destroyAllWindows()
print('RECORD(start)',start_time,'~(finish)',time.time())