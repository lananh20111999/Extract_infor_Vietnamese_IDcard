
import torch
import cv2
import numpy as np
from cropper.cropper import cropper
from detector.detector import detectIDcard
from matplotlib import pyplot as plt
model = torch.hub.load('ultralytics/yolov5', 'custom', path='detector/bestFinal.pt',_verbose=False) 
pathIMG = "imgTest/Hoang Van Tuan 1.jpg"
IDCard = cv2.imread(pathIMG)
listCorners = model(pathIMG).pandas().xyxy[0]
listCorners = listCorners.values.tolist()
listCorners, type = detectIDcard(listCorners)
IDcardType = {"newFront", "newBack", "oldFront", "oldBack"}
if type in IDcardType:
    crop_IdCard = cropper(IDCard, listCorners)
# # %%
# cv2.imshow("idcard", crop_IdCard)
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 


