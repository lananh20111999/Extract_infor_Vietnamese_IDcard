# %%
import torch
import cv2
import numpy as np
from cropper.cropper import cropper
from detector.detector import detectIDcard
from textDetection.textDetection import text_detection
from textDetection.imgproc  import drawBox
model = torch.hub.load('ultralytics/yolov5', 'custom', path='detector/bestFinal.pt',_verbose=False) 
pathIMG = "imgTest/Hoang Van Tuan 1.jpg"
IDCard = cv2.imread(pathIMG)
listCorners = model(pathIMG).pandas().xyxy[0]
listCorners = listCorners.values.tolist()
listCorners, type = detectIDcard(listCorners)
IDcardType = {"newFront", "newBack", "oldFront", "oldBack"}
if type in IDcardType:
    crop_IdCard = cropper(IDCard, listCorners)

# %%
crop_IdCard2 = crop_IdCard[110:300, 135:500]
bboxes, polys, score_text = text_detection(crop_IdCard2)
polys = np.array(bboxes).astype(np.int32)
# print(polys)
# %%
crop_IdCard2 = drawBox(crop_IdCard2, polys)
cv2.imshow("idcard", crop_IdCard2)
cv2.waitKey(0) 
cv2.destroyAllWindows()

# %%
