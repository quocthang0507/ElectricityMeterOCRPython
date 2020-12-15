import cv2
import easyocr
import os
from os import listdir
import numpy as np
import time

bordersize = 10
CONFIDENCE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.5

reader = easyocr.Reader(['en'], gpu=False)
net = cv2.dnn.readNet("Model\yolov4-custom_last.weights",
                      "Model\yolov4-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255)


def cleanup_text(result):
    ret_text = ""
    for r in result:
        ret_text += r[1]
    return ret_text


def LoadImages(folderPath):
    files = listdir(folderPath)
    images = []
    for file in files:
        img = cv2.imread(os.path.join(folderPath, file))
        if img is not None:
            images.append(img)
    return images


def OCR(image):
    try:
        _, _, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        if len(boxes) != 1:
            return "E98"
        else:
            for box in boxes:
                image = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

                image = cv2.copyMakeBorder(
                    image,
                    top=bordersize,
                    bottom=bordersize,
                    left=bordersize,
                    right=bordersize,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

                result = reader.readtext(
                    image, allowlist='0123456789', text_threshold=0.3)
                result = cleanup_text(result)
                return result

    except:
        return "E99"

    return ""


def OCR(imagePath, outputPath):
    start = time.time()
    outputFileName = os.path.basename(imagePath)
    if outputPath != "":
        outputFilePath = os.path.dirname(imagePath)
        outputFileName = os.path.join(outputFilePath, '_' + outputFileName)
    else:
        outputFileName = '_' + outputFileName
    image = cv2.imread(imagePath)

    result = OCR(image)

    end = time.time()
    print(str.format('Processed image in {:.2f} ms', (end - start) * 1000))
