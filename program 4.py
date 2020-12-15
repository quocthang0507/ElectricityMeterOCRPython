import cv2
import easyocr
import os
from os import listdir
import numpy as np
import time

# For OCR
BORDER_SIZE = 10
CONFIDENCE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.5
LANGS = ["en", "ar"]

reader = easyocr.Reader(LANGS, False, 'Model/', 'Model/',
                        download_enabled=False)  # Without gpu
net = cv2.dnn.readNet("Model\yolov4-custom_last.weights",
                      "Model\yolov4-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255)

# For putText
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2


def CleanupText(result):
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


def ProcessOCR(image):
    try:
        _, _, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        if len(boxes) != 1:
            return "E98"
        else:
            for box in boxes:
                image = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

                image = cv2.copyMakeBorder(
                    image,
                    top=BORDER_SIZE,
                    bottom=BORDER_SIZE,
                    left=BORDER_SIZE,
                    right=BORDER_SIZE,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

                result = reader.readtext(
                    image, allowlist='0123456789', text_threshold=0.3)
                result = CleanupText(result)
                return result

    except:
        return "Error"

    return ""


def OCR(imagePath, outputPath):
    outputFileName = os.path.basename(imagePath)
    if outputPath != "":
        outputFilePath = os.path.dirname(imagePath)
        outputFileName = os.path.join(outputFilePath, '_' + outputFileName)
    else:
        outputFileName = '_' + outputFileName
    image = cv2.imread(imagePath)

    start = time.time()
    result = ProcessOCR(image)
    image = cv2.putText(image, result, org, font, fontScale, color, thickness)
    cv2.imwrite(outputFileName, image)
    end = time.time()

    print(str.format('Processed image in {:.2f} ms', (end - start) * 1000))


def OCR_WithOutput(imageFolderPath, outputFolderPath):
    images = LoadImages(imageFolderPath)
    index = 0
    for image in images:
        index += 1
        print(str.format('Processing no. {} image', index))

        start = time.time()
        result = ProcessOCR(image)
        image = cv2.putText(image, result, org, font,
                            fontScale, color, thickness)
        outputFileName = os.path.join(outputFolderPath, str(index) + ".jpg")
        cv2.imwrite(outputFileName, image)
        end = time.time()

        print(str.format('Processed no. {} image in {:.2f} ms',
                         index, (end - start) * 1000))


if __name__ == "__main__":
    OCR_WithOutput('D:\Github\OCR\Images',
                   outputFolderPath='D:\Github\OCR\Output')
