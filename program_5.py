from easyocr import Reader
import cv2
import os
from os import listdir
import time


def CleanupText(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


def LoadImages(folderPath):
    files = listdir(folderPath)
    images = []
    for file in files:
        img = cv2.imread(os.path.join(folderPath, file))
        if img is not None:
            images.append(img)
    return images


def OCR(imagePath, langs=["en", "ar"], outputPath=""):
    outputFileName = os.path.basename(imagePath)
    if outputPath != "":
        outputFilePath = os.path.dirname(imagePath)
        outputFileName = os.path.join(outputFilePath, '_' + outputFileName)
    else:
        outputFileName = '_' + outputFileName

    start = time.time()
    image = cv2.imread(imagePath)

    scale_percent = 70
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    reader = Reader(langs, False, 'Model/', 'Model/',
                    download_enabled=False)  # Without gpu

    results = reader.readtext(image)

    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        text = CleanupText(text)
        if len(text) in range(4, 20):
            if ((tl[0] > -1) and (tr[0] > -1)):
                if (text[0] == '0'):
                    cv2.rectangle(image, tl, br, (0, 255, 0), 2)
                    cv2.putText(image, text[0:9], (tl[0], tl[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(outputFileName, image)
    end = time.time()
    print(str.format('Processed image in {:.2f} ms', (end - start) * 1000))


def OCR_WithOutput(imageFolderPath, langs=["en", "ar"], outputFolderPath=""):
    images = LoadImages(imageFolderPath)
    reader = Reader(langs, False, 'Model/', 'Model/',
                    download_enabled=False)  # Without gpu

    index = 0
    scale_percent = 70
    for image in images:
        index += 1
        print(str.format('Processing no. {} image', index))

        start = time.time()
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        results = reader.readtext(image)

        for (bbox, text, prob) in results:
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))

            text = CleanupText(text)
            if len(text) in range(4, 20):
                if ((tl[0] > -1) and (tr[0] > -1)):
                    if (text[0] == '0'):
                        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
                        cv2.putText(image, text[0:9], (tl[0], tl[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        outputFileName = os.path.join(outputFolderPath, str(index) + ".jpg")
        cv2.imwrite(outputFileName, image)
        end = time.time()
        print(str.format('Processed no. {} image in {:.2f} ms',
                         index, (end - start) * 1000))


if __name__ == "__main__":
    OCR_WithOutput('D:\Github\ElectricityMeterOCRPython\Images',
                   outputFolderPath='D:\Github\ElectricityMeterOCRPython\Output')
