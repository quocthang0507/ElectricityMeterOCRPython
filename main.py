# Chạy code trên terminal bằng câu lệnh sau
# python easy_ocr.py --image images/arabic_sign.jpg --langs en,ar

# import the necessary packages
from easyocr import Reader
import argparse
import cv2
import os


def cleanup_text(text):
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


def ocr(imagePath, langs="en"):
    filename = os.path.basename(imagePath)
    image = cv2.imread(imagePath)
    reader = Reader(langs, False)  # Without gpu
    results = reader.readtext(image)

    for (bbox, text, prob) in results:
        print("[INFO] {:.4f}: {}".format(prob, text))

        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        text = cleanup_text(text)
        if len(text) in range(4, 20):
            if ((tl[0] > 20) and (tr[0] > 20)):
                if (text[0] == '0'):
                    cv2.rectangle(image, tl, br, (0, 255, 0), 2)
                    cv2.putText(image, text[0:9], (tl[0], tl[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite('_' + filename, image)
