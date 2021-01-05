# Chạy code trên terminal bằng câu lệnh sau
# python easy_ocr.py --image images/arabic_sign.jpg --langs en,ar

# import the necessary packages
from easyocr import Reader
import argparse
import cv2


def cleanup_text(text):
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


# Phần khai báo các đối số
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image to be OCR'd")
ap.add_argument("-l", "--langs", type=str, default="en",
                help="comma separated list of languages to OCR")
ap.add_argument("-g", "--gpu", type=int, default=-1,
                help="whether or not GPU should be used")
args = vars(ap.parse_args())

# Tách phần chữ nếu có dấu ','
langs = args["langs"].split(",")
print("[INFO] OCR'ing with the following languages: {}".format(langs))

# Cho ảnh đàu vào
image = cv2.imread(args["image"])

# Đọc ảnh sử dụng OCR
print("[INFO] OCR'ing input image...")
reader = Reader(langs, gpu=args["gpu"] > 0)
results = reader.readtext(image)

# Lặp lại các kết quả thu được
for (bbox, text, prob) in results:
    # Hiển thị xác xuất để thu được ở các phần text
    print("[INFO] {:.4f}: {}".format(prob, text))

    # Lấy toạ độ các bounding box
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))

    # Một vài điều kiện để chọn đầu ra
    text = cleanup_text(text)
    if len(text) in range(4, 20):
        if ((tl[0] > 20) and (tr[0] > 20)):
            if (text[0] == '0'):
                cv2.rectangle(image, tl, br, (0, 255, 0), 2)
                cv2.putText(image, text[0:9], (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

filename = 'savedImage_7.jpg'

cv2.imwrite(filename, image)
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
# Đọc code bằng câu lênh ở terminal
# python easy_ocr.py --image images/arabic_sign.jpg --langs en,ar
