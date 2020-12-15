import cv2
import easyocr
import base64
from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
import io
import numpy as np

# Doan ma khoi tao server
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


bordersize = 10


def cleanup_text(result):
    ret_text = ""
    for r in result:
        ret_text += r[1]
    # using OpenCV
    return ret_text


# need to run only once to load model into memory
reader = easyocr.Reader(['en'], gpu=False)

CONFIDENCE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.5

net = cv2.dnn.readNet("Model\yolov4-custom_last.weights",
                      "Model\yolov4-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255)


def from_base64_to_image(staff_pix):
    try:
        staff_pix = np.fromstring(base64.b64decode(staff_pix), dtype=np.uint8)
        staff_pix = cv2.imdecode(staff_pix, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return staff_pix

# Khai bao ham xu ly request hello_word


@app.route('/mark', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def mark():
    # Read image
    try:

        request_json = request.json
        print("In = ", str(request_json)[:100])
        frame = request_json["frame"]
        frame = from_base64_to_image(frame)

        _, _, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        if len(boxes) != 1:
            return "E98"
        else:
            for box in boxes:
                frame = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

                frame = cv2.copyMakeBorder(
                    frame,
                    top=bordersize,
                    bottom=bordersize,
                    left=bordersize,
                    right=bordersize,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

                result = reader.readtext(
                    frame, allowlist='0123456789', text_threshold=0.3)
                result = cleanup_text(result)
                print(result)
                return result
    except:
        return "E99"

    return ""


# Thuc thi server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000, use_reloader=False)
