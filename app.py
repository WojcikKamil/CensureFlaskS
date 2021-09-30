import base64
import io
import sys
import cv2
import cv2 as cv
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from disp import Disp

app = Flask(__name__)


@app.route('/Censure', methods=['POST'])
def censure_image():
    censure = cv.CascadeClassifier('static/cascade/cascade.xml')
    disp_censure = Disp(None)
    # print(request.files , file=sys.stderr)
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # fixing color bug
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv.imshow(npimg)
    rectangles = censure.detectMultiScale(img)
    detection_image = disp_censure.draw_rectangles(img, rectangles)
    cv.imshow('Matches', detection_image)

    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status': str(img_base64)})

@app.route('/')
def censure():
    return render_template('./censure.html')

@app.route('/index')
def home():
    return render_template('./index.html')

@app.route('/aboutus')
def aboutus():
    return render_template('./aboutus.html')

@app.route('/contact')
def contact():
    return render_template('./contact.html')

@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
    app.run(debug=True)
