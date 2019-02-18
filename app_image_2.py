from flask import Flask, render_template, request
import base64
from PIL import Image
import io
import re
import numpy as np
import cv2
import PIL
import numpy
import math
from scipy import ndimage

import sys
sys.path.insert(0, "C://Users//Alvertos//Documents//portfolio//convolution//convolution")

from data_process import data_pre_process
from train import make_prediction




app = Flask(
    __name__)

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


@app.route('/')
def index():
    return render_template("canvas_flask.html")





@app.route("/saveimg", methods=['GET', 'POST'])
def get_image():
    if request.method == "POST":
        image_b64 = request.form["img"]
        image_data = re.sub('^data:image/.+;base64,', '', image_b64)
        image_data = image_data.replace(" ", "+")
        image_data_2 = base64.b64decode(image_data)
        image_PIL = Image.open(io.BytesIO(image_data_2))
        image_np = np.array(image_PIL)[:, :, 1:4]
        img = Image.fromarray(image_np, 'RGB')

        pil_image = PIL.Image.open(io.BytesIO(image_data_2))

        gray = data_pre_process(pil_image)
        make_prediction(gray)
        cv2.imshow('ImageWindow', gray)
        cv2.waitKey()
        make_prediction(gray)
    

    return render_template("canvas_flassk.html")


if __name__ == '__main__':
    app.run(debug=True)



    