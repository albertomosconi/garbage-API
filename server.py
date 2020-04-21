# from flask import Flask, request
# app = Flask(__name__)


# @app.route('/', methods=['GET', 'POST'])
# def hello_world():
#     if request.method == 'GET':
#         return "hello world!"
#     elif request.method == 'POST':
#         return request.files
import os
from flask import Flask, flash, request, jsonify, redirect, url_for, Response
from werkzeug.utils import secure_filename

from PIL import Image
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import base64

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])


def image_loader(loader, image_name):
    image = Image.fromarray(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


UPLOAD_FOLDER = 'prova/'
ALLOWED_EXTENSIONS = {"jpg", "png", "jpeg"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        r = request
        image = r.files.get("photo", '')
        content = image.read()
        # convert string of image data to uint8
        # print(data['image'])
        # d = base64.decodebytes(data['image'])
        with open("aaa.jpg", "wb") as f:
            f.write(content)
        # print(str(content))
        nparr = np.frombuffer(content, dtype=np.uint8)
        print(nparr, len(nparr))
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        width = len(img[0])
        cropped_image = img[int(width/2):-int(width/2)]
        # cv2.imshow('image', cropped_image)
        # cv2.waitKey(0)

        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model.fc = nn.Linear(num_ftrs, 6)
        model.load_state_dict(torch.load("model.pt", map_location='cpu'))
        model.eval()

        pred = model(image_loader(
            data_transforms['val'], cropped_image)).detach().numpy()
        print(pred)

        predicted_class = str(np.argmax(pred))
        if predicted_class == '0':
            predicted_class = 'cardboard'
        elif predicted_class == '1':
            predicted_class = 'glass'
        elif predicted_class == '2':
            predicted_class = 'metal'
        elif predicted_class == '3':
            predicted_class = 'paper'
        elif predicted_class == '4':
            predicted_class = 'plastic'
        elif predicted_class == '5':
            predicted_class = 'trash'

        return jsonify('{"result":"' + predicted_class + '"}')
    elif request.method == 'GET':
        response = '{"name": "Brian", "city": "Seattle"}'
        return jsonify(response)


@app.route('/store', methods=['GET', 'POST'])
def store_file():
    if request.method == 'POST':
        r = request
        print(r, r.files)
        image = r.files.get("photo", '')
        label = r.get("label")
        width = r.get("width")
        content = image.read()
        print(label, width)
        # convert string of image data to uint8
        # print(data['image'])
        # d = base64.decodebytes(data['image'])

        # lokking for he highest number in the label
        max = 0
        dir = "/saved_photos/"+label

        for dirname, _, filenames in os.walk(dir):
            for filename in filenames:
                number = os.path.splitext(filename)[0]
                n = int(number)
                if(n > max):
                    max = n

        # for file in os.listdir(dir):
         #   number = os.path.splitext(file)[0]
          #  n = int(number)
           # if(n > max):
            #    max = n
        number = dir + "user" + label + str(max+1) + ".jpg"
        print(number)
        width = len(content)
        cropped_image = content[int(width/2):-int(width/2)]

        with open(number, "wb") as f:
            f.write(cropped_image)

        # cv2.imshow('image', cropped_image)
        # cv2.waitKey(0)


if __name__ == "__main__":
    app.run(host="192.168.178.106", port=5010)
