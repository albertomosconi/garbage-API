# from flask import Flask, request
# app = Flask(__name__)


# @app.route('/', methods=['GET', 'POST'])
# def hello_world():
#     if request.method == 'GET':
#         return "hello world!"
#     elif request.method == 'POST':
#         return request.files
import os
from flask import Flask, flash, request, redirect, url_for, Response
from werkzeug.utils import secure_filename

from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

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
        # convert string of image data to uint8
        nparr = np.fromstring(r.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model.fc = nn.Linear(num_ftrs, 6)
        model.load_state_dict(torch.load("model.pt"))
        model.eval()
        print(
            model(image_loader(data_transforms['val'], img)).detach().numpy())

        response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                    }
        return Response(response=response, status=200, mimetype="application/json")

    return Response(response={'request failed'}, status=400)
