import base64
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, Response, flash, jsonify, redirect, request, url_for
from PIL import Image
from torchvision import models, transforms
from werkzeug.utils import secure_filename

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


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        r = request
        image = r.files.get("photo", '')
        content = image.read()

        with open("aaa.jpg", "wb") as f:
            f.write(content)

        nparr = np.frombuffer(content, dtype=np.uint8)
        print(nparr, len(nparr))

        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        width = len(img[0])
        cropped_image = img[int(width/2):-int(width/2)]

        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features

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


if __name__ == "__main__":
    app.run(host="192.168.0.9", port=5010)
