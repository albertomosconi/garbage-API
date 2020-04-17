import json
import requests
import cv2

url = 'http://192.168.0.3:5010/'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('busta.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
# response = requests.post(
#     url, data=img_encoded.tostring(), headers=headers)

res = requests.get(url)
# decode response
print(res.text)
