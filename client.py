import requests

url = 'http://127.0.0.1:5000/'
f = open('file.txt', 'r')
files = {'file': f}
# values = {'DB': 'photcat', 'OUT': 'csv', 'SHORT': 'short'}

r = requests.post(url, files=files)
