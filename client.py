import requests

url = 'http://127.0.0.1:5000/'

files = {'upload_file': open('file.txt', 'rb')}
# values = {'DB': 'photcat', 'OUT': 'csv', 'SHORT': 'short'}

r = requests.post(url, files=files)
