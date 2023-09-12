# code to test the API working 
# send request to localhost:5000/predict using curl or postman
# curl -X POST -F image=@<path_to_image> localhost:5000/predict

import requests

resp = requests.post("http://localhost:5000/predict", files={"image": open("test_image.JPG", "rb")})
print(resp.json())
