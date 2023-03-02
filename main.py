from flask import Flask, jsonify
from flask import request
from PIL import Image
import torch
import base64
import pandas as pd
import numpy as np
import pandas
import cv2
import io
import json
import statistics
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
def calculate_measure(image,coordinate):
    CroppedImage = image.crop((int(coordinate[0]["xmin"]), int(coordinate[0]["ymin"]),int(coordinate[0]["xmax"]) ,int(coordinate[0]["ymax"])))
    rate  =[]
    threshold_ranges=range(40,161,5)
    originalImage = CroppedImage
    originalImage = np.array(originalImage, dtype="uint8")
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    for i in threshold_ranges:
        try:
            (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, i, 255, cv2.THRESH_BINARY)
            ds = pd.DataFrame(blackAndWhiteImage)
            ds = ds.loc[:,:len(ds.columns)//2]
            black = ds.sum().sum()/255
            white = (ds.shape[0]*ds.shape[1]*255-ds.sum().sum())/255
            if black==0:
                black = 0.00001
            if white==0:
                white = 0.00001
            rate.append(abs(1.50-black/white))
        except:
            rate.append(0)
    threshold = threshold_ranges[np.argmin(rate)]
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, threshold, 255, cv2.THRESH_BINARY)

    ds = pd.DataFrame(blackAndWhiteImage)
    ds = ds.loc[:,:len(ds.columns)//2]
    substring = "0,0,0,255,255,255"
    count = []

    for i in ds.columns:
        listToStr = ','.join(map(str, ds.loc[:,i]))
        count.append(listToStr.count(substring))
    cm = (statistics.mode(count))*2   
    calculation=100-cm

    return calculation, threshold



@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/ProcessImage", methods=['POST'])
def process_image():
    confidence = 0
    xmax = 0
    xmin = 0
    ymax = 0
    ymin = 0
    threshold = 0
    calculation = 0
    error = "" 
    try :         
        image = request.files['image']
        image_bytes = image.read()
        img = Image.open(io.BytesIO(image_bytes))

        results = model(img)  
        resultJson = results.pandas().xyxy[0].to_json(orient="records")
        data = json.loads(resultJson)
        calculation, threshold = calculate_measure(img,data) 
        confidence =  data[0]["confidence"]
        xmax = data[0]["xmax"]
        xmin = data[0]["xmin"]
        ymax = data[0]["ymax"]
        ymin = data[0]["ymin"]
        
    except Exception as e:
        error = str(e)

    return jsonify({'Confidence':confidence,'Xmax': xmax,'Xmin': xmin ,'Ymax':ymax ,'Ymin': ymin ,'Threshold': threshold,'Result': calculation,'Error': error})

if __name__ == "__main__":
    model = torch.hub.load("C:/Users/huseyin_urtekin/PycharmProjects/pythonProject1", 'custom', "best.pt",
                           source='local',
                           autoshape=True)
    model.confidence = 0.85
    app.run(host="0.0.0.0", port=5000, debug=True)
