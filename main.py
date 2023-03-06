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

def decition_measure(cm,calculation):   
    if len(np.where((min(cm)+10>np.array(calculation))&(np.array(calculation)>min(cm)-10))[0])>0:
        result = calculation[np.where((min(cm)+10>np.array(calculation))&(np.array(calculation)>min(cm)-10))[0][0]]
    else:
        result = min(cm)
    if result<0:
        result = 0
    plaka_no = np.where(min(cm)==np.array(cm))[0][0]
    return result,plaka_no

def detect_number(data,image,model_number):
    cm = []
    for i in range(len(data)):
        CroppedImage = image.crop((int(data[i]["xmin"]), int(data[i]["ymin"]),int(data[i]["xmax"]) ,int(data[i]["ymax"])))
        results = model_number(CroppedImage)
        ds = results.pandas().xyxy[0]
        if len(np.where(ds["class"].unique()>ds["class"].min())[0])>(10-ds["class"].min())*0.3:
            cm.append(ds["class"].min()*10)
        else:
            cm.append(10000000)
    
    return cm
    
def detect_line(data,image,cm):    
    calculation = []
    for i in range(len(data)):
        if (cm[i]>=0 )& (cm[i]<10000):
            CroppedImage = im.crop((int(data[i]["xmin"]), int(data[i]["ymin"]),int(data[i]["xmax"]) ,int(data[i]["ymax"])))
            rate  =[]
            threshold_ranges=range(40,161,5)
            originalImage = CroppedImage
            originalImage = np.array(originalImage, dtype="uint8")
            grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
            for z in threshold_ranges:
                try:
                    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, z, 255, cv2.THRESH_BINARY)
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
            substring = "0,255"
            count = []

            for z in ds.columns:
                listToStr = ','.join(map(str, ds.loc[:,z]))
                count.append(listToStr.count(substring))
            o = (statistics.mode(count))*2   
            calculation.append(100-o)
    return calculation , threshold

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
        cm = detect_number(data,im,model_number)
        calculation , threshold = detect_line(data,im,cm)
        result,plaka_no = decition_measure(cm,calculation) 
        confidence =  data[plaka_no]["confidence"]
        xmax = data[plaka_no]["xmax"]
        xmin = data[plaka_no]["xmin"]
        ymax = data[plaka_no]["ymax"]
        ymin = data[plaka_no]["ymin"]
        
    except Exception as e:
        error = str(e)

    return jsonify({'Confidence':confidence,'Xmax': xmax,'Xmin': xmin ,'Ymax':ymax ,'Ymin': ymin ,'Threshold': threshold,'Result': result,'Error': error})

if __name__ == "__main__":
    model = torch.hub.load("C:/Users/huseyin_urtekin/PycharmProjects/pythonProject1", 'custom', "best.pt",
                           source='local',
                           autoshape=True)
    model.confidence = 0.50
    
    model_number = torch.hub.load("C:/Users/huseyin_urtekin/PycharmProjects/pythonProject1", 'custom', "best1.pt",
                       source='local',
                       autoshape=True)
    model_number.confidence = 0.50
    app.run(host="0.0.0.0", port=5000, debug=True)
