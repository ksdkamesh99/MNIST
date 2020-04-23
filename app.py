from flask import Flask, render_template,request
from keras.preprocessing import image#for matrix math
import numpy as np
import keras.models as km
import re
import sys
import base64
from skimage import transform
import matplotlib.pyplot as plt

import os
app = Flask(__name__)
model=km.load_model('model.h5')
def images():
    image_read=[]
    image1=image.load_img("output.png")
    image2=image.img_to_array(image1)
    image3=transform.resize(image2,(28,28,1),anti_aliasing=True)
    
    image4=image3/255
    image_read.append(image4)
    img_array=np.asarray(image_read)
    return img_array
def parseImg(imgData):
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    print(imgstr)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        img=request.get_data()
        parseImg(img)
        img_array=images()
        out = model.predict(img_array)
        print(out)
        predict= np.argmax(out,axis=1)
        response=str(predict[0])
        print(predict)
        return response
	
    
    
	

if __name__ == "__main__":
    app.run(debug=True)
