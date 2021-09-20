from flask import Flask, render_template, request
#from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys 
import os
import base64
sys.path.append(os.path.abspath("./model"))
#from load_model import * 
import cv2
import tensorflow as tf
from keras.models import model_from_json
from PIL import Image

#Initialize the useless part of the base64 encoded image.
init_Base64 = 21

global graph, model


app = Flask(__name__)

@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/predict/',methods=['POST'])
def predict():
 
    parseImage(request.get_data())
    img = Image.open("output.png")
    resized_img = img.resize((28, 28))
    resized_img.save("resized_image.png")

    # read parsed image back in 8-bit, black and white mode (L)
    image = cv2.imread('resized_image.png',0)
    x = image
    x = np.invert(x)
    # x = resize(x,( 28, 28))
    resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
    vect = np.asarray(resized, dtype="uint8")
    x = vect.reshape(1, 28, 28, 1).astype('float32')
    x = x.reshape(1,28,28,1)
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        json_file = open('model/model.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model/weights.h5")
        loaded_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        out = loaded_model.predict(x)
        print('output',out)
        index = np.argmax(out[0])
        print(index)

        response = np.array_str(np.argmax(out,axis=1))
        return response	

def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
