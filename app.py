from flask import Flask, render_template, request
#from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import base64

from keras.models import model_from_json
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import model_from_json 
import PIL.ImageOps

app = Flask(__name__)

@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/predict/',methods=['POST'])
def predict():
 
    parseImage(request.get_data())
    # img = Image.open("output.png")
    # resized_img = img.resize((28, 28))
    # resized_img.save("resized_image.png")

    # read parsed image back in 8-bit, black and white mode (L)
    # image = cv2.imread('resized_image.png',0)
    image = Image.open('output.png')

    if image.mode == 'RGBA':
        r,g,b,a = image.split()
        rgb_image = Image.merge('RGB', (r,g,b))

        inverted_image = PIL.ImageOps.invert(rgb_image)

        r2,g2,b2 = inverted_image.split()

        final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))

        final_transparent_image.save('inverted.png')

    else:
        inverted_image = PIL.ImageOps.invert(image)
        inverted_image.save('inverted.png')

    img = load_image('inverted.png')
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()

    # use Keras model_from_json to make a loaded model
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded Model from disk")

    # compile and evaluate loaded model

    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    digit = loaded_model.predict(img)
    print(digit[0])
    response = np.array_str(np.argmax(digit,axis=1))
    prob = {}
    
    label = 0
    for i in digit[0]:
        prob[str(label)] = str(i)
        label = label + 1

    ans = {}
    ans['probability'] = prob
    ans['result'] = response[1]
    return ans


def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

if __name__ == "__main__":
    app.run()
