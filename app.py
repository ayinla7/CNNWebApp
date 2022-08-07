from flask import Flask, redirect, render_template, request, send_from_directory, url_for
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import random
import string

## # # # # #
import os
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Convolution2D,
    Dropout,
    Conv2D,
    MaxPool2D,
    AveragePooling2D,
    BatchNormalization,
    Flatten,
    GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
# Flask utils
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Define a flask app

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

# # # # # # # # # # # # # # # # # # # # # #
COUNT = 0

@app.route("/")
def hello_world():
    return render_template("index.html")


NAME = ""
PATH = ""
i = 0
j = 0

finalLabels = []
finalnp = np.load('CNNlabels5.npy')
# finalLabels = np.sort(finalnp)\
#
finalLabels = finalnp



def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


@app.route('/prediction2', methods=['POST'])
def predictor2():
    global j, COUNT
    if j == 0:
        global model2
        model2 = load_model("static/models/CustomCNNmodel5.h5")
    image = request.files['image']
    name = get_random_string(8)
    global NAME
    NAME = name
    print(f"Name = {name}")
    path = fr'static/user_images/{name}.jpg'
    global PATH
    PATH = path
    image.save(path)
    # image = Image.open(image.stream)
    # image = image.resize((100, 100))
    # image = image.convert('RGB')
    # image.save(path)

    # image = cv2.imread(path)
    #
    # img = np.array(image) / 255.0
    # print(img.shape)

    image = cv2.imread(path)


    img_arr = cv2.imread(path)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    ary = Image.fromarray(img_arr, 'RGB')
    img = np.array(ary.resize((100, 100)))
    print(img.shape)

    la = LabelEncoder()

    train_labels = np.load("CNNlabels5.npy")

    arr = []
    arr = np.load("CNNlabels5.npy")
    arr = np.sort(arr)


    labels = pd.DataFrame(train_labels)
    print(labels)
    print(arr[0])
    train_labels_encoded = la.fit_transform(labels[0])


    img = np.expand_dims(img, axis=0)

    # la = LabelEncoder()
    # pred_t = np.argmax(model2.predict(img), axis=1)
    # prediction_t = la.inverse_transform(pred_t)
    #
    # print(pred_t)
    # COUNT += 1
    # print(prediction_t[0])

    result = model2.predict(img)
    pred_t = np.argmax(result, axis=1)
    prediction_t = la.inverse_transform(pred_t)





    result = result.squeeze()
    print(result)

    COUNT += 1

    # print(arr)
    # print(train_labels_encoded);
    print(prediction_t)

    result = np.array(result.tolist())
    count = 0
    for x in result:
        print(x)
        count=count+1
        if x > 0.6:
            print("Found")
            # print(result[x])
            p = x.squeeze() * 100
            print(str(p) + "%")
            print("final" + str(arr[[count-1]]))

            # print(finalLabels)
            return render_template('prediction2.html', data=[str(prediction_t[0]), name, str(p) + "%"])
            break
    else:
        p = 100 - (x.squeeze() * 100)
        print("No Fruits")
        print(str(p) + "%")
        return render_template('prediction2.html', data=[f"No fruits or Vegetable detected", name, "0" ])






@app.route('/load_img')
def load_img():
    # return render_template('index.html', data=PATH)
    global PATH, NAME
    print(PATH, NAME)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'{NAME}.jpg')
    print(full_filename)
    render_template('prediction1.html', name=NAME)
    # return redirect(url_for('static', path=PATH), code=301)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    app.config['UPLOAD_FOLDER'] = PATH
    # app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

