import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join

DATA = 'Predict/'

model = tf.keras.models.load_model(f'Models/dog_vs_cat.h5')
classes = ['Cat', 'Dog']

files = [join(DATA, f) for f in listdir(DATA) if isfile(join(DATA, f))]

for x in files:
    img = tf.keras.preprocessing.image.load_img(x, target_size=(200, 200))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0)
    prediciton = model.predict(img)
    score = tf.nn.softmax(prediciton[0])

    print(f'Image: {x.split("/")[1]} | Category: {classes[np.argmax(score)]} | Confidence: {np.round(100 * float(np.max(score)), 2)}%')