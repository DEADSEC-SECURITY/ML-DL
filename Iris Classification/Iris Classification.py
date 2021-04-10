import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
How to use:

- Training the model:
iris = IrisClass()
iris.train_model()
iris.plot_loss() # Will display a graph with the loss
iris.plot_accuracy() # Will display a graph with accuracy
iris.save_model() # Save model to file

- Make prediction:
iris = IrisClass()
iris.load_model()
# Example data to predict
data = {
    'SepalLengthCm': [6, 5],
    'SepalWidthCm': [3, 2],
    'PetalLengthCm': [5, 4],
    'PetalWidthCm': [2, 1]
}
data = pd.DataFrame(data=data)
predictions = iris.make_prediction(data)
print(predictions) # This returns a 2D array of predictions and percentages
"""

class IrisClass():
    def __init__(self, models_dir='Models', models_file_name='iris_model.h5', logging=True,
                 data_csv='Data/Iris.csv', log_file='Logs'):
        self.md_dir = models_dir
        self.md_file_name = models_file_name
        self.data_csv = data_csv
        self.log_file = log_file

        self.logging = logging

    def train_model(self):
        data = pd.read_csv(self.data_csv)
        data.pop('Id')
        target = data['Species']
        data.pop('Species')

        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(x_train.shape[1])),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        if self.logging:
            log_dir = f'{self.log_file}/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            self.fitted_model = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=300, callbacks=[tensorboard_callback])
        else:
            self.fitted_model = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=300)

    def plot_loss(self):
        plt.title('Loss')
        plt.plot(self.fitted_model.history['loss'], label='loss')
        plt.plot(self.fitted_model.history['val_loss'], label='val_loss')
        plt.show()

    def plot_accuracy(self):
        plt.title('Accuracy')
        plt.plot(self.fitted_model.history['accuracy'], label='acc')
        plt.plot(self.fitted_model.history['val_accuracy'], label='val_acc')
        plt.show()

    def save_model(self):
        self.model.save(f'{self.md_dir}/{self.md_file_name}')

    def load_model(self):
        self.model = tf.keras.models.load_model(f'{self.md_dir}/{self.md_file_name}')

    def make_prediction(self, data_set):
        p = self.model.predict(data_set)
        perc = [[j * 100 for j in i] for i in p]
        perc = np.round(perc)
        p = np.round(p)

        return {'percent': perc, 'predictions': p}
