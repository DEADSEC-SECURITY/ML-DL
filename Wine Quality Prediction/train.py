import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split

data = pd.read_csv('Data/winequality-red.csv', sep=';')

target = data['quality']
data.pop('quality')

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33)

model = Sequential()

hidden_layers = 100

model.add(Input(shape=x_train.shape[1]))

for x in range(hidden_layers):
    model.add(Dense(x_train.shape[1]*8, activation='relu'))

model.add(Dense(1))

print(model.summary())

optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mse'])

log_dir = f'Logs/fit/'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

fitted_model = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=500,
    callbacks=[tensorboard_callback, tf.keras.callbacks.ModelCheckpoint("Models/Checkpoint/save_at_{epoch}.h5")],
)

model.save(f'Models/dog_vs_cat.h5')

prediction = model.predict([[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4],
                            [9.4, 0.3, 0.56, 2.8, 0.08, 6, 17, 0.9964, 3.15, 0.92, 11.7],
                            [8.3, 1.02, 0.02, 3.4, 0.084, 6, 11, 0.99892, 3.48, 0.49, 11],
                            [7.3, 0.305, 0.39, 1.2, 0.059, 7, 11, 0.99331, 3.29, 0.52, 11.5]]).flatten()

prediction = np.round(prediction)

true_prediction = [5, 8, 3, 6]

print(prediction)
print(true_prediction)