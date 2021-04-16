import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

DATA = 'Data/'
BATCH_SIZE = 32
VAL_SPLIT = 0.2

DATA = pathlib.Path(DATA)

train = tf.keras.preprocessing.image_dataset_from_directory(
    DATA,
    validation_split=VAL_SPLIT,
    subset='training',
    seed=3,
    image_size=(200, 200),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
)

val = tf.keras.preprocessing.image_dataset_from_directory(
    DATA,
    validation_split=VAL_SPLIT,
    subset='validation',
    seed=3,
    image_size=(200, 200),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
)

model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(.1),
    tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)

model.compile(optimizer=optimizer,
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

log_dir = f'Logs/fit/' + '20210413-124250'#datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

fitted_model = model.fit(
    train,
    validation_data=val,
    epochs=20,
    use_multiprocessing=True,
    callbacks=[tensorboard_callback, tf.keras.callbacks.ModelCheckpoint("Models/Checkpoint/save_at_{epoch}.h5")],
)

model.save(f'Models/dog_vs_cat.h5')