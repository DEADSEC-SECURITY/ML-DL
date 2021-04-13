import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

DATA = 'Data/'
BATCH_SIZE = 64
VAL_SPLIT = 0.2

DATA = pathlib.Path(DATA)

train = tf.keras.preprocessing.image_dataset_from_directory(
    DATA,
    validation_split=VAL_SPLIT,
    subset='training',
    seed=3391283,
    image_size=(200, 200),
    batch_size=BATCH_SIZE
)

val = tf.keras.preprocessing.image_dataset_from_directory(
    DATA,
    validation_split=VAL_SPLIT,
    subset='validation',
    seed=3391283,
    image_size=(200, 200),
    batch_size=BATCH_SIZE
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)

model.compile(optimizer=optimizer,
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

fitted_model = model.fit(
    train,
    validation_data=val,
    epochs=20
)

model.save(f'Models/dog_vs_cat.h5')

plt.title('Loss')
plt.plot(fitted_model.history['loss'], label='loss')
plt.plot(fitted_model.history['val_loss'], label='val_loss')
plt.show()

plt.title('Accuracy')
plt.plot(fitted_model.history['accuracy'], label='acc')
plt.plot(fitted_model.history['val_accuracy'], label='val_acc')
plt.show()