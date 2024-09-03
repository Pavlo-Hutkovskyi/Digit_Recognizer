import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

df = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sub = pd.read_csv('data/sample_submission.csv')

x_train = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
y_train = to_categorical(df.iloc[:, 0].values, num_classes=10)

scaler = MinMaxScaler()
x_train = x_train.reshape(-1, 28 * 28)
x_train = scaler.fit_transform(x_train).reshape(-1, 28, 28, 1)

test_x = test.values.reshape(-1, 28 * 28)
test_x = scaler.transform(test_x).reshape(-1, 28, 28, 1)

datagen = ImageDataGenerator(
    # rotation_range=10,
    # width_shift_range=0.2,
    # height_shift_range=0.1,
    # zoom_range=0.1,
    validation_split=0.2
)


def custom_function(x):
    return x ** 3 + x


train_generator = datagen.flow(x_train, y_train, batch_size=32, subset='training')
validation_generator = datagen.flow(x_train, y_train, batch_size=32, subset='validation')

# model = tf.keras.Sequential([
#     Conv2D(8, (3, 3), padding='same', input_shape=(28, 28, 1)),
#     Dropout(0.1),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2, 2), strides=2),
#     Conv2D(16, (3, 3), padding='same'),
#     Dropout(0.1),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2, 2), strides=2),
#     Flatten(),
#     Dropout(0.2),
#     Dense(100, activation='relu'),
#     Dense(10, activation='softmax')
# ])

# model = tf.keras.Sequential([
#     Conv2D(16, (3, 3), padding='same', input_shape=(28, 28, 1)),
#     # Dropout(0.1),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2, 2), strides=2),
#     Conv2D(16, (3, 3), padding='same'),
#     # Dropout(0.1),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2, 2), strides=2),
#     # Conv2D(16, (3, 3), padding='same'),
#     # # Dropout(0.1),
#     # Activation('relu'),
#     # MaxPooling2D(pool_size=(2, 2), strides=2),
#     Flatten(),
#     Dropout(0.3),
#     Dense(144, activation='relu'),
#     Dense(10, activation='softmax')
# ])

model = tf.keras.Sequential([
    Conv2D(16, (3, 3), padding='same', input_shape=(28, 28, 1), kernel_initializer='he_normal'),
    Dropout(0.2),
    # BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'),
    Dropout(0.2),
    # BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'),
    Dropout(0.2),
    # BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dropout(0.3),
    Dense(200, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(train_generator, validation_data=validation_generator, epochs=100, callbacks=[early_stopping])

for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        max_weights = [np.max(w) for w in weights]
        print('Max: ', layer.get_config()['name'], max_weights)
        min_weights = [np.min(w) for w in weights]
        print('Min: ', layer.get_config()['name'], min_weights)
        mean_weights = [np.mean(w) for w in weights]
        print('Mean: ', layer.get_config()['name'], mean_weights)
    else:
        print(layer.get_config()['name'], "No weights")

model.save('Model.keras')

plt.figure(figsize=(13, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.grid()
plt.show()

plt.figure(figsize=(13, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.grid()
plt.show()

pred = model.predict(test_x, verbose=1)
pred_labels = np.argmax(pred, axis=1)
sub['Label'] = pred_labels
sub.to_csv("data/sample_submission.csv", index=False)
print(sub.head())
