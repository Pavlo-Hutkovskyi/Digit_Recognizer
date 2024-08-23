import numpy as np
import tensorflow as tf
import pandas as pd
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, \
    GlobalMaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Завантаження даних
df = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sub = pd.read_csv('data/sample_submission.csv')

x_train = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
y_train = to_categorical(df.iloc[:, 0].values, num_classes=10)

# Використання MinMaxScaler для нормалізації
scaler = MinMaxScaler()
x_train = x_train.reshape(-1, 28 * 28)
x_train = scaler.fit_transform(x_train).reshape(-1, 28, 28, 1)

# Підготовка тестових даних
test_x = test.values.reshape(-1, 28 * 28)
test_x = scaler.transform(test_x).reshape(-1, 28, 28, 1)

plt.figure(figsize=(15, 8))

for i in range(100):
    # Відображаємо зображення у сітці 5x10
    plt.subplot(10, 10, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')

    # Вимикаємо осі
    plt.axis('off')

    # Додаємо лейбл як підпис до зображення
    plt.title(np.argmax(y_train[i]))

plt.tight_layout()  # Оптимізуємо розміщення зображень на фігурі
plt.show()

plt.figure(figsize=(15, 8))

for i in range(100):
    # Відображаємо зображення у сітці 5x10
    plt.subplot(10, 10, i + 1)
    plt.imshow(test_x[i].reshape(28, 28), cmap='gray')

    # Вимикаємо осі
    plt.axis('off')

    # Додаємо лейбл як підпис до зображення
    plt.title('a')

plt.tight_layout()  # Оптимізуємо розміщення зображень на фігурі
plt.show()

# Використання ImageDataGenerator для аугментації
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2  # Для створення валідаційного набору
)


def custom_function(x):
    return x ** 3 + x


# Генератори для тренувальних та валідаційних даних
train_generator = datagen.flow(x_train, y_train, batch_size=32, subset='training')
validation_generator = datagen.flow(x_train, y_train, batch_size=32, subset='validation')

# Створення моделі
# model = tf.keras.Sequential([
#     Conv2D(4, (3, 3), padding='same', input_shape=(28, 28, 1)),
#     # Conv2D(4, (3, 3), padding='same'),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2, 2), strides=2),
#     Conv2D(4, (3, 3), padding='same'),
#     # Conv2D(4, (3, 3), padding='same'),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2, 2), strides=2),
#     Flatten(),
#     Dense(30, activation='relu'),
#     Dense(10, activation='softmax')
# ])

model = tf.keras.Sequential([
    Conv2D(16, (3, 3), padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(32, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dropout(0.3),
    Dense(200, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

# Компіляція моделі
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Використання EarlyStopping для запобігання переобученню
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Навчання моделі з використанням генераторів
history = model.fit(train_generator, validation_data=validation_generator, epochs=100, callbacks=[early_stopping])

for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        mean_weights = [np.mean(w) for w in weights]
        print(layer.get_config()['name'], mean_weights)
    else:
        print(layer.get_config()['name'], "No weights")

model.save('Model.keras')

# Візуалізація результатів навчання
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

# Прогнозування на тестових даних
pred = model.predict(test_x, verbose=1)
pred_labels = np.argmax(pred, axis=1)
sub['Label'] = pred_labels
sub.to_csv("data/sample_submission.csv", index=False)
print(sub.head())

image = (np.array(Image.open('C:\\Users\\pasag\\Desktop\\digit photo\\7.jpg').convert('L').resize((28, 28)))
         .reshape(1, 28, 28, 1))
print(np.argmax(model.predict(image, verbose=1), axis=1))
