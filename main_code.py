import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
#Создадим некоторые глобальные переменные
data = []
labels = []
#Укажем количество классов дорожных знаков используемых в нашем датафрейме и укажем абсолютный путь к файлу проекта
classes = 43
cur_path = os.getcwd()

#Добавление изображения, изменение его размеров.
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            #Добавляем изобаржения в метки data и labels
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

#Конвертация данных в numpy
data = np.array(data)
labels = np.array(labels)
#Выведем информацию о датафрейме
print(data.shape, labels.shape)
#Разделение набора данных для обучения и тестирования
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)


#Построим модель CNN и добавим слои
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Связь модели с оптимизатором,функцией потерь и метриками
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 20
#Сохраняем историю модели обучения
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")

#График точности обучения и проверки.
plt.figure(0)
plt.plot(history.history['accuracy'], label='Точность обучения')
plt.plot(history.history['val_accuracy'], label='Точность проверки')
plt.title('Точность обучения')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='Потеря при обучение')
plt.plot(history.history['val_loss'], label='Потеря при проверки')
plt.title('Тестирование потерь за эпохи')
plt.xlabel('Эпохи')
plt.ylabel('Потеря')
plt.legend()
plt.show()




# plt.figure(3)
# plt.plot(history.history[''], label='')
# plt.plot(history.history[''], label='')
# plt.title('')
# plt.xlabel('')
# plt.ylabel('')
# plt.legend()
# plt.show()
