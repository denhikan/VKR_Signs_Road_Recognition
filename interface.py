import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np

# загрузка обученной модели
from keras.models import load_model

model = load_model('my_model.h5')

# Создадим классы дорожных знаков
classes = {1: 'Ограничение скорости (20км/ч)',
           2: 'Ограничение скорости (30км/ч)',
           3: 'Ограничение скорости (50км/ч)',
           4: 'Ограничение скорости (60км/ч)',
           5: 'Ограничение скорости (70км/ч)',
           6: 'Ограничение скорости (80км/ч)',
           7: 'Ограничение скорости (80км/ч)',
           8: 'Ограничение скорости (100км/ч)',
           9: 'Ограничение скорости (120км/ч)',
           10: 'Не тот знак',
           11: 'Не тот знак',
           12: 'Не тот знак',
           13: 'Не тот знак',
           14: 'Не тот знак',
           15: 'Не тот знак',
           16: 'Не тот знак',
           17: 'Не тот знак',
           18: 'Не тот знак',
           19: 'Не тот знак',
           20: 'Не тот знак',
           21: 'Не тот знак',
           22: 'Не тот знак',
           23: 'Не тот знак',
           24: 'Не тот знак',
           25: 'Не тот знак',
           26: 'Не тот знак',
           27: 'Не тот знак',
           28: 'Не тот знак',
           29: 'Не тот знак',
           30: 'Не тот знак',
           31: 'Не тот знак',
           32: 'Не тот знак',
           33: 'Не тот знак',
           34: 'Не тот знак',
           35: 'Не тот знак',
           36: 'Не тот знак',
           37: 'Не тот знак',
           38: 'Не тот знак',
           39: 'Не тот знак',
           40: 'Не тот знак',
           41: 'Не тот знак',
           42: 'Не тот знак',
           43: 'Не тот знак',
           }

# Создадим графическое окно для отображения интерфейса
top = tk.Tk()
top.geometry('1200x720')
top.title('Никитин Д.В.')
top.configure(background='#ffffff')
label = Label(top, background='#ffffff',  font=('Calibre', 17, 'bold'))
sign_image = Label(top)


# Создадим вывод и классификацию выходных данных
def classify(file_path):
    global label_packed
    img = Image.open(file_path)
    img = img.resize((30, 30))
    img = np.expand_dims(img, axis=0)
    img = np.array(img)
    print(img.shape)
    predict_classes = model.predict([img])[0]
    pred = np.argmax(predict_classes)
    sign = classes[pred + 1]
    print(sign)
    label.configure(foreground='#011638', text=sign)


# создание кнопки вывода результата
def show_classify_button(file_path):
    classify_b = Button(top, text="Распознать", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#FA8072', foreground='white', font=('Calibre', 13, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


# Создадим кнопку загрузки изображения
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload = Button(top, text="Загрузить изображение", command=upload_image, padx=10, pady=5)
upload.configure(background='#FA8072', foreground='white', font=('Calibre', 15, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

# Укажем заголовок
heading = Label(top, text="Распознование дорожного знака", pady=20, font=('Calibre', 20, 'bold'))
heading.configure(background='#ffffff', foreground='#364156')
heading.pack()
top.mainloop()