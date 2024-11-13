import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
import time
from PIL import Image, ImageTk
import webbrowser
import customtkinter as ctk

# Создание директории для моделей, если она еще не существует
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

# Загрузка всех моделей из директории
model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.keras')]
models = {os.path.basename(path): load_model(path) for path in model_paths}


# Определение пути к изображению для PyInstaller
def resource_path(relative_path):
    """ Получает путь к ресурсу при сборке в один файл """
    try:
        # PyInstaller создает временный каталог _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath("..")
    return os.path.join(base_path, relative_path)


# Открытие GitHub-ссылки
def open_github():
    webbrowser.open("https://github.com/Tetsuyomi")


# GUI приложение на CustomTkinter
class ModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FreshCheck")
        self.root.geometry("670x520+600+200")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Верхняя шапка с меню и логотипом
        header_frame = ctk.CTkFrame(root, fg_color="#222831")
        header_frame.pack(fill="x")  # Заполнение по ширине окна

        # Невидимый элемент для увеличения высоты
        spacer = ctk.CTkLabel(header_frame, text="", height=70)
        spacer.grid(row=0, column=0)

        # Настройка сетки для центровки элементов в шапке
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=1)
        header_frame.grid_columnconfigure(2, weight=1)

        # Кнопка для меню с выпадающим списком
        self.menu_button = ctk.CTkButton(
            header_frame, text="≡", font=("Arial Rounded MT Bold", 16),
            command=self.show_menu, width=30
        )
        self.menu_button.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Надпись "FreshCheck" как логотип
        self.logo_label = ctk.CTkLabel(
            header_frame, text="FreshCheck", font=("Arial Rounded MT Bold", 25)
        )
        self.logo_label.grid(row=0, column=1, pady=5)

        # Стили и шрифты
        self.font_style = ("Arial Rounded MT Bold", 16)

        # Основная рамка для центровки остальных элементов
        main_frame = ctk.CTkFrame(root)
        main_frame.pack(expand=True)

        # Выпадающий список для выбора модели
        self.model_label = ctk.CTkLabel(main_frame, text="Выберите модель:", font=self.font_style)
        self.model_label.grid(row=0, column=0, padx=(30, 10), pady=(10, 10), sticky="ew")

        self.model_combobox = ctk.CTkComboBox(main_frame, values=list(models.keys()), width=200, height=60)
        self.model_combobox.grid(row=0, column=1, padx=10, pady=(10, 10), sticky="ew")

        # Кнопка для добавления новой модели
        self.add_model_button = ctk.CTkButton(
            main_frame, text="Добавить модель", command=self.add_model, font=self.font_style, width=200, height=60
        )
        self.add_model_button.grid(row=1, column=0, columnspan=2, pady=(10, 10), sticky="ew")

        # Кнопка для запуска выбранной модели
        self.run_model_button = ctk.CTkButton(
            main_frame, text="Запустить модель", command=self.run_model, font=self.font_style, width=200, height=60
        )
        self.run_model_button.grid(row=2, column=0, columnspan=2, pady=(10, 10), sticky="ew")

    def show_menu(self):
        # Создаем выпадающий список
        menu = tk.Menu(self.root, tearoff=0)

        # Настройка стиля выпадающего меню
        menu.config(
            bg='#222831',  # Фон, как у шапки программы
            fg='#ffffff',  # Белый цвет текста для соответствия темной теме
            activebackground='#30475e',  # Синий фон при наведении, как в стиле кнопок
            activeforeground='#ffffff'  # Белый цвет текста при наведении
        )

        # Добавляем пункты меню
        menu.add_command(label="Инструкция", command=self.rules_model_window)
        menu.add_command(label="GitHub создателя", command=open_github)
        menu.add_separator()
        menu.add_command(label="Будущие функции", state="disabled")

        # Показываем меню под кнопкой
        menu.post(self.root.winfo_rootx() + 20, self.root.winfo_rooty() + 50)

    def add_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.keras")])
        if model_path:
            model_name = os.path.basename(model_path)
            new_model_path = os.path.join(model_dir, model_name)
            os.rename(model_path, new_model_path)
            models[model_name] = load_model(new_model_path)
            self.model_combobox.configure(values=list(models.keys()))
            messagebox.showinfo("Успешно", f"Модель {model_name} добавлена.")

    def run_model(self):
        model_name = self.model_combobox.get()
        if model_name and model_name in models:
            model = models[model_name]
            self.run_model_window(model)
        else:
            messagebox.showwarning("Предупреждение", "Выберите валидную модель.")

    def rules_model_window(self):
        new_window = ctk.CTkToplevel(self.root)
        new_window.title("Инструкция")
        new_window.geometry("820x230")
        new_window.grab_set()

        label = ctk.CTkLabel(
            new_window,
            text=("Для того чтобы запустить модель вам нужно:\n"
                  "1. Загрузить модель, нажав 'Добавить модель', выбрав модель с расширением .keras.\n"
                  "2. В выпадающем списке выберите загруженную модель.\n"
                  "3. Нажмите 'Запустить модель'."),
            anchor="w",
            justify="left",
            font=("Arial Rounded MT Bold", 18)
        )
        label.pack(pady=30, padx=30, fill="x")

        close_button = ctk.CTkButton(new_window, text="Закрыть", command=new_window.destroy, font=self.font_style)
        close_button.pack(pady=0)

    def run_model_window(self, model):
        cap = None
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Камера с индексом {i} успешно открыта.")
                break

        if cap is None or not cap.isOpened():
            print("Ошибка: Не удалось открыть веб-камеру.")
            return

        last_prediction = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка: Не удалось считать кадр.")
                break

            preprocessed_frame = self.preprocess_image(frame)
            prediction = model.predict(preprocessed_frame)[0][0]

            label = "Ripe/Fresh" if prediction >= 0.5 else "Unripe/Rotten"
            color = (0, 255, 0) if prediction >= 0.5 else (0, 0, 255)

            cv2.putText(frame, f'Class: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            if last_prediction is not None and last_prediction != label:
                time.sleep(0.7)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join("frame", f'frame_change_{timestamp}.jpg')
                cv2.imwrite(save_path, frame)
                print(f"Сохранен фрейм при изменении предсказания: {label}")

            last_prediction = label
            cv2.imshow('Video with cam - Detect ripe and unripe', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def preprocess_image(image):
        img = cv2.resize(image, (64, 64))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img


if __name__ == "__main__":
    root = ctk.CTk()
    app = ModelApp(root)
    root.mainloop()
