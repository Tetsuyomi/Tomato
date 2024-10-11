import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Путь к папке, содержащей сохраненные модели
models_dir = r'D:\Python\PycharmProjects\neiro\Mag\models_dataset3'
image_path = r'D:\Python\PycharmProjects\neiro\Mag\photo_2024-10-09_15-32-33.jpg'  # Замените на путь к изображению помидора, который хотите использовать

# Функция для предобработки изображения перед тестированием
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  # Изменение размера изображения для модели
    img = img.astype('float32') / 255.0  # Нормализация
    img = np.expand_dims(img, axis=0)  # Добавление дополнительного измерения для пакета изображений
    return img

# Загрузка всех моделей из директории
models = []
for i in range(1, 21):  # Предполагается, что модели названы в формате simplified_cnn_epoch_{номер}_model.keras
    model_path = os.path.join(models_dir, f'simplified_cnn_epoch_{i:02d}_model.keras')
    model = load_model(model_path)
    models.append((f'Model Epoch {i}', model))

# Предсказание всех моделей на одном изображении
def predict_with_all_models(models, image_path):
    img = preprocess_image(image_path)
    predictions = []

    for model_name, model in models:
        prediction = model.predict(img)[0][0]
        predicted_label = 'Спелый' if prediction >= 0.5 else 'Неспелый'
        predictions.append((model_name, predicted_label))

    return predictions

# Отображение всех предсказаний в одном окне
def display_predictions(image_path, predictions):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))

    for i, (model_name, predicted_label) in enumerate(predictions):
        plt.subplot(4, 5, i + 1)  # Сетка 4x5 для отображения 20 изображений
        plt.imshow(img_rgb)
        plt.title(f'{model_name}\nПредсказание: {predicted_label}', fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Использование функции для получения и отображения предсказаний всех моделей
predictions = predict_with_all_models(models, image_path)
display_predictions(image_path, predictions)
