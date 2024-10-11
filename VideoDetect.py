import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os
# import torch

# Загрузка обученной модели
model = load_model(r'D:\Python\PycharmProjects\neiro\Mag\models_dataset3\simplified_cnn_epoch_08_model.keras')

# # Загрузка предобученной модели YOLOv5 для распознавания листьев и помидоров
# yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

save_dir = r'D:\Python\PycharmProjects\neiro\Mag\frame'

# Функция для предобработки изображения перед предсказанием
def preprocess_image(image):
    img = cv2.resize(image, (64, 64))  # Изменяем размер изображения
    img = img.astype('float32') / 255.0  # Нормализуем значения пикселей
    img = np.expand_dims(img, axis=0)  # Добавляем измерение пакета
    return img

# Попытка захвата видео с разных индексов камеры
cap = None
for i in range(4):  # Пробуем индексы от 0 до 3
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Камера с индексом {i} успешно открыта.")
        break

if cap is None or not cap.isOpened():
    print("Ошибка: Не удалось открыть веб-камеру.")
    exit()

last_prediction = None

# Чтение кадров с веб-камеры в реальном времени
while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка: Не удалось считать кадр.")
        break

    # Предобработка кадра и предсказание
    preprocessed_frame = preprocess_image(frame)
    prediction = model.predict(preprocessed_frame)[0][0]

    # Определение класса помидора
    label = "ripe" if prediction >= 0.5 else "unripe"
    color = (0, 255, 0) if prediction >= 0.5 else (0, 0, 255)  # Зеленый для спелого, красный для неспелого



    # Добавление метки на кадр
    cv2.putText(frame, f'Class: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


    # # Использование модели YOLOv5 для распознавания листьев и помидоров
    # results = yolo_model(frame)
    # detections = results.pandas().xyxy[0]
    #
    # for _, row in detections.iterrows():
    #     x1, y1, x2, y2, confidence, class_id, name = row
    #     if name == 'tomato' or name == 'leaf':
    #         # Определение цвета рамки в зависимости от типа объекта
    #         box_color = (0, 255, 255) if name == 'leaf' else (255, 0, 0)
    #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
    #         cv2.putText(frame, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
    #

    # Выделение помидоров (по цвету)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([0, 100, 100])  # Нижняя граница красного цвета
    # upper_red = np.array([10, 255, 255])  # Верхняя граница красного цвета
    # mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    # Диапазоны цветов для спелых и неспелых помидоров

    lower_red_ripe = np.array([0, 100, 100])  # Нижняя граница красного цвета (спелые)
    upper_red_ripe = np.array([10, 255, 255])  # Верхняя граница красного цвета (спелые)

    # lower_green_unripe = np.array([35, 50, 50])  # Нижняя граница зеленого цвета (неспелые)
    # upper_green_unripe = np.array([85, 255, 255])  # Верхняя граница зеленого цвета (неспелые)

    # Маски для выделения спелых и неспелых помидоров
    mask_ripe = cv2.inRange(hsv_frame, lower_red_ripe, upper_red_ripe)
    # mask_unripe = cv2.inRange(hsv_frame, lower_green_unripe, upper_green_unripe)

    # Нахождение контуров помидоров
    contours_ripe, _ = cv2.findContours(mask_ripe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_ripe:
        area = cv2.contourArea(contour)
        if area > 500:  # Фильтр по минимальной площади
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Желтая рамка вокруг помидора
    #
    # # Нахождение контуров помидоров
    # contours_unripe, _ = cv2.findContours(mask_unripe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours_unripe:
    #     area = cv2.contourArea(contour)
    #     if area > 1000:  # Фильтруем по площади
    #         x, y, w, h = cv2.boundingRect(contour)
    #         roi = frame[y:y + h, x:x + w]  # Извлекаем область интереса (ROI)
    #         preprocessed_roi = preprocess_image(roi)
    #         prediction = model.predict(preprocessed_roi)[0][0]
    #
    #         if prediction < 0.5:  # Если модель определила, что это неспелый помидор
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Красная рамка вокруг неспелого помидора

    # Сохранение фрейма при изменении предсказания
    if last_prediction is not None and last_prediction != label:
        time.sleep(0.7)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.putText(frame, f'Class: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        save_path=os.path.join(save_dir, f'frame_change_{timestamp}.jpg')
        cv2.imwrite(save_path, frame)
        print(f"Сохранен фрейм при изменении предсказания: {label}")

    last_prediction = label

    # Отображение кадра
    cv2.imshow('Video with cam - Detect ripe and unripe', frame)

    # Завершение программы при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()




# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
#
# # Загрузка обученной модели
# model = load_model(r'D:\Python\PycharmProjects\neiro\Mag\models_dataset3\simplified_cnn_epoch_08_model.keras')
#
# # Функция для предобработки изображения перед предсказанием
# def preprocess_image(image):
#     img = cv2.resize(image, (64, 64))  # Изменяем размер изображения до 64x64
#     img = img.astype('float32') / 255.0  # Нормализуем значения пикселей
#     img = np.expand_dims(img, axis=0)  # Добавляем измерение пакета
#     return img
#
# # Захват видео с веб-камеры
# cap = cv2.VideoCapture(0)  # Индекс 0 — это основная камера
#
# if not cap.isOpened():
#     print("Ошибка: Не удалось открыть веб-камеру.")
#     exit()
#
# # Чтение кадров с веб-камеры в реальном времени
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         print("Ошибка: Не удалось считать кадр.")
#         break
#
#     # Предобработка кадра и предсказание
#     preprocessed_frame = preprocess_image(frame)
#     prediction = model.predict(preprocessed_frame)[0][0]
#
#     # Определение класса помидора
#     label = "Спелый" if prediction >= 0.5 else "Неспелый"
#     color = (0, 255, 0) if prediction >= 0.5 else (0, 0, 255)  # Зеленый для спелого, красный для неспелого
#
#     # Добавление метки на кадр
#     cv2.putText(frame, f'Класс: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
#
#     # Отображение кадра
#     cv2.imshow('Видео с веб-камеры - Распознавание спелости помидора', frame)
#
#     # Завершение программы при нажатии клавиши 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Освобождение ресурсов
# cap.release()
# cv2.destroyAllWindows()
