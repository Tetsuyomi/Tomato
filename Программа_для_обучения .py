import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Пути к папкам с изображениями и метками
images_dir = r'D:\Python\PycharmProjects\neiro\Mag\dataset3\Images'
labels_dir = r'D:\Python\PycharmProjects\neiro\Mag\dataset3\labels'

# Функция для чтения и предобработки изображений и меток
def load_data(images_dir, labels_dir):
    images = []
    labels = []

    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # Чтение и предобработка изображения
        img = cv2.imread(image_path)
        img = cv2.resize(img, (64, 64))
        img = img.astype('float32') / 255.0

        # Чтение меток из соответствующего txt файла
        with open(label_path, 'r') as file:
            label_data = file.readlines()
            # Проверяем, есть ли метка спелого помидора (предположим, что метка 1 — спелый помидор)
            ripe_label = 0
            for line in label_data:
                class_id = int(line.split()[0])
                if class_id == 1:  # Предполагается, что класс 1 — это спелый помидор
                    ripe_label = 1
                    break

        images.append(img)
        labels.append(ripe_label)

    return np.array(images), np.array(labels)

# Загрузка данных
X_train, y_train = load_data(images_dir, labels_dir)

# Создание простой модели CNN
model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Бинарная классификация (спелый/неспелый)
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обратный вызов для сохранения модели после каждой эпохи
checkpoint = ModelCheckpoint(
    'simplified_cnn_epoch_{epoch:02d}_model.keras',  # Сохраняем модель после каждой эпохи с указанием номера эпохи
    save_best_only=False,
    save_weights_only=False,
    mode='auto'
)

# Ранняя остановка для предотвращения переобучения
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Обучение модели
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint]  # Добавляем сохранение модели и раннюю остановку
)

# Визуализация истории обучения
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
