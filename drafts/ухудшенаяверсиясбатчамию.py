from ultralytics import YOLO
import cv2
import os
from threading import Thread

# Создание папки для сохранения кадров
output_folder = "saved_frames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Загрузка модели YOLOv8s
model = YOLO("yolov8s.pt")

# Захват видеопотока
cap = cv2.VideoCapture("video_2025-01-19_21-33-09.mp4")

# Определение области интереса (ROI)
roi = [(130, 130), (600, 600)]  # Пример координат области

# Счетчик кадров
frame_count = 0

# Размер батча (количество кадров, обрабатываемых одновременно)
batch_size = 8

# Список для хранения кадров и их метаданных
frames = []
metadata = []

# Функция для обработки батча
def process_batch(frames, metadata):
    global frame_count
    # Детекция объектов в батче
    results = model(frames)

    # Обработка результатов детекции для каждого кадра в батче
    for i, result in enumerate(results):
        frame = frames[i]
        height, width = metadata[i]

        # Обработка результатов детекции
        for box in result.boxes:
            if result.names[int(box.cls)] == "car":  # Проверка класса "car"
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Проверка, что bounding box пересекается с ROI или полностью внутри
                if (x1 < roi[1][0] and x2 > roi[0][0] and y1 < roi[1][1] and y2 > roi[0][1]) or \
                   (x1 >= roi[0][0] and y1 >= roi[0][1] and x2 <= roi[1][0] and y2 <= roi[1][1]):
                    # Увеличение счетчика кадров
                    frame_count += 1

                    # Сохранение всего кадра (не обрезанного)
                    output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(output_path, frame)
                    print(f"Сохранен кадр: {output_path}")

        # Отображение кадра с выделенными объектами и ROI
        cv2.rectangle(frame, roi[0], roi[1], (0, 255, 0), 2)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Уменьшение разрешения кадра на 20% (для повышения производительности)
    height, width = frame.shape[:2]
    new_width = int(width * 0.8)
    new_height = int(height * 0.8)
    frame = cv2.resize(frame, (new_width, new_height))

    # Добавление кадра в батч
    frames.append(frame)
    metadata.append((new_height, new_width))

    # Если батч заполнен, обрабатываем его в отдельном потоке
    if len(frames) == batch_size:
        # Создание потока для обработки батча
        thread = Thread(target=process_batch, args=(frames.copy(), metadata.copy()))
        thread.start()

        # Очистка батча
        frames = []
        metadata = []

cap.release()
cv2.destroyAllWindows()