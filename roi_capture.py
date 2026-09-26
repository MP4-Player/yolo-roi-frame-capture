from ultralytics import YOLO
import cv2
import os

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Пропуск каждого второго кадра
    if frame_count % 2 != 0:
        frame_count += 1
        continue

    # Уменьшение разрешения кадра на 20%
    height, width = frame.shape[:2]
    new_width = int(width * 0.8)
    new_height = int(height * 0.8)
    frame = cv2.resize(frame, (new_width, new_height))

    # Детекция объектов
    results = model(frame)

    # Обработка результатов детекции
    for result in results:
        for box in result.boxes:
            if result.names[int(box.cls)] == "car":  # Проверка класса "car"
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Проверка, что bounding box пересекается с ROI или полностью внутри
                if (x1 < roi[1][0] and x2 > roi[0][0] and y1 < roi[1][1] and y2 > roi[0][1]) or \
                   (x1 >= roi[0][0] and y1 >= roi[0][1] and x2 <= roi[1][0] and y2 <= roi[1][1]):
                    # Увеличение счетчика кадров
                    frame_count += 1

                    # Обрезка кадра по ROI
                    roi_frame = frame[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

                    # Сохранение обрезанного кадра
                    output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(output_path, roi_frame)
                    print(f"Сохранен кадр: {output_path}")

        # Отображение кадра с выделенными объектами и ROI
        cv2.rectangle(frame, roi[0], roi[1], (0, 255, 0), 2)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()