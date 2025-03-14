import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt6.QtCore import QTimer
from breath_detector import BreathDetector


class BreathTrainerApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Интеллектуальный тренажёр дыхания")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.video_label = QLabel(self)
        layout.addWidget(self.video_label)

        self.status_label = QLabel("Статус дыхания: Ожидание...")
        layout.addWidget(self.status_label)

        self.indicator_label = QLabel(self)
        layout.addWidget(self.indicator_label)

        video_path = "resources/test_video.mp4"
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            print(f"Видео {video_path} не найдено, переключаюсь на веб-камеру...")
            self.cap = cv2.VideoCapture(0)

        self.detector = BreathDetector()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Видео закончилось или произошла ошибка")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        status = self.detector.detect_breathing(frame)
        self.status_label.setText(f"Статус дыхания: {status}")


        if "Грудное дыхание" in status or "Асинхронное дыхание" in status:
            color = QColor(255, 0, 0)
        else:
            color = QColor(0, 255, 0)

        self.update_indicator(color)

        height, width, _ = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def update_indicator(self, color):
        size = 50
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(255, 255, 255, 0))

        painter = QPainter(pixmap)
        painter.setBrush(color)
        painter.setPen(color)
        painter.drawEllipse(0, 0, size, size)
        painter.end()

        self.indicator_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BreathTrainerApp()
    window.show()
    sys.exit(app.exec())
