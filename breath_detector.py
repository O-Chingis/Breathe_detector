import cv2
import numpy as np
import time


class BreathDetector:
    def __init__(self):
        self.prev_positions = {"chest": None, "stomach": None}
        self.movement_history = {"chest": [], "stomach": []}
        self.last_warning_time = 0

    def detect_breathing(self, frame):
        mask = self.detect_green_markers(frame)
        chest_pos, stomach_pos = self.get_marker_positions(mask)

        if chest_pos and stomach_pos:
            self.update_movement(chest_pos, stomach_pos)
            return self.analyze_breathing()
        else:
            return "Маркеры не найдены"

    def detect_green_markers(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return mask

    def get_marker_positions(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        positions = []

        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                positions.append((cx, cy))

        positions = sorted(positions, key=lambda x: x[1])

        if len(positions) >= 2:
            return positions[0], positions[1]
        return None, None

    def update_movement(self, chest_pos, stomach_pos):
        if self.prev_positions["chest"] and self.prev_positions["stomach"]:
            chest_movement = abs(chest_pos[1] - self.prev_positions["chest"][1])
            stomach_movement = abs(stomach_pos[1] - self.prev_positions["stomach"][1])

            self.movement_history["chest"].append(chest_movement)
            self.movement_history["stomach"].append(stomach_movement)

            if len(self.movement_history["chest"]) > 30:
                self.movement_history["chest"].pop(0)
                self.movement_history["stomach"].pop(0)

        self.prev_positions = {"chest": chest_pos, "stomach": stomach_pos}

    def analyze_breathing(self):
        avg_chest = np.mean(self.movement_history["chest"]) if self.movement_history["chest"] else 0
        avg_stomach = np.mean(self.movement_history["stomach"]) if self.movement_history["stomach"] else 0
        current_time = time.time()

        if avg_stomach > avg_chest:
            return "Диафрагмальное дыхание (норма)"
        elif avg_chest > avg_stomach:
            return "Грудное дыхание (поверхностное)"
        elif abs(avg_chest - avg_stomach) > 5:
            self.last_warning_time = current_time
            return "Асинхронное дыхание"
        return "Дыхание в норме."
