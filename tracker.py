from collections import defaultdict

import cv2
import numpy as np


class Tracker:
    def __init__(self):
        self.colors = defaultdict(lambda: tuple(int(v) for v in np.random.randint(0, 255, 3)))
        self.track_history = defaultdict(lambda: [])
        self.valid_objects = {}
        self.labels = {
            'person': (0, 0, 255),
            'bicycle': (255, 0, 0),
            'car': (0, 255, 0),
            'motorbike': (255, 255, 0),
            'truck': (0, 255, 255),
            'bus': (255, 0, 255),
        }
        self.translate = {
            'person': 'Pessoas',
            'bicycle': 'Bicicletas',
            'car': 'Carros',
            'motorbike': 'Motos',
            'truck': 'Caminhoes',
            'bus': 'Onibus',
        }

    def update(self, objects, roi):
        roi_x, roi_y, roi_width, roi_height = roi
        valid_objects = []

        for obj in self.valid_objects.values():
            obj['lives'] -= 1

        for obj in objects:
            self.track_history[obj.track_id].append((obj.x + obj.width / 2, obj.y + obj.height / 2))
            obj.track = self.track_history[obj.track_id]

            if roi_x <= obj.x <= roi_x + roi_width and roi_y <= obj.y <= roi_y + roi_height:
                self.valid_objects[obj.track_id] = {
                    'label': obj.label,
                    'lives': 5,
                }
            obj.valid = obj.track_id in self.valid_objects
            if obj.valid:
                valid_objects.append(obj)
        return valid_objects

    def draw(self, frame, objects):
        for obj in objects:
            if len(obj.track) < 2 or not obj.valid:
                continue
            points = np.hstack(obj.track).astype(np.int32).reshape((-1, 1, 2))
            color = self.colors[obj.track_id]
            # cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)
        return frame

    def count_valid(self, label):
        return len([obj for obj in self.valid_objects.values() if obj['label'] == label])

    def count_valid_alive(self, label):
        return len([obj for obj in self.valid_objects.values() if obj['label'] == label and obj['lives'] > 0])
