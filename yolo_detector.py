# https://medium.com/@tejasdalvi927/object-detection-with-yolo-and-opencv-a-practical-guide-cf7773481d11
from collections import namedtuple
from pathlib import Path

import cv2
from ultralytics import YOLO

from obj_detection import DetectedObject


MODELS = Path(__file__).parent / "models"

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class ObjectDetector:
    def __init__(self) -> None:
        self.model = YOLO(MODELS / "yolov10n.pt", task="track", verbose=False)

    def detect(self, frame):
        results = self.model.track(frame, stream=True, persist=True, verbose=False)
        cars = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = box.cls[0]
                label = CLASS_NAMES[int(cls)]
                if label in ("person", "bicycle", "car", "motorbike", "truck", "bus"):
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    cars.append(DetectedObject(x1, y1, x2 - x1, y2 - y1, box.id, label))
        return cars
