# detector.py
from ultralytics import YOLO
import cv2

class PPEDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame, conf_threshold=0.5):
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            detections.append({
                "class": label,
                "bbox": (x1, y1, x2, y2),
                "confidence": conf
            })
        return detections
