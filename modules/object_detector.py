import math
from ultralytics import YOLO


class ObjectDetector():
    def __init__(self, confidence_threshold=0.5):
        self.model = YOLO("model/yolov8n.pt")
        self.model.to('cuda')
        self.confidence_threshold = confidence_threshold

    def detect_objects(self, frame, class_name, allowed_class=[]):
        predictions = self.model(frame)[0]
        results = []

        for box in predictions.boxes:
            confidence = math.ceil((box.conf[0]*100))/100
            if confidence > self.confidence_threshold:
                x, y, w, h = box.xywh[0]
                x, y, w, h = int(x), int(y), int(w), int(h)
                class_id = int(box.cls[0])

                if class_name[class_id] in allowed_class:
                    results.append(
                        [[x-(w/2), y-(h/2), w, h], confidence, class_id]
                    )
        return results
