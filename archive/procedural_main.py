# TODO: buat agar menjadi OOP; Camera, ObjectDetector, ObjectTracker, dan CarCounter.

import math
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Const
CLASS_NAME = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
ALLOWED_CLASS = [
    "car", "bus", "truck"
]
CONFIDENCE_THRESHOLD = 0.5

# Counting ROI
# region_1 = 100,450; 490,450
# region_2 = 530,480; 725,480
# region_3 = 830,490; 1160, 490
region_1 = [(100, 450), (490, 450), (490, 500), (100, 500)]
region_2 = [(530, 480), (725, 480), (725, 530), (530, 530)]
region_3 = [(830, 490), (1160, 490), (1160, 540), (830, 540)]

counter_ids_1 = set()
counter_ids_2 = set()
counter_ids_3 = set()

# Model
model = YOLO("yolov8n.pt")
model.to('cuda')
tracker = DeepSort(max_age=30)
fpsReader = cvzone.FPS()

# mask
# mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
# mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

# window
WIN_NAME = "Camera Original"
# WIN_NAME_2 = "Camera Masked"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
# cv2.namedWindow(WIN_NAME_2, cv2.WINDOW_NORMAL)


class Camera():
    def __init__(self, source=0) -> None:
        self.cap = cv2.VideoCapture(source)
        print("Open camera source")

    def get_frame(self):
        has_frame, frame = self.cap.read()
        if has_frame:
            pass
        return frame

    def mirror_frame(self, frame):
        frame = cv2.flip(frame, 1)
        return frame

    def release_camera(self):
        self.cap.release()


def main():
    camera = Camera("video/road_traffic.mp4")
    count = 0
    # camera = Camera(0)
    while cv2.waitKey(1) != ord("q"):
        frame = camera.get_frame()
        # masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        # count += 1
        # if count % 4 == 0:
        #     continue

        # * model predict
        detections = model(frame)[0]
        # result akan berisi:
        # [left,top,w,h], confidence, detection_class
        results = []

        # * Object Detection
        for box in detections.boxes:
            confidence = math.ceil((box.conf[0]*100))/100
            if confidence > CONFIDENCE_THRESHOLD:
                # * mengambil data dengan format xywh
                # xy adalah centroid;
                # wh adalah panjang dan tingginya;
                x, y, w, h = box.xywh[0]
                x, y, w, h = int(x), int(y), int(w), int(h)
                class_id = int(box.cls[0])

                # cek hanya kelas yang diperbolehkan saja
                if CLASS_NAME[class_id] in ALLOWED_CLASS:
                    # data harus berformat [[left,top,w,h], confidence, detection_class]
                    results.append(
                        [[x-(w/2), y-(h/2), w, h], confidence, class_id]
                    )

                # * mengambil data dengan format xyxy
                # # x1y1 adalah left top
                # # x2y2 adalah right bottom
                # x1, y1, x2, y2 = box.xyxy[0]
                # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # class_id = int(box.cls[0])

                # # data harus  berformat [[left,top,w,h], confidence, detection_class]
                # results.append(
                #     [[x1, y1, x2-x1, y2-y1], confidence, class_id]
                # )

        # * Object Tracking
        tracks = tracker.update_tracks(results, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            class_id = track.det_class

            # mengubah bounding box xywh menjadi xyxy (left,top,right,bottom)
            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = int(ltrb[0]), int(
                ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # mengambil centroid dari bounding box object
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # * Vehicle Counter
            # deteksi apakah center_x berada diantara sumbu x garis hitung
            inside_roi_1 = cv2.pointPolygonTest(
                np.array(region_1), (center_x, center_y), False)
            inside_roi_2 = cv2.pointPolygonTest(
                np.array(region_2), (center_x, center_y), False)
            inside_roi_3 = cv2.pointPolygonTest(
                np.array(region_3), (center_x, center_y), False)

            if inside_roi_1 > 0:
                counter_ids_1.add(track_id)

            if inside_roi_2 > 0:
                counter_ids_2.add(track_id)

            if inside_roi_3 > 0:
                counter_ids_3.add(track_id)

            # display result
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y),
                       4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x1, y1 - 20),
                          (x1 + 20, y1), (0, 255, 0), -1)
            label = "{} {}".format(
                CLASS_NAME[class_id], str(track_id))
            cv2.putText(frame, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # * show frame
        # :region count 1
        x, y = region_1[0]
        cv2.rectangle(frame, region_1[0], region_1[1], (255, 255, 0), 2)
        cv2.rectangle(frame, region_1[0], (x+90, y-20), (255, 255, 0), -1)
        cv2.putText(frame, "Car: {}".format(len(counter_ids_1)),
                    region_1[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        # :region count 2
        x, y = region_2[0]
        cv2.rectangle(frame, region_2[0], region_2[1], (0, 255, 255), 2)
        cv2.rectangle(frame, region_2[0], (x+90, y-20), (0, 255, 255), -1)
        cv2.putText(frame, "Car: {}".format(len(counter_ids_2)),
                    region_2[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        # :region count 3
        x, y = region_3[0]
        cv2.rectangle(frame, region_3[0], region_3[1], (255, 0, 255), 2)
        cv2.rectangle(frame, region_3[0], (x+90, y-20), (255, 0, 255), -1)
        cv2.putText(frame, "Car: {}".format(len(counter_ids_3)),
                    region_3[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # show fps
        fps, frame = fpsReader.update(frame, pos=(
            50, 80), color=(0, 255, 0), scale=5, thickness=5)

        # * show final result
        cv2.imshow(WIN_NAME, frame)
        # cv2.imshow(WIN_NAME_2, masked_frame)
    camera.release_camera()
    return


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
