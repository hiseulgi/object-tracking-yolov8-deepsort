from modules.camera import Camera
from modules.car_counter import CarCounter
from modules.object_detector import ObjectDetector
from modules.object_tracker import ObjectTracker
from modules.utils import random_pastel_color

import cvzone
import cv2

# const
CLASS_NAME = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
ALLOWED_CLASS = [
    "car",
    # "bus", "truck"
]

# counting region of interest
region_1 = [(20, 600), (490, 600), (490, 700), (20, 700)]
region_2 = [(530, 600), (725, 600), (725, 700), (530, 700)]
region_3 = [(830, 490), (1160, 490), (1160, 540), (830, 540)]
region_colors = [random_pastel_color() for _ in range(3)]

# init new window
WIN_NAME = "Camera Original"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

# init mask
# TODO: belajar masking, agar komputasi lebih ringan dan akurat
# mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
# mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]


class CarCounterApp():
    def __init__(self, video_source):
        self.camera = Camera(video_source)
        self.object_detector = ObjectDetector(confidence_threshold=0.64)
        self.object_tracker = ObjectTracker()
        self.car_counter = CarCounter([region_1, region_2, region_3])
        self.fps_reader = cvzone.FPS()

    def run(self):
        while cv2.waitKey(1) != ord("q"):
            frame = self.camera.get_frame()
            # masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            frame = cv2.resize(frame, (1280, 720))

            detections = self.object_detector.detect_objects(
                frame, CLASS_NAME, ALLOWED_CLASS)
            tracks = self.object_tracker.track_objects(
                detections, frame)
            self.car_counter.count_cars(tracks)

            # display object detection and tracking results
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                class_id = track.det_class

                ltrb = track.to_ltrb()

                x1, y1, x2, y2 = int(ltrb[0]), int(
                    ltrb[1]), int(ltrb[2]), int(ltrb[3])

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                color = (64, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)
                cv2.rectangle(frame, (x1, y1 - 20),
                              (x1 + 60, y1), color, -1)
                label = "{} {}".format(CLASS_NAME[class_id], str(track_id))
                cv2.putText(frame, label, (x1 + 5, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # display object count results
            counts = self.car_counter.get_count()
            regions = self.car_counter.get_region()

            for i, count in enumerate(counts):
                x, y = regions[i][0]
                cv2.rectangle(frame, regions[i][0],
                              regions[i][1], region_colors[i], 2)
                cv2.rectangle(frame, regions[i][0],
                              (x+90, y-20), region_colors[i], -1)
                cv2.putText(frame, "Car: {}".format(count),
                            regions[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            fps, frame = self.fps_reader.update(frame, pos=(
                20, 40), color=(40, 80, 255), scale=2, thickness=4)
            cv2.imshow(WIN_NAME, frame)
        self.camera.release_camera()
        return


if __name__ == "__main__":
    app = CarCounterApp("video/road_traffic.mp4")
    app.run()
    cv2.destroyAllWindows()
