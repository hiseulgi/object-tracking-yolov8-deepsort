import cv2


class Camera():
    def __init__(self, source=0):
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
