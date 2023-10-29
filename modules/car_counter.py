import numpy as np
import cv2


class CarCounter:
    def __init__(self, regions):
        self.regions = regions
        self.counter_ids = [set() for _ in range(len(regions))]

    def count_cars(self, tracks):
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id

            ltrb = track.to_ltrb()
            center_x = int((ltrb[0] + ltrb[2]) / 2)
            center_y = int((ltrb[1] + ltrb[3]) / 2)

            for i, region in enumerate(self.regions):
                if (cv2.pointPolygonTest(
                        np.array(region), (center_x, center_y), False)) > 0:
                    self.counter_ids[i].add(track_id)

    def get_region(self):
        return self.regions

    def get_count(self):
        return [len(ids) for ids in self.counter_ids]
