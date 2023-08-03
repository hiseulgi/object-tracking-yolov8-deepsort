from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker():
    def __init__(self):
        self.tracker = DeepSort()

    def track_objects(self, results, frame):
        tracks = self.tracker.update_tracks(results, frame=frame)
        return tracks
