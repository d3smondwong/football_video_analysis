from ultralytics import YOLO
import supervision as sv
import logging

# Set up logging
logger = logging.getLogger(__name__)

class Tracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):

        # Process frames in batches to ensure it does not run out of memory
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            detections_batch = self.model.predict(batch, conf=0.1, verbose=False)

            detections += detections_batch
            break

        return detections

    def get_object_tracks(self, frames):

        # Get detections for each frame in the video as a list
        detections = self.detect_frames(frames)

        # The model is not predicting Goalkeeper class too well. To change the Goalkeeper class ID to 0 (person class ID) in the detections

        if detections is not None:
            for frame_num, detection in enumerate(detections):

                class_names = detection.names

                # inverse the class names dictionary to get class name to class ID mapping. E.g. {0: 'person', 1: 'bicycle'} to {'person': 0, 'bicycle': 1}
                class_names_inv = {v: k for k, v in class_names.items()}

                # Convert to supervision Detections object
                detections_supervision = sv.Detections.from_ultralytics(detection)

        else:
            logger.warning("No detections found.")