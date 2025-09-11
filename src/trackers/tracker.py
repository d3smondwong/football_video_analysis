import supervision as sv
import logging
import numpy as np
import pickle
import os

from ultralytics import YOLO

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

        return detections

    def get_object_tracking(self, frames, read_from_stub=False, stub_path=None):

        logger.info(f'Start object tracking function')

        # Check if stub file exists, if yes, read from it. To prevent re-computation of tracking
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracking_list = pickle.load(f)
                logger.info(f'Stub file exists at {stub_path}. Loading object detection and tracking data from stub file.')
            return tracking_list

        logger.info(f'Stub file does not exists. Performing object detection and tracking.')

        # Get detections for each frame in the video as a list
        detections = self.detect_frames(frames)
        logger.info(detections)

        tracking_list = {"players": [],
                        "referees": [],
                        "ball": []}

        # The model is not predicting Goalkeeper class too well. Probably due to class imbalance. To change the Goalkeeper class ID to 0 (person class ID) in the detections
        if detections is not None:
            for frame_num, detection in enumerate(detections):

                class_names = detection.names

                # inverse the class names dictionary to get class name to class ID mapping. E.g. {0: 'ball', 1: 'goalkeeper'} to {'ball': 0, 'goalkeeper': 1}
                class_names_inv = {v: k for k, v in class_names.items()}

                # Convert to supervision Detections object
                detections_supervision = sv.Detections.from_ultralytics(detection)

                # {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'} change all class_id == 1 to 2 and Goalkeeper to Player
                if detections_supervision.class_id is not None:
                    for object_ind, class_id in enumerate(detections_supervision.class_id):
                        if class_names[class_id] == 'goalkeeper':
                            detections_supervision.class_id[object_ind] = class_names_inv['player']

                # Track the objects across frames
                detection_with_tracking = self.tracker.update_with_detections(detections=detections_supervision)

                # Initialize empty dictionaries for the current frame
                tracking_list["players"].append({})
                tracking_list["referees"].append({})
                tracking_list["ball"].append({})

                # detection_with_tracking is a list of tuples. Each tuple contains in the following order (xyxy, mask, confidence, class_id, track_id, data)
                for frame_detection in detection_with_tracking:
                    bbox = frame_detection[0].tolist()
                    class_id = frame_detection[3]
                    track_id = frame_detection[4]

                    if class_id == class_names_inv['player']:
                        tracking_list["players"][frame_num][track_id] = {"bbox": bbox}

                    if class_id == class_names_inv['referee']:
                        tracking_list["referees"][frame_num][track_id] = {"bbox": bbox}

                    if class_id == class_names_inv['ball']:
                        tracking_list["ball"][frame_num][track_id] = {"bbox": bbox}

                if stub_path is not None:
                   with open(stub_path, 'wb') as f:
                       pickle.dump(tracking_list, f)

                logger.info(f'Saving stub file to {stub_path}')

                return tracking_list

        else:
            logger.warning("No detections found.")

    def draw_annotations(self, video_frames, tracking_list):

        output_video_frames = []

        for frames_num, frame in enumerate(video_frames):

            frame = frame.copy()

