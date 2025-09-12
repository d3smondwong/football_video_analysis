import supervision as sv
import logging
import numpy as np
import pickle
import os
import cv2

from ultralytics import YOLO
from src.utils.bbox_utils import get_center_of_bbox, get_bbox_width, measure_distance, measure_xy_distance, get_foot_position

# Set up logging
logger = logging.getLogger(__name__)

class Tracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def _detect_frames(self, frames):
        logger.info(f'Start detect_frames function')

        # Process frames in batches to ensure it does not run out of memory
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            detections_batch = self.model.predict(batch, conf=0.1, verbose=False)

            detections += detections_batch

        return detections

    def get_object_tracking(self, frames, read_from_stub=False, stub_path=None):

        logger.info(f'Start get_object_tracking function')

        # Check if stub file exists, if yes, read from it. To prevent re-computation of tracking
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracking_list = pickle.load(f)
                logger.info(f'Stub file exists at {stub_path}. Loading object detection and tracking data from stub file.')
            return tracking_list

        logger.info(f'Stub file does not exists. Performing object detection and tracking.')

        # Get detections for each frame in the video as a list
        detections = self._detect_frames(frames)

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
                        tracking_list["ball"][frame_num][track_id] = {"bbox":bbox}

                # As there is only one ball per frame. If multiple balls are detected, take the first one. Update: Will use ball tracking to track the ball better
                # for frame_detection in detections_supervision:
                #     bbox = frame_detection[0].tolist()
                #     class_id = frame_detection[3]

                #     if class_id == class_names_inv['ball']:
                #         tracking_list["ball"][frame_num][1] = {"bbox":bbox}

            # After processing all frames, save the tracking_list to stub file
            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracking_list, f)

            logger.info(f'Saving stub file to {stub_path}')

            return tracking_list

        else:
            logger.warning("No detections found.")

    def _draw_ellipse(self, frame, bbox, color,track_id):
        x1, y1, x2, y2 = map(int, bbox)
        x_center, y_center = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        axes_length = (int(width), int(0.35*width))
        angle = 0
        # Ellipse is drawn from start_angle to end_angle. Draw 45 - 235 degrees for players to stand out
        start_angle = 45
        end_angle = 235
        thickness = 2
        line_type = cv2.LINE_4

        # Draw the ellipse on the frame
        cv2.ellipse(frame, (x_center, y2), axes_length, angle, start_angle, end_angle, color, thickness, line_type)

        # Put the track ID text near the ellipse
        # cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def draw_annotations(self, video_frames, tracking_list):
        logger.info(f'Start draw_annotations function')

        output_video_frames = []

        # tracks is loaded from the stub file
        num_stub_frames = len(tracking_list["players"])
        num_video_frames = len(video_frames)

        logger.info(f"Number of frames in stub file: {num_stub_frames}")
        logger.info(f"Number of frames in video_frames: {num_video_frames}")

        for frames_num, frame in enumerate(video_frames):

            frame = frame.copy()

            player_dict = tracking_list["players"][frames_num]
            ball_dict = tracking_list["ball"][frames_num]
            referee_dict = tracking_list["referees"][frames_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self._draw_ellipse(frame, player["bbox"],color, track_id)

            output_video_frames.append(frame)

        return output_video_frames
