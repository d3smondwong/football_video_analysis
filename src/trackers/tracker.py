import supervision as sv
import logging
import numpy as np
import pickle
import os
import cv2
import pandas as pd

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

                # Filter player/referee detections with confidence > 0.8, keep ball detections as-is
                player_id = class_names_inv['player']
                referee_id = class_names_inv['referee']
                ball_id = class_names_inv['ball']
                if detections_supervision.class_id is not None and detections_supervision.confidence is not None:
                    indices = [
                        i for i, (cid, conf) in enumerate(zip(detections_supervision.class_id, detections_supervision.confidence))
                        if ((cid == player_id or cid == referee_id) and conf > 0.8) or (cid == ball_id)
                    ]
                else:
                    indices = []

                detections_supervision = sv.Detections(
                    xyxy=detections_supervision.xyxy[indices] if detections_supervision.xyxy is not None else np.array([]),
                    confidence=detections_supervision.confidence[indices] if detections_supervision.confidence is not None else np.array([]),
                    class_id=detections_supervision.class_id[indices] if detections_supervision.class_id is not None else np.array([]),
                    tracker_id=None if detections_supervision.tracker_id is None else detections_supervision.tracker_id[indices],
                    data={k: np.array(v)[np.array(indices)] for k, v in detections_supervision.data.items()} if detections_supervision.data else {}
                )

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

                    # Commenting out the ball tracking for now as the ball is not tracked very well. Probably due to training data size. Will use the first detected ball in each frame instead
                    # if class_id == class_names_inv['ball']:
                    #     tracking_list["ball"][frame_num][track_id] = {"bbox":bbox}

                # As there is only one ball per frame. If multiple balls are detected, take the one with the highest confidence.
                # Find all ball detections in the current frame
                ball_indices = [i for i, cid in enumerate(detections_supervision.class_id) if cid == class_names_inv['ball']] if detections_supervision.class_id is not None else []

                if ball_indices and detections_supervision.confidence is not None:
                    # Get the index of the ball detection with the highest confidence
                    ball_confidences = [detections_supervision.confidence[i] for i in ball_indices]
                    max_conf_idx = ball_indices[np.argmax(ball_confidences)]
                    bbox = detections_supervision.xyxy[max_conf_idx].tolist()
                    tracking_list["ball"][frame_num][1] = {"bbox": bbox}

                elif ball_indices:
                    # If confidence is None, just take the first ball index
                    bbox = detections_supervision.xyxy[ball_indices[0]].tolist()
                    tracking_list["ball"][frame_num][1] = {"bbox": bbox}

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

        ###
        # Drawing the ellipse to represent the player or referee
        ###

        # Define ellipse parameters
        axes_length = (int(width), int(0.35*width))
        angle = 0
        # Ellipse is drawn from start_angle to end_angle. Draw -45 - 235 degrees for players to stand out
        start_angle = -45
        end_angle = 235
        thickness = 2
        line_type = cv2.LINE_4

        # Draw the ellipse on the frame
        cv2.ellipse(frame, (x_center, y2), axes_length, angle, start_angle, end_angle, color, thickness, line_type)

        ###
        # Drawing the rectangle place the player number
        ###

        # Define rectangle parameters
        rectangle_width = 40
        rectangle_height = 20

        rectangle_x1 = x_center - rectangle_width // 2
        rectangle_y1 = (y2- rectangle_height//2) +15
        rectangle_x2 = x_center + rectangle_width//2
        rectangle_y2 = (y2+ rectangle_height//2) +15

        # Define text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color_text = (0, 0, 0)  # Black color for text
        thickness = 2


        if track_id is not None:
            # Draw the rectangle on the frame
            cv2.rectangle(frame, (int(rectangle_x1), int(rectangle_y1)), (int(rectangle_x2), int(rectangle_y2)), color, cv2.FILLED)

            # Calculate position to center the text in the rectangle
            x1_text = rectangle_x1 + 12
            if track_id >= 99:
                x1_text -= 10

            # Put the track ID text on the rectangle
            cv2.putText(frame, str(track_id), (int(x1_text), int(rectangle_y1 + 15)), font, font_scale, color_text, thickness)

        return frame

    def _draw_triangle(self, frame, bbox, color):
        x1, y1, x2, y2 = map(int, bbox)
        x_center, y_center = get_center_of_bbox(bbox)

        # Define triangle parameters
        triangle_points = np.array([
                [x_center,y1],
                [x_center-10,y1-20],
                [x_center+10,y1-20],
            ])

        # Draw the triangle on the frame
        cv2.drawContours(frame, [triangle_points], 0, color, -1)  # -1 fills the triangle
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)  # 2 is the thickness of the outline

        return frame

    def interpolate_ball_positions(self, ball_positions):

        # Not in use currently. Will use the detected ball position with highest confidence in each frame instead.

        # Extract the bounding box coordinates for the ball
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]

        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing ball positions
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit_direction='both')
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def draw_annotations(self, video_frames, tracking_list):
        logger.info(f'Start draw_annotations function')

        output_video_frames = []

        # Check stub file data
        num_stub_frames = len(tracking_list["players"])
        num_video_frames = len(video_frames)

        if num_stub_frames != num_video_frames:
            logger.info(f"Number of frames in stub file: {num_stub_frames}")
            logger.info(f"Number of frames in input video: {num_video_frames}")

            logger.warning("Number of frames in stub file does not match number of frames in input video. There might be an error in the stub file or stub file video is different from input video.")
            return None

        for frames_num, frame in enumerate(video_frames):

            frame = frame.copy()

            player_dict = tracking_list["players"][frames_num]
            ball_dict = tracking_list["ball"][frames_num]
            referee_dict = tracking_list["referees"][frames_num]

            # Draw ellipse to annotate players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self._draw_ellipse(frame, player["bbox"],color, track_id)

            # Draw ellipse to annotate referees
            for track_id, referee in referee_dict.items():
                frame = self._draw_ellipse(frame, referee["bbox"],(0,255,255), track_id)

            # Draw triangle to annotate ball
            for track_id, ball in ball_dict.items():
                frame = self._draw_triangle(frame, ball["bbox"],(0,255,0))

            output_video_frames.append(frame)

        return output_video_frames
