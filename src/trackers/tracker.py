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

    def _detect_frames(self, frames: list) -> list:
        """
        Processes a list of video frames in batches and performs object detection on each batch using the model.
        Args:
            frames (list): A list of video frames to be processed.
        Returns:
            list: A list of detections for all frames.
        """
        logger.info(f'Start detect_frames function')

        # Process frames in batches to ensure it does not run out of memory
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            detections_batch = self.model.predict(batch, conf=0.1, verbose=False)

            detections += detections_batch

        return detections

    def get_object_tracking(
        self,
        frames: list,
        read_from_stub: bool = False,
        stub_path: str = None
    ) -> dict:
        """
        Performs object detection and tracking on a sequence of video frames, specifically for football video analysis.
        Optionally reads tracking results from a stub file to avoid redundant computation.
        Args:
            frames (list): List of video frames to process for object detection and tracking.
            read_from_stub (bool, optional): If True, attempts to read tracking results from a stub file. Defaults to False.
            stub_path (str, optional): Path to the stub file for reading/writing tracking results. Defaults to None.
        Returns:
            dict: A dictionary containing tracking information for players, referees, and the ball across all frames.
                Structure:
                    {
                        "players": [ {track_id: {"bbox": [...]}, ... }, ... ],
                        "referees": [ {track_id: {"bbox": [...]}, ... }, ... ],
                        "ball": [ {1: {"bbox": [...]}, ... }, ... ]
                    }
                Each list entry corresponds to a frame, and each dictionary maps track IDs to bounding box coordinates.
        Notes:
            - Goalkeeper detections are reclassified as players due to class imbalance.
            - Player and referee detections are filtered by confidence (>0.8).
            - Ball tracking uses the highest-confidence detection per frame.
            - Results can be cached to a stub file for efficiency.
        """

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

    def _draw_ellipse(
        self,
        frame: 'np.ndarray',
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
        track_id: int | None
    ) -> 'np.ndarray':
        """
        Draws an ellipse to represent a player or referee on the given frame, along with a rectangle displaying the track ID.
        Args:
            frame (np.ndarray): The image frame on which to draw.
            bbox (tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2) of the detected object.
            color (tuple[int, int, int]): The color (B, G, R) to use for drawing the ellipse and rectangle.
            track_id (int | None): The unique identifier for the tracked object. If None, the rectangle and text are not drawn.
        Returns:
            np.ndarray: The frame with the drawn ellipse and, if applicable, the rectangle and track ID.
        """
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

    def _draw_triangle(self, frame: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int]) -> np.ndarray:
        """
        Draws a filled triangle with an outlined border at the top center of the given bounding box on the frame.
        The triangle is positioned such that its base is at the top center of the bounding box, and its tip points upward.
        The triangle is filled with the specified color and outlined in black.
        Args:
            frame (np.ndarray): The image frame on which to draw the triangle.
            bbox (tuple[int, int, int, int]): The bounding box coordinates in the format (x1, y1, x2, y2).
            color (tuple[int, int, int]): The BGR color tuple for filling the triangle.
        Returns:
            np.ndarray: The frame with the triangle drawn on it.
        """
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

    def interpolate_ball_positions(self, ball_positions: list) -> list:
        """
        Interpolates missing ball positions in a sequence of frames.
        This method extracts bounding box coordinates for the ball from each frame,
        interpolates missing values using linear interpolation, and fills any remaining
        missing values by propagating the next valid observation backward. The result is
        a list of ball positions with missing data filled in.
        Args:
            ball_positions (list): A list of dictionaries containing ball detection data for each frame.
                Each dictionary should have the format {1: {"bbox": [x1, y1, x2, y2]}}.
        Returns:
            list: A list of dictionaries with interpolated ball positions in the same format as the input.

        Not in use currently as a few false positives were detected. False positive causes the interpolation to be unreliable.
        Will use the detected ball position with highest confidence in each frame instead.
        """
        # Extract the bounding box coordinates for the ball
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]

        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing ball positions
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit_direction='both')
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def draw_possession_info_box(
        self,
        frame: np.ndarray,
        frame_num: int,
        ball_possession_info: np.ndarray,
        team_1_name: str,
        team_2_name: str
    ) -> np.ndarray:
        """
        Draws a semi-transparent information box on the given frame displaying ball possession statistics for two teams.
        The box is rendered at the bottom right of the frame and shows the percentage of ball possession for each team
        up to the current frame number. The possession information is calculated based on the provided ball_possession_info array.
        Args:
            frame (np.ndarray): The current video frame on which to draw the possession info box.
            frame_num (int): The index of the current frame.
            ball_possession_info (np.ndarray): Array containing possession info per frame (1 for team 1, 2 for team 2).
            team_1_name (str): Name of the first team.
            team_2_name (str): Name of the second team.
        Returns:
            np.ndarray: The frame with the possession info box drawn.
        """

        # Draw a semi-transparent box at bottom right
        overlay = frame.copy()

        # Draw a semi-transparent rectangle
        cv2.rectangle(overlay, (frame.shape[1]-220, frame.shape[0]-110), (frame.shape[1]-10, frame.shape[0]-10), (255, 255, 255), -1)
        # cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )

        # Apply transparency
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Create a list of for all frames up to current frame
        possession_info_frame = ball_possession_info[:frame_num + 1]

        # Get the number of possession info
        team_1_possession_in_frames = possession_info_frame[possession_info_frame==1].shape[0]
        team_2_possession_in_frames = possession_info_frame[possession_info_frame==2].shape[0]
        total_possession = len(possession_info_frame)

        team_1_possession_percent = (team_1_possession_in_frames / total_possession) * 100 if total_possession > 0 else 0
        team_2_possession_percent = (team_2_possession_in_frames / total_possession) * 100 if total_possession > 0 else 0

        # Put possession info text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color_text = (0, 0, 0)  # Black color for text

        cv2.putText(frame, f"Ball Possession",(frame.shape[1]-210, frame.shape[0]-75), font, font_scale, color_text, thickness=2)
        cv2.putText(frame, f"{team_1_name}: {team_1_possession_percent:.0f}%",(frame.shape[1]-210, frame.shape[0]-50), font, font_scale, color_text, thickness=1)
        cv2.putText(frame, f"{team_2_name}: {team_2_possession_percent:.0f}%", (frame.shape[1]-210, frame.shape[0]-25), font, font_scale, color_text, thickness=1)

        # cv2.putText(frame, f"Team 1: {team_1_possession_percent:.0f}%", (1400, 900), font, font_scale, color_text, thickness=1)
        # cv2.putText(frame, f"Team 2: {team_2_possession_percent:.0f}%", (1400, 950), font, font_scale, color_text, thickness=1)

        return frame

    def draw_annotations(
        self,
        video_frames: list,
        tracking_list: dict,
        ball_possession_info: dict,
        team_1_name: str,
        team_2_name: str
    ) -> list:
        """
        Annotates video frames with player, ball, and referee tracking information, as well as ball possession details.
        Args:
            video_frames (list): List of video frames (numpy arrays) to annotate.
            tracking_list (dict): Dictionary containing tracking data for players, ball, and referees for each frame.
                Expected keys: "players", "ball", "referees", each mapping to a list of dicts per frame.
            ball_possession_info (dict): Dictionary containing ball possession information per frame.
            team_1_name (str): Name of the first team.
            team_2_name (str): Name of the second team.
        Returns:
            list: List of annotated video frames. Returns None if the number of frames in tracking_list does not match video_frames.
        """
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

                # Draw triangle to annotate player with ball. Will be False unless has_ball is set to True
                if player.get("has_ball", False):
                    frame = self._draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw ellipse to annotate referees
            for track_id, referee in referee_dict.items():
                frame = self._draw_ellipse(frame, referee["bbox"],(0,255,255), track_id)

            # Draw triangle to annotate ball
            for track_id, ball in ball_dict.items():
                frame = self._draw_triangle(frame, ball["bbox"],(0,255,0))

            # Draw possession info if available
            self.draw_possession_info_box(frame, frames_num, ball_possession_info, team_1_name, team_2_name)

            output_video_frames.append(frame)

        return output_video_frames
