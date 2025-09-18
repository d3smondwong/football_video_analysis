import os
import hydra
import logging
import cv2

from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from inference_sdk import InferenceHTTPClient

from src.utils.video_utils import read_video, save_video
from src.trackers.tracker import Tracker
from src.team_identifier.team_identifier import TeamIdentifier

@hydra.main(config_path="../config", config_name="app.yaml", version_base="1.2")
def main(cfg: DictConfig):

    # Set up logging
    logger = logging.getLogger(os.path.basename(__file__))

    ###
    # Read the frames from the input video file
    ###

    input_dir = cfg.directories.input_video_directory
    video_name = cfg.video.video_name
    input_video_path = Path(get_original_cwd()) / input_dir / video_name

    # Checkers
    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")
    if not input_video_path.is_file():
        raise ValueError(f"Input path is not a file: {input_video_path}")
    if not input_video_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        raise ValueError(f"Unsupported video file format: {input_video_path.suffix}. Supported formats are .mp4, .avi, .mov.")

    # Read video frames
    logger.info(f"Reading video frames from: {input_video_path}")
    video_frames = read_video(str(input_video_path))

    ###
    # Trackers
    ###
    # Model Path
    model_directory = cfg.directories.fine_tuned_models
    model_path = Path(model_directory) / cfg.fine_tuned_model.yolo12m_best_model

    # Initialize the Tracker
    tracker = Tracker(str(model_path))

    # Create stubs directory if it does not exist
    stub_dir = cfg.directories.stubs
    stub_name = f"{input_video_path.stem}_stub.pkl"
    stub_dir_path = Path(get_original_cwd()) / stub_dir
    stub_file_path = stub_dir_path / stub_name

    if not stub_dir_path.exists():
        stub_dir_path.mkdir(parents=True, exist_ok=True)

    # Get object tracking from the video frames
    tracking_list = tracker.get_object_tracking(video_frames,
                                       read_from_stub=True,
                                       stub_path=stub_file_path
                                       )

    ###
    # Interpolate the ball positions
    ###
    # Commenting out the ball interpolation for now as the ball is not tracked very well. Probably due to training data size. Will use the highest confidence detected ball in each frame instead
    # if tracking_list is not None and tracking_list.get("ball") is not None:
    #     tracking_list["ball"] = tracker.interpolate_ball_positions(tracking_list["ball"])
    # else:
    #     logger.error("No ball tracking data found in tracking_list['ball'].")
    #     return

    ###
    # Identify team for each player using jersey color
    ###
    team_identifier = TeamIdentifier()
    if tracking_list is not None and tracking_list.get("players") is not None and len(tracking_list["players"]) > 0:

        # Initialize Kmeans and team colors using the detected players in the first frame
        team_identifier.identify_team_colors(video_frames[0], tracking_list["players"][0])

        # Iterate through all frames and players to assign team colors
        for frame_num, player_tracking in enumerate(tracking_list["players"]):

            # Update team colors for each player in the current frame
            for player_id, player_info in player_tracking.items():
                # logger.info(f"player_id: {player_id}, Player Info: {player_info}, Team Color: {team_identifier.team_colors}")

                """
                player_tracking.items() consist of
                a key: player_id (np.int64)
                a value: player_info dict with the following: 'bbox':[x1, y1, x2, y2] , 'team': np.int32(2) or np.int32(1), 'team_color': array([ 57.094, 57.212, 41.525])
                """
                # Get the team color for the player
                team_colors = team_identifier.get_player_team(video_frames[frame_num],
                                                                player_info['bbox'],
                                                                player_id)

                # Update the tracking list with team and team_color
                tracking_list['players'][frame_num][player_id]['team'] = team_colors
                tracking_list['players'][frame_num][player_id]['team_color'] = team_identifier.team_colors[team_colors]
    else:
        logger.error("No player tracking data found in tracking_list['players'].")
        return



    ###
    # Draw output on the video frames
    ###
    output_video_frames = tracker.draw_annotations(video_frames, tracking_list)
    if output_video_frames is None:
        logger.error("No output video frames were generated. Please check tracker.draw_annotations method.")
        return

    ###
    # Save video frames to the output video file
    ###

    # Output video path
    output_dir = Path(cfg.directories.output_video_directory)
    output_name = f"{input_video_path.stem}_analysis.avi"
    output_video_path = Path(get_original_cwd()) / output_dir / output_name

    # Check if the parent directory of the output video path exists, if not, create it
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving video frames to: {output_video_path}")

    # Save processed video frames to a new video file
    save_video(output_video_frames, str(output_video_path))

if __name__ == "__main__":
    # python -m src.app
    main()