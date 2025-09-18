import os
import hydra

from ultralytics import YOLO
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from inference_sdk import InferenceHTTPClient
from supervision.detection.core import Detections

from src.utils.video_utils import read_video, save_video

@hydra.main(config_path="../config", config_name="app.yaml", version_base="1.2")
def main(cfg: DictConfig):
    def main(cfg: DictConfig) -> None:
        """
        To test the model inference using YOLOv12x on a sample video.

        Args:
            cfg (DictConfig): Configuration object containing directories and video settings.
        Workflow:
            - Download the YOLOv12x model if not already present.
            - Reads the input video from the specified directory.
            - Loads the YOLO model for object detection.
            - Runs prediction on the input video, saving results and detection boxes.
            - Creates a folder runs-detect-<video_name> to store the output video and text files.
            - Prints summary information and detection results.
        Note:
            The function assumes the existence of supporting functions and classes such as
            `read_video`, `YOLO`, and `get_original_cwd`.
        """

    input_dir = cfg.directories.input_video_directory
    video_name = cfg.video.video_name
    input_video_path = Path(get_original_cwd()) / input_dir / video_name

    video_frames = read_video(str(input_video_path))
    print(f"Read {len(video_frames)} frames from the video.")

    # Initialize Roboflow InferenceHTTPClient
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if api_key is None:
        raise ValueError("ROBOFLOW_API_KEY environment variable is not set.")

    model_id = cfg.model.id

    client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
    )

    detections_per_frame = []
    for frame in video_frames:
        result = client.infer(inference_input=frame, model_id=model_id)
        detections_per_frame.append(result)

    # output_dir = cfg.directories.output_video_directory
    # output_video_path = Path(f"{output_dir}/{input_video_path.stem}_analysis.avi")
    # print(f"Saving video frames to: {output_video_path}")

    # # Save processed video frames to a new video file
    # save_video(detections_per_frame, str(output_video_path))

if __name__ == "__main__":
    # python -m model_inference
    main()