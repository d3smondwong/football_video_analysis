from src.utils.video_utils import read_video, save_video
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

@hydra.main(config_path="../config", config_name="app.yaml", version_base="1.2")
def main(cfg: DictConfig):

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
    print(f"Reading video frames from: {input_video_path}")
    video_frames = read_video(str(input_video_path))

    # Process frames (this is just a placeholder for actual processing logic)
    processed_frames = [frame for frame in video_frames]  # No processing in this example

    # Ensure output directory exists
    output_video_path = Path("output_videos/output_video.avi")

    # Check if the parent directory of the output video path exists, if not, create it
    if not output_video_path.parent.exists():
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    ###
    # Save video frames to the output video file
    ###
    output_dir = cfg.directories.output_video_directory
    output_video_path = Path(f"{output_dir}/{input_video_path.stem}_analysis.avi")
    print(f"Saving video frames to: {output_video_path}")

    # Save processed video frames to a new video file
    save_video(processed_frames, str(output_video_path))

if __name__ == "__main__":
    # python -m src.app
    main()