import hydra

from ultralytics import YOLO
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

from datetime import datetime

@hydra.main(config_path="../config", config_name="app.yaml", version_base="1.2")
def main(cfg: DictConfig):
    def main(cfg: DictConfig) -> None:
        """
        To test the fine tune or raw model on a sample video to see the inference results.

        Args:
            cfg (DictConfig): Configuration object containing directories and video settings.
        Workflow:
            - Utilises the model path to retrieve a saved model or download the model if not already present.
            - Loads the YOLO model for object detection.
            - Runs prediction on the input video, saving results and detection boxes.
            - Creates a folder runs-detect-<video_name>_<timestamp> to store the output video and text files.
        Note:
            The function assumes the existence of supporting functions and classes such as
            `YOLO`, and `get_original_cwd`.
        """

    # Construct the full path to the input video
    input_dir = cfg.directories.input_video_directory
    video_name = cfg.video.video_name
    input_video_path = Path(get_original_cwd()) / input_dir / video_name

    # Load the fine-tuned YOLO model
    ft_models_directory = cfg.directories.fine_tuned_models
    best_model_path = Path(ft_models_directory) / "yolo12m_best.pt"

    model = YOLO(best_model_path)

    # Ensure a unique output directory for each prediction by using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{input_video_path.stem}_{timestamp}"

    # Run prediction on the input video and save results
    results = model.predict(
        str(input_video_path),
        save=True,
        save_txt=True,
        project="model_test/runs/detect",
        name=output_name,
        exist_ok=True
    )

    print(results[0])
    print("========================================")
    if results[0].boxes is not None:
        for box in results[0].boxes:
            print(box)
    else:
        print("No boxes found in the results.")

if __name__ == "__main__":
    # python -m model_test.model_inference
    main()