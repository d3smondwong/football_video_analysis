import cv2

def read_video(video_path: str) -> list:
    """
    Reads a video file and returns its frames as a list of images.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: A list of frames (numpy.ndarray) extracted from the video.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize a list to store the frames
    frames = []

    # Read frames from the video. ret is a boolean indicating if the frame was read successfully. frame is the actual frame.
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(ouput_video_frames: list, output_video_path: str):
    """
    Saves a sequence of video frames to a video file.

    Args:
        ouput_video_frames (list): List of video frames (numpy arrays) to be saved.
        output_video_path (str): Path to the output video file.

    Returns:
        None
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'XVID')

    # Assuming all frames have the same size, get the size (width x height) from the first frame. save videos at 24 frames per second
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))

    # Write each frame to the video file to string the frames together to be a video
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()