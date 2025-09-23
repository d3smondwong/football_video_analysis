from sklearn.cluster import KMeans
import numpy as np

class TeamIdentifier:
    """Class to identify teams based on jersey colors using KMeans clustering."""

    def __init__(self):
        """
        Initializes the TeamIdentifier class.

        Attributes:
            team_colors (dict): A dictionary to store team colors.
            player_team_dict (dict): A dictionary mapping players to their respective teams.
        """
        self.team_colors = {}
        self.player_team_dict = {}

    def get_player_color(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extracts the dominant jersey color of a player from the frame by defining a region of interest and using KMeans clustering.

        Args:
            frame (np.ndarray): The image frame containing the player.
            bbox (list): Bounding box coordinates [x1, y1, x2, y2] for cropping the player.

        Returns:
            np.ndarray: The RGB value of the player's jersey color.
        """

        # Crop the player from the frame using the bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)
        image = frame[y1:y2, x1:x2]

        # Define a larger, central region for sampling the color
        height, width, _ = image.shape
        center_x = width // 2
        center_y = int(height * 0.25)  # Focus on the upper half

        # A larger sample region of 20% of the image's height and 30% of its width
        region_height = int(height * 0.2)
        region_width = int(width * 0.3)

        # Ensure the region is within the image bounds
        top_y = max(0, center_y - region_height // 2)
        bottom_y = min(height, center_y + region_height // 2)
        left_x = max(0, center_x - region_width // 2)
        right_x = min(width, center_x + region_width // 2)

        sample_region = image[top_y:bottom_y, left_x:right_x]

        if sample_region.size == 0:
            return np.zeros(3) # Return black if the sample region is empty

        # Reshape the pixels to be used with KMeans
        pixels = sample_region.reshape(-1, 3)

        # Use KMeans with a single cluster to find the average color of the region
        try:
            kmeans = KMeans(n_clusters=1, init='k-means++', n_init=1, random_state=0)
            kmeans.fit(pixels)
            player_color = kmeans.cluster_centers_[0]
        except ValueError:
            # Fallback to simple mean if clustering fails
            player_color = np.mean(pixels, axis=0)

        return player_color

    def identify_team_colors(
        self,
        frame: np.ndarray,
        player_detections: dict
    ) -> list:
        """
        Identifies the dominant colors of player jerseys in a given video frame and clusters them to determine team colors.
        Args:
            frame (np.ndarray): The current video frame containing players.
            player_detections (dict): A dictionary of player detections, where each value contains a 'bbox' key representing the bounding box coordinates of a player.
        Returns:
            list: A list of dominant colors (as arrays or tuples) for each detected player in the frame.
        Side Effects:
            - Fits a KMeans clustering model to the detected player colors.
            - Updates `self.kmeans` with the fitted KMeans model.
            - Updates `self.team_colors` with the identified team colors from the cluster centers.
        """
        player_colors = []

        for _, player_detection in player_detections.items():
            # Extract the bounding box coordinates
            bbox = player_detection['bbox']

            # Get the dominant color of the player jersey using clustering
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Cluster the player colors to identify teams
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=0)
        kmeans.fit(player_colors)

        # Save the KMeans model after fitting for future use
        self.kmeans = kmeans

        # The cluster centers represent the team colors
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        return player_colors

    def get_player_team(self, frame: np.ndarray, player_bbox: tuple, player_id: int) -> int:
        """
        Determines the team label for a given player in a video frame.
        If the player's team has already been identified, returns the stored team label.
        Otherwise, extracts the dominant jersey color from the player's bounding box,
        predicts the team using a k-means clustering model, stores the result, and returns the team label.
        Args:
            frame (np.ndarray): The current video frame containing the player.
            player_bbox (tuple): The bounding box coordinates of the player (x, y, w, h).
            player_id (int): The unique identifier for the player.
        Returns:
            int: The team label assigned to the player (1 or 2).
        """

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Get the dominant color of the player jersey using clustering
        player_color = self.get_player_color(frame, player_bbox)

        # Predict the team based on the closest cluster center
        team_label = self.kmeans.predict(np.array(player_color).reshape(1, -1))[0]
        team_label += 1  # To make team labels 1 and 2 instead of 0 and 1

        # Store the player ID and their corresponding team label
        self.player_team_dict[player_id] = team_label

        return team_label
