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
        Extracts the dominant jersey color of a player from the frame using KMeans clustering.

        Args:
            frame (np.ndarray): The image frame containing the player.
            bbox (list): Bounding box coordinates [x1, y1, x2, y2] for cropping the player.

        Returns:
            np.ndarray: The RGB value of the player's jersey color.
        """

        # Crop the player from the frame using the bounding box coordinates
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Focus on the upper half of the player (jersey area)
        height = image.shape[0]
        top_half_image = image[0:int(height/2), :, :]

        # Reshape the image (H, W, Color) into 2d array (Pixel, Color). KMeans needs 2d array as input
        image_2d = top_half_image.reshape(-1, 3)

        # Clustering model. init with 2 clusters (1 for jersey color, 1 for background(field)), k-means++ for better initialisation, n_init=1 to avoid FutureWarning
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=0)
        kmeans.fit(image_2d)

        # Get the cluster labels (0 or 1) for each pixel
        labels = kmeans.labels_

        # Reshape the labels into the orginal image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Since the player is usually in the center of the image, we can assume that the cluster at the corners is the background (Field)
        # We need to find the cluster label that is allocated to the team jersey
        # Get the cluster labels at the 4 corners of the image
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]

        # Get the most common cluster label among the corners. set(corner_clusters) gets all unique values. max() with key=corner_clusters.count gets the most common value
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

        player_cluster = 1-non_player_cluster

        # RGB value of the player jersey color
        player_color = kmeans.cluster_centers_[player_cluster]

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
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=0)
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
