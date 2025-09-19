from src.utils.bbox_utils import get_center_of_bbox

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 100  # Maximum distance to consider a player "with the ball"

    def assign_ball_to_player(self, player_info, ball_bbox):

        if not player_info or ball_bbox is None:
            return None

        # Get the center position of the ball
        ball_position = get_center_of_bbox(ball_bbox)
        closest_player_id = None
        min_distance = float('inf')

        # Iterate through all players to find the closest player to the ball
        for player_id, player_info in player_info.items():

            # Get the center position of the player
            player_center = get_center_of_bbox(player_info['bbox'])

            # Calculate the Euclidean distance between the player and the ball
            distance = ((player_center[0] - ball_position[0]) ** 2 + (player_center[1] - ball_position[1]) ** 2) ** 0.5

            # Update the closest player if this player is closer
            if distance < min_distance and distance <= self.max_player_ball_distance:
                min_distance = distance
                closest_player_id = player_id

        return closest_player_id