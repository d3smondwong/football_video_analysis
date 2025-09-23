class BallPossession:
    """
    Tracks ball possession information for each frame in a football video analysis.
    Attributes:
        possession_data (list[Optional[int]]): Stores possession info for each frame, where each entry is a team number or None.
    Methods:
        add_possession(team: Optional[int]) -> None:
            Adds possession info for a frame. The team parameter can be a team number or None.
        get_possession_data() -> list[Optional[int]]:
            Returns possession info for all frames as a list.
        get_possession_stats() -> tuple[dict[int, int], int]:
            Returns possession statistics as a tuple containing:
                - A dictionary mapping team numbers to their possession counts.
                - The total number of frames with valid (non-None) possession data.
    """
    def __init__(self) -> None:
        self.possession_data: list[int | None] = []

    def add_possession(self, team: int | None) -> None:
        self.possession_data.append(team)

    def get_possession_data(self) -> list[int | None]:
        return self.possession_data

    def get_possession_stats(self) -> tuple[dict[int, int], int]:
        """
        Calculates possession statistics based on recorded team possession data.

        Returns:
            tuple[dict[int, int], int]:
                - A dictionary mapping each team ID (int) to the number of times they possessed the ball (int).
                - The total number of valid possession data points (int).
        """
        valid_data: list[int] = [team for team in self.possession_data if team is not None]
        total: int = len(valid_data)
        stats: dict[int, int] = {}
        for team in set(valid_data):
            stats[team] = valid_data.count(team)
        return stats, total
