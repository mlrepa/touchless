from src.utils.landmarks import HandLandmarkPoints


def fist_closed(points: HandLandmarkPoints) -> bool:
    """Detect if the hand is in a closed fist gesture.

    Args:
        points (HandLandmarkPoints): The hand landmark points.

    Returns:
        bool: True if the hand is in a closed fist gesture, False otherwise.
    """

    if hand_up(points):
        return (
            points.index_mcp.y < points.index_tip.y and
            points.middle_mcp.y < points.middle_tip.y and
            points.ring_mcp.y < points.ring_tip.y and
            points.pinky_mcp.y < points.pinky_tip.y
        )

    if hand_down(points):
        return (
            points.index_pip.y > points.index_tip.y and
            points.middle_pip.y > points.middle_tip.y and
            points.ring_pip.y > points.ring_tip.y and
            points.pinky_pip.y > points.pinky_tip.y
        )


def hand_down(points: HandLandmarkPoints) -> bool:
    """Detect if the hand is facing downwards.

    Args:
        points (HandLandmarkPoints): The hand landmark points.

    Returns:
        bool: True if the hand is facing downwards, False otherwise.
    """

    return points.middle_tip.y > points.wrist.y


def hand_up(points: HandLandmarkPoints) -> bool:
    """Detect if the hand is facing upwards.

    Args:
        points (HandLandmarkPoints): The hand landmark points.

    Returns:
        bool: True if the hand is facing upwards, False otherwise.
    """

    return points.middle_tip.y < points.wrist.y
