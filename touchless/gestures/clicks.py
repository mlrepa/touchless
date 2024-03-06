from src.gestures.fingers import two_fingers_4_8, two_fingers_8_12
from src.utils.landmarks import HandLandmarkPoints
from src.utils.math_utils import euclidean


def click_8_12(points: HandLandmarkPoints, dist_threshold: float = 0.05) -> bool:
    """Detect if a click gesture (index and middle fingers close together) is performed.
    
    Args:
        points (HandLandmarkPoints): The hand landmark points.
        dist_threshold (float, optional): The distance threshold for considering the fingers as clicked. Defaults to 0.05.

    Returns:
        bool: True if the click gesture is detected, False otherwise.
    """

    dist: float = euclidean(
        pt1=(points.index_tip.x, points.index_tip.y),
        pt2=(points.middle_tip.x, points.middle_tip.y)
    )
    
    return (
        two_fingers_8_12(points) and
        dist < dist_threshold
    )


def click_4_6(points: HandLandmarkPoints, dist_threshold: float = 0.04) -> bool:
    """Detect if a click gesture (thumb and index fingers close together) is performed.
    
    Args:
        points (HandLandmarkPoints): The hand landmark points.
        dist_threshold (float, optional): The distance threshold for considering the fingers as clicked. Defaults to 0.04.

    Returns:
        bool: True if the click gesture is detected, False otherwise.
    """

    dist: float = euclidean(
        pt1=(points.thumb_tip.x, points.thumb_tip.y),
        pt2=(points.index_pip.x, points.index_pip.y)
    )

    return (
        two_fingers_4_8(points) and
        dist < dist_threshold
    )


def click_6_8(points: HandLandmarkPoints) -> bool:
    """Detect if a click gesture (thumb moving upwards) is performed.
    
    Args:
        points (HandLandmarkPoints): The hand landmark points.

    Returns:
        bool: True if the click gesture is detected, False otherwise.
    """

    return (
        points.thumb_tip.x > points.index_pip.x and
        points.thumb_tip.y > points.middle_pip.y and
        points.middle_mcp.y < points.middle_tip.y and
        points.ring_mcp.y < points.ring_tip.y and
        points.pinky_mcp.y < points.pinky_tip.y and
        (points.index_dip.y <= points.index_tip.y < points.index_mcp.y)
    )
