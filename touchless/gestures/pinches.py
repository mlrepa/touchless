from src.utils.landmarks import HandLandmarkPoints
from src.utils.math_utils import euclidean


def get_dists_for_all_fingers(points: HandLandmarkPoints) -> tuple[float, float, float, float]:
    """
    Calculate the distances between thumb tip and tips of all other fingers.

    Args:
        points (HandLandmarkPoints): Instance of HandLandmarkPoints containing landmark points.

    Returns:
        tuple: A tuple containing distances between thumb tip and tips of index, middle, ring, and pinky fingers.
    """

    thumb_index_dist: float = euclidean(
        pt1=(points.thumb_tip.x, points.thumb_tip.y),
        pt2=(points.index_tip.x, points.index_tip.y)
    )
    thumb_middle_dist: float = euclidean(
        pt1=(points.thumb_tip.x, points.thumb_tip.y),
        pt2=(points.middle_tip.x, points.middle_tip.y)
    )
    thumb_ring_dist: float = euclidean(
        pt1=(points.thumb_tip.x, points.thumb_tip.y),
        pt2=(points.ring_tip.x, points.ring_tip.y)
    )
    thumb_pinky_dist: float = euclidean(
        pt1=(points.thumb_tip.x, points.thumb_tip.y),
        pt2=(points.pinky_tip.x, points.pinky_tip.y)
    )

    return (thumb_index_dist, thumb_middle_dist, thumb_ring_dist, thumb_pinky_dist)


def pinch_4_8(points: HandLandmarkPoints) -> bool:
    """
    Check if the hand is in the 4-8 pinch gesture.

    Args:
        points (HandLandmarkPoints): Instance of HandLandmarkPoints containing landmark points.

    Returns:
        bool: True if the hand is in 4-8 pinch gesture, False otherwise.
    """

    thumb_index_dist, thumb_middle_dist, thumb_ring_dist, thumb_pinky_dist = get_dists_for_all_fingers(points)

    return (
        thumb_index_dist < 0.03 and
        thumb_middle_dist > 0.05 and
        thumb_ring_dist > 0.05 and
        thumb_pinky_dist > 0.05
    )


def pinch_4_12(points: HandLandmarkPoints) -> bool:
    """
    Check if the hand is in the 4-12 pinch gesture.

    Args:
        points (HandLandmarkPoints): Instance of HandLandmarkPoints containing landmark points.

    Returns:
        bool: True if the hand is in 4-12 pinch gesture, False otherwise.
    """

    thumb_index_dist, thumb_middle_dist, thumb_ring_dist, thumb_pinky_dist = get_dists_for_all_fingers(points)

    return (
        thumb_middle_dist < 0.03 and
        thumb_index_dist > 0.05 and
        thumb_ring_dist > 0.05 and
        thumb_pinky_dist > 0.05
    )


def pinch_4_16(points: HandLandmarkPoints) -> bool:
    """
    Check if the hand is in the 4-16 pinch gesture.

    Args:
        points (HandLandmarkPoints): Instance of HandLandmarkPoints containing landmark points.

    Returns:
        bool: True if the hand is in 4-16 pinch gesture, False otherwise.
    """

    thumb_index_dist, thumb_middle_dist, thumb_ring_dist, thumb_pinky_dist = get_dists_for_all_fingers(points)

    return (
        thumb_ring_dist < 0.03 and
        thumb_index_dist > 0.05 and
        thumb_middle_dist > 0.05 and
        thumb_pinky_dist > 0.05
    )


def pinch_4_20(points: HandLandmarkPoints) -> bool:
    """
    Check if the hand is in the 4-20 pinch gesture.

    Args:
        points (HandLandmarkPoints): Instance of HandLandmarkPoints containing landmark points.

    Returns:
        bool: True if the hand is in 4-20 pinch gesture, False otherwise.
    """

    thumb_index_dist, thumb_middle_dist, thumb_ring_dist, thumb_pinky_dist = get_dists_for_all_fingers(points)

    return (
        thumb_pinky_dist < 0.03 and
        thumb_index_dist > 0.05 and
        thumb_middle_dist > 0.05 and
        thumb_ring_dist > 0.05
    )
