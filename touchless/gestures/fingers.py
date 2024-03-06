from src.utils.landmarks import HandLandmarkPoints


def two_fingers_4_8(points: HandLandmarkPoints) -> bool:
    """Detect if thumb and index fingers are detected (4-8 gesture).

    Args:
        points (HandLandmarkPoints): The hand landmark points.

    Returns:
        bool: True if thumb and index fingers are detected, False otherwise.
    """

    return (
        points.thumb_tip.x < points.index_pip.x and
        points.thumb_mcp.y > points.thumb_tip.y and
        points.index_mcp.y > points.index_tip.y and
        points.middle_mcp.y < points.middle_tip.y and
        points.ring_mcp.y < points.ring_tip.y and
        points.pinky_mcp.y < points.pinky_tip.y
    )


def two_fingers_8_12(points: HandLandmarkPoints) -> bool:
    """Detect if thumb and index fingers are detected (8-12 gesture).

    Args:
        points (HandLandmarkPoints): The hand landmark points.

    Returns:
        bool: True if thumb and index fingers are detected, False otherwise.
    """

    return (
        points.thumb_tip.x > points.index_pip.x and
        points.index_tip.y < points.index_pip.y and
        points.middle_tip.y < points.middle_pip.y and
        points.ring_tip.y > points.ring_pip.y and
        points.pinky_tip.y > points.pinky_pip.y
    )


def two_fingers_8_20(points: HandLandmarkPoints) -> bool:
    """Detect if thumb and index fingers are detected (8-20 gesture).

    Args:
        points (HandLandmarkPoints): The hand landmark points.

    Returns:
        bool: True if thumb and index fingers are detected, False otherwise.
    """

    return(
        points.thumb_tip.x > points.index_pip.x and
        points.index_tip.y < points.index_mcp.y and
        points.middle_tip.y > points.middle_pip.y and
        points.ring_tip.y > points.ring_pip.y and
        points.pinky_tip.y < points.pinky_mcp.y
    )


def three_fingers_4_8_12(points: HandLandmarkPoints) -> bool:
    """Detect if thumb, index, and middle fingers are detected (4-8-12 gesture).

    Args:
        points (HandLandmarkPoints): The hand landmark points.

    Returns:
        bool: True if thumb, index, and middle fingers are detected, False otherwise.
    """

    return (
        points.thumb_tip.x < points.index_pip.x and
        points.thumb_mcp.y > points.thumb_tip.y and
        points.index_mcp.y > points.index_tip.y and
        points.middle_mcp.y > points.middle_tip.y and
        points.ring_mcp.y < points.ring_tip.y and
        points.pinky_mcp.y < points.pinky_tip.y
    )


def three_fingers_8_12_16(points: HandLandmarkPoints) -> bool:
    """Detect if thumb, index, and middle fingers are detected (8-12-16 gesture).

    Args:
        points (HandLandmarkPoints): The hand landmark points.

    Returns:
        bool: True if thumb, index, and middle fingers are detected, False otherwise.
    """

    return (
        points.thumb_tip.x > points.index_pip.x and
        points.index_tip.y < points.index_pip.y and
        points.middle_tip.y < points.middle_pip.y and
        points.ring_tip.y < points.ring_pip.y and
        points.pinky_tip.y > points.pinky_pip.y
    )


def five_fingers(points: HandLandmarkPoints) -> bool:
    """Detect if all fingers are extended.

    Args:
        points (HandLandmarkPoints): The hand landmark points.

    Returns:
        bool: True if all fingers are extended, False otherwise.
    """

    return (
        points.thumb_tip.y < points.thumb_mcp.y and
        points.index_tip.y < points.index_pip.y and
        points.middle_tip.y < points.middle_pip.y and
        points.ring_tip.y < points.ring_pip.y and
        points.pinky_tip.y < points.pinky_pip.y and
        (points.thumb_tip.x < points.index_tip.x < points.middle_tip.x < points.ring_tip.x < points.pinky_tip.x)
    )
