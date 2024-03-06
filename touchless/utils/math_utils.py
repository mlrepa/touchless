import math


def angle_between_vectors(
    v1: tuple[int, int],
    v2: tuple[int, int],
    in_degrees: bool = True,
    accuracy: int = 3
) -> float:
    """Calculate the angle (in radians or degrees) between two vectors.

    Args:
        v1 (tuple[int, int]): The first vector.
        v2 (tuple[int, int]): The second vector.
        in_degrees (bool, optional): Whether to return the angle in degrees. Defaults to True.
        accuracy (int, optional): The number of decimal places for rounding the result. Defaults to 3.

    Returns:
        float: The angle between the vectors.
    """

    v1_len: float = math.hypot(*v1)
    v2_len: float = math.hypot(*v2)
    scalar_product = v1[0] * v2[0] + v1[1] * v2[1]
    angle: float = math.acos(scalar_product / (v1_len * v2_len))

    if in_degrees:
        angle = math.degrees(angle)

    angle = round(angle, accuracy)

    return angle


def euclidean(pt1: tuple[float, float], pt2: tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points.

    Args:
        pt1 (tuple[float, float]): The coordinates of the first point.
        pt2 (tuple[float, float]): The coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    d: float = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    return d
