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


def heron_area_by_points(p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int], accuracy: int = 3) -> float:
    """Calculate triangle area by its vertices using Heron's formula.

    Args:
        p1 (tuple[int, int]): The first point.
        p2 (tuple[int, int]): The second point.
        p3 (tuple[int, int]): The third point.
        accuracy (int, optional): The number of decimal places for rounding the result. Defaults to 3.

    Returns:
        float: Area of a triangle.
    """
    
    a: float = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    b: float = math.hypot(p3[0] - p1[0], p3[1] - p1[1])
    c: float = math.hypot(p3[0] - p2[0], p3[1] - p2[1])

    p: float = (a + b + c) / 2
    s: float = math.sqrt(p * (p - a) * (p - b) * (p - c))
    s = round(s, accuracy)

    return s


def dist_from_triangle_0_5_17_to_camera(area: float, accuracy: int = 3) -> float:
    """Calculate a distance from a triangle (wrist, index and pinky fingers MCP) to camera.
    Notes:
    - approximation function y = k/x + b is used for distance calculation
    - coefficients k and b are computed by collected data result and may be improved later.

    Args:
        area (float): The triangle area.
        accuracy (int, optional): The number of decimal places for rounding the result. Defaults to 3.

    Returns:
        float: Distance to camera.
    """

    k: float = 103703.3834876518
    b: float = 15.012570158084339
    dist: float = round(k / area + b, accuracy)

    return dist
