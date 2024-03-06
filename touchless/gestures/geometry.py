import cv2
import numpy as np
from src.utils.math_utils import euclidean, angle_between_vectors
from src.utils.landmarks import HandLandmarkPoints


def change_magnitude(
    points: HandLandmarkPoints,
    image: np.ndarray,
    img_size: tuple[int,int],
    roi: list[tuple[int, int]] | None = None,
    connect: bool = True
) -> tuple[bool, float | None]:
    """Change the magnitude of an image based on hand landmark points.

    Args:
        points (HandLandmarkPoints): The hand landmark points.
        image (np.ndarray): The input image.
        img_size (tuple[int,int]): The size of the image (width, height).
        roi (list[tuple[int, int]] | None, optional): Region of interest (top-left and bottom-right corners). Defaults to None.
        connect (bool, optional): Whether to draw a line connecting thumb and index tips. Defaults to True.

    Returns:
        tuple[bool, float | None]: A tuple indicating success (True/False) and the magnitude change value (if successful).
    """

    if roi is None:
        roi = [(0, 0), img_size]

    width, height = img_size

    thumb_tip_px = (int(points.thumb_tip.x * width), int(points.thumb_tip.y * height))
    index_tip_px = (int(points.index_tip.x * width), int(points.index_tip.y * height))

    if (
        thumb_tip_px[0] >= roi[0][0] and
        thumb_tip_px[1] >= roi[0][1] and
        index_tip_px[0] >= roi[0][0] and
        index_tip_px[1] >= roi[0][1] and

        thumb_tip_px[0] <= roi[1][0] and
        thumb_tip_px[1] <= roi[1][1] and
        index_tip_px[0] <= roi[1][0] and
        index_tip_px[1] <= roi[1][1]
    ):
        if connect:
            cv2.line(image, pt1=thumb_tip_px, pt2=index_tip_px, color=(0, 255, 0), thickness=2)
        return True, euclidean(pt1=thumb_tip_px, pt2=index_tip_px)
    else:
        return False, None


def vertical_angle(vertex: tuple[int, int], pointer: tuple[int, int], frame_height: int) -> float:
    """Calculate the vertical angle between a vertex and a pointer.

    Args:
        vertex (tuple[int, int]): The vertex coordinates (x, y).
        pointer (tuple[int, int]): The pointer coordinates (x, y).
        frame_height (int): The height of the frame.

    Returns:
        float: The vertical angle.
    """
    v1 = (0, frame_height)
    v2 = (pointer[0] - vertex[0], pointer[1] - vertex[1])
    angle = angle_between_vectors(v1, v2)

    """
    Define angle sign:
    - positive, if the pointer is in the right half-plane
    - negative, if the pointer is in the left half-plane
    """

    if pointer[0] < vertex[0]:
        angle = -angle

    return angle
