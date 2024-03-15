from typing import NamedTuple

from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from mediapipe.python.solutions.hands import HandLandmark
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass


@dataclass
class Point:
    """A 3D point in space represented by its coordinates (x, y, z)."""
    x: float
    y: float
    z: float


class HandLandmarkPoints(BaseModel):
    """A class representing landmarks of a hand."""
    wrist: Point = Field(alias="WRIST")
    thumb_cmc: Point = Field(alias="THUMB_CMC")
    thumb_mcp: Point = Field(alias="THUMB_MCP")
    thumb_ip: Point = Field(alias="THUMB_IP")
    thumb_tip: Point = Field(alias="THUMB_TIP")
    index_mcp: Point = Field(alias="INDEX_FINGER_MCP")
    index_pip: Point = Field(alias="INDEX_FINGER_PIP")
    index_dip: Point = Field(alias="INDEX_FINGER_DIP")
    index_tip: Point = Field(alias="INDEX_FINGER_TIP")
    middle_mcp: Point = Field(alias="MIDDLE_FINGER_MCP")
    middle_pip: Point = Field(alias="MIDDLE_FINGER_PIP")
    middle_dip: Point = Field(alias="MIDDLE_FINGER_DIP")
    middle_tip: Point = Field(alias="MIDDLE_FINGER_TIP")
    ring_mcp: Point = Field(alias="RING_FINGER_MCP")
    ring_pip: Point = Field(alias="RING_FINGER_PIP")
    ring_dip: Point = Field(alias="RING_FINGER_DIP")
    ring_tip: Point = Field(alias="RING_FINGER_TIP")
    pinky_mcp: Point = Field(alias="PINKY_MCP")
    pinky_pip: Point = Field(alias="PINKY_PIP")
    pinky_dip: Point = Field(alias="PINKY_DIP")
    pinky_tip: Point = Field(alias="PINKY_TIP")


def get_pointer(points: HandLandmarkPoints | None, img_size: tuple[int, int]) -> tuple[int, int] | None:
    """Get pointer - coordinates of the index finger TIP.

    Args:
        points (HandLandmarkPoints | None): Landmarks for one hand or None if no hand detected.
        img_size (tuple[int, int]): Image size = (width, height).

    Returns:
        tuple[int, int] | None: 
            The index finger tip coordinates if a hand landmarks is not None,
            otherwise None.
    """

    if points:
        x, y = points.index_tip.x, points.index_tip.y
        width, height = img_size
        pointer = (int(width * x), int(height * y))
        return pointer
    return None
