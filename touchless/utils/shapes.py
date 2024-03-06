from collections.abc import Callable

import cv2
import numpy as np

from src.utils.events import is_shape_selected


SHAPES: list[dict] = [
    {
        "type": "circle",
        "name": "red_circle",
        "params": {
            "default": {
                "center": (100, 100),
                "radius": 30,
                "color": (0, 0, 255),
                "thickness": 2
            },
            "on_select": {
                "thickness": -1
            }
        }
    },
    {
        "type": "circle",
        "name": "yellow_circle",
        "params": {
            "default": {
                "center": (100, 200),
                "radius": 30,
                "color": (0, 255, 255)
            },
            "on_select": {
                "thickness": -1
            }
        }
    },
    {
        "type": "circle",
        "name": "green_circle",
        "params": {
            "default": {
                "center": (100, 300),
                "radius": 30,
                "color": (0, 255, 0)
            },
            "on_select": {
                "thickness": -1
            }
        }
    },
    {
        "type": "rectangle",
        "name": "rect1",
        "params": {
            "default": {
                "pt1": (160, 10),
                "pt2": (210, 60),
                "color": (100, 150, 200)
            },
            "on_select": {
                "thickness": 10
            }
        }
    },
    {
        "type": "rectangle",
        "name": "rect2",
        "params": {
            "default": {
                "pt1": (360, 10),
                "pt2": (410, 60),
                "color": (200, 150, 200)
            },
            "on_select": {
                "thickness": 10
            }
        }
    },
    {
        "type": "rectangle",
        "name": "rect3",
        "params": {
            "default": {
                "pt1": (560, 10),
                "pt2": (610, 60),
                "color": (150, 150, 200)
            },
            "on_select": {
                "thickness": 10
            }
        }
    }
]


DRAW_SHAPE: dict[str, Callable] = {
    "circle": cv2.circle,
    "rectangle": cv2.rectangle
}


def draw_pointer(frame: np.ndarray, pointer: tuple[int, int] | None) -> None:
    """Draw (render) pointer - point of the index finger TIP position.

    Args:
        frame (np.ndarray): Frame which pointer draw on.
        pointer (Tuple[int, int] | None): Pointer coordinates tuple.
    """
    if pointer is not None:
        cv2.circle(frame, center=pointer, radius=5, color=(255, 255, 100), thickness=-1)


def draw_shape(frame: np.ndarray, shape: dict, pointer: tuple[int, int] | None) -> None:
    """Draw (render) single shape by its definition.

    Args:
        frame (np.ndarray): Frame which shape draw on.
        shape (Dict): Shape definition dictionary.
        pointer (Tuple[int, int] | None): Pointer coordinates tuple.
    """

    event: str = "default"

    if is_shape_selected(shape=shape, pointer=pointer):
        event = "on_select"

    params = shape["params"]["default"].copy()
    params.update(shape["params"][event])
    DRAW_SHAPE[shape["type"]](frame, **params)


def render_shapes(frame: np.ndarray, shapes: list[dict], pointer: tuple[int, int] | None) -> None:
    """Render shapes from shapes list.

    Args:
        frame (np.ndarray): Frame which shapes draw on.
        shapes (List[Dict]): List of shapes definitions.
        pointer (Tuple[int, int] | None): Pointer coordinates tuple.
    """
    for shape in shapes:
        draw_shape(frame, shape, pointer)


def draw_angle_vertex(frame: np.ndarray, vertex: tuple[int, int]) -> None:
    """Draw a vertex as a circle on the frame.

    Args:
        frame (np.ndarray): The image frame.
        vertex (tuple[int, int]): The coordinates of the vertex (x, y).
    """

    cv2.circle(frame, center=vertex, radius=3, color=(0, 0, 255), thickness=-1)


def draw_vertical(frame: np.ndarray, vertex: tuple[int, int], frame_height: int) -> None:
    """Draw a vertical line from a vertex to the bottom of the frame.

    Args:
        frame (np.ndarray): The image frame.
        vertex (tuple[int, int]): The coordinates of the vertex (x, y).
        frame_height (int): The height of the frame.
    """

    cv2.line(frame, pt1=vertex, pt2=(vertex[0], frame_height), color=(0, 0, 225), thickness=2)


def draw_angle(frame: np.ndarray, vertex: tuple[int, int], pointer: tuple[int, int]) -> None:
    """Draw a line representing an angle between a vertex and a pointer on the frame.

    Args:
        frame (np.ndarray): The image frame.
        vertex (tuple[int, int]): The coordinates of the vertex (x, y).
        pointer (tuple[int, int]): The coordinates of the pointer (x, y).
    """

    cv2.line(frame, pt1=vertex, pt2=pointer, color=(0, 0, 225), thickness=2)
