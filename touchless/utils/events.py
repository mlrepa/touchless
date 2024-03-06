from collections.abc import Callable


def is_circle_selected(circle: dict, pointer: tuple[int, int]) -> bool:
    """Check if a circle selected.

    Args:
        circle (dict): Circle object definition.
        pointer (tuple[int, int]): Coordinates of the pointer.

    Returns:
        bool: True if the circle is selected, otherwise False.
    """
    
    params: dict = circle["params"]["default"]
    cx, cy = params["center"]
    cr = params["radius"]
    px, py = pointer

    if (cx - px) ** 2 + (cy - py) ** 2 <= cr ** 2:
        return True
    return False


def is_rectangle_selected(rectangle: dict, pointer: tuple[int, int]) -> bool:
    """Check if a rectangle selected.

    Args:
        circle (dict): Rectangle object definition.
        pointer (tuple[int, int]): Coordinates of the pointer.

    Returns:
        bool: True if the rectangle is selected, otherwise False.
    """

    params: dict = rectangle["params"]["default"]
    x1, y1 = params["pt1"]
    x2, y2 = params["pt2"]
    px, py = pointer

    if  (x1 <= px <= x2) and (y1 <= py <= y2):
        return True
    return False


IS_SELECTED: dict[str, Callable] = {
    "circle": is_circle_selected,
    "rectangle": is_rectangle_selected
}

def is_shape_selected(shape: dict, pointer: tuple[int, int] | None) -> bool:
    """Check if a shape is selected.

    Args:
        shape (dict): Shape object definition.
        pointer (tuple[int, int]): Coordinates of the pointer.

    Returns:
        bool: True if the shape is selected, otherwise False.
    """

    if pointer is None:
        return False

    return IS_SELECTED[shape["type"]](shape, pointer)
