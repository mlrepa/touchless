"""

Main idea:
- try to implement dynamic click gesture depending on time delay/frame series length.

Hypothesis:
- collect and analyze:
    - landmarks coordinates (for the fingers involved in the gesture), its relative landmarks positions
    - relative landmarks positions for the wrist (hand may move simultaneously with clicking)
    - statistics on frame series
    - timestamps
- research, experiment and adjust optimal time delay or/and frames series length which detect dynamic click on

Additional thoughts & ideas:
- detect click on arbitrary frames series, maybe detect how many times it was and get frames subseries with clicks
"""

import os
from pathlib import Path
import time

import cv2

from touchless.camera import Camera
from touchless.hands import HandsProvider
from touchless.utils.landmarks import (
    HandLandmarkPoints,
    Point,
    get_pointer
)
from touchless.utils.shapes import draw_pointer


def get_pointer3d_normalized(points: HandLandmarkPoints | None) -> tuple[float, float, float] | None:
    """Get the normalized 3D coordinates of the index finger tip.

    Args:
        points (HandLandmarkPoints | None): Landmarks for one hand or None if no hand detected.

    Returns:
        tuple[float, float, float] | None: 
            The normalized 3D coordinates of the index finger tip if a hand landmarks is not None,
            otherwise None.
    """
    
    if points:
        index_tip: Point = points.index_tip
        pointer: tuple[float, float, float] = (index_tip.x, index_tip.y, index_tip.z)
        return pointer
    return None


def main():

    LOGS_DIR: str = Path("logs")
    os.makedirs(LOGS_DIR, exist_ok=True)

    cam: Camera = Camera()

    FRAME_WIDTH: int = cam.resolution.width
    FRAME_HEIGHT: int = cam.resolution.height
    FRAME_SIZE: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT)
    print(f"Frame size = {FRAME_SIZE}")

    CV_WIN_NAME: str = "window"
    cv2.namedWindow(CV_WIN_NAME, cv2.WINDOW_GUI_NORMAL)

    # Create hands provider
    hands_provider: HandsProvider = HandsProvider()

    check_every: int = 20
    frame_cnt: int = 0

    coord_deltas_log = open(LOGS_DIR / f"coord_deltas_{check_every}.txt", "w")
    coord_log = open(LOGS_DIR / f"coords_{int(time.time())}.txt", "w")

    while cam.is_active:

        frame = cam.read()

        if frame is not None:

            hands_provider.update(frame)
            keypoints: HandLandmarkPoints | None = hands_provider.right_hand.data.keypoints
            pointer: tuple[int, int] | None = get_pointer(keypoints, FRAME_SIZE)
            pointer3d: tuple[float, float, float] | None = get_pointer3d_normalized(keypoints)
            draw_pointer(frame, pointer)

            x = y = z = None

            if pointer is not None:
                x, y, z = list(map(lambda coord: round(coord, 3), pointer3d))
                coord_log.write(f"x = {x} \t y = {y} \t ts = {time.time()}\n")
                coord_log.flush()
                if frame_cnt == 0:
                    start_x, start_y, start_z = x, y, z
                    start_time = time.time()
                frame_cnt += 1
                if frame_cnt == check_every:
                    delta_x = round(x - start_x, 3)
                    delta_y = round(y - start_y, 3)
                    delta_z = round(z - start_z, 3)
                    time_delta: float = round(time.time() - start_time, 3)
                    coord_deltas_log.write(f"dx={delta_x} \t dy={delta_y} \t dz={delta_z} \t dtime = {time_delta}\n")
                    coord_deltas_log.flush()
                    frame_cnt = 0
            else:
                frame_cnt = 0

            # Display pointer three coords values (normalized)
            cv2.putText(
                frame,
                f"Pointer = {{{x}, {y}, {z}}}",
                (100, FRAME_HEIGHT - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 255, 0),
                thickness=2
            )
            cv2.imshow(CV_WIN_NAME, frame)

    print(cam.release_status)
    cv2.destroyWindow(CV_WIN_NAME)
    coord_deltas_log.close()
    coord_log.close()


if __name__ == "__main__":

    main()
