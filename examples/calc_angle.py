import argparse

import cv2

from touchless.camera import Camera
from touchless.hands import HandType, Hand, HandTrackingProvider
from touchless.gestures.geometry import vertical_angle
from touchless.utils.landmarks import get_pointer
from touchless.utils.shapes import draw_pointer, draw_angle_vertex, draw_vertical, draw_angle


def main(v_x: int | None = None, v_y: int | None = None):

    cam: Camera = Camera()

    FRAME_WIDTH: int = cam.resolution.width
    FRAME_HEIGHT: int = cam.resolution.height
    FRAME_SIZE: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT)
    print(f"Frame size = {FRAME_SIZE}")

    CV_WIN_NAME: str = "window"
    cv2.namedWindow(CV_WIN_NAME, cv2.WINDOW_GUI_NORMAL)

    v_x = v_x if v_x is not None else FRAME_WIDTH // 2
    v_y = v_y if v_y is not None else 0
    VERTEX: tuple[int, int] = (v_x, v_y)
    
    # Create providers
    hand_tracking_provider: HandTrackingProvider = HandTrackingProvider()

    while cam.is_active:

        frame = cam.read()

        if frame is not None:

            right_hand: Hand = Hand(type=HandType.RIGHT)
            hand_tracking_provider.update(frame, right_hand)

            angle: float | None = None

            draw_angle_vertex(frame, VERTEX)
            draw_vertical(frame, VERTEX, FRAME_HEIGHT)

            pointer: tuple[int, int] | None = get_pointer(right_hand.data.keypoints, FRAME_SIZE)
            draw_pointer(frame, pointer)

            if pointer is not None:

                draw_angle(frame, VERTEX, pointer)
                angle = vertical_angle(VERTEX, pointer, FRAME_HEIGHT)  

            # Display angle value
            cv2.putText(
                frame,
                f"Angle = {angle} degrees",
                (100, FRAME_HEIGHT - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2
            )
            cv2.imshow(CV_WIN_NAME, frame)
    
    print(cam.release_status)
    cv2.destroyWindow(CV_WIN_NAME)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-x", type=int)
    args_parser.add_argument("-y", type=int)
    args = args_parser.parse_args()

    main(v_x=args.x, v_y=args.y)
