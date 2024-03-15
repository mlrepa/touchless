import math
import cv2

from touchless.camera import Camera
from touchless.hands import HandType, Hand, HandTrackingProvider
from touchless.utils.landmarks import HandLandmarkPoints, get_pointer
from touchless.utils.shapes import SHAPES, draw_pointer, render_shapes


def main():

    cam: Camera = Camera()

    FRAME_WIDTH: int = cam.resolution.width
    FRAME_HEIGHT: int = cam.resolution.height
    FRAME_SIZE: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT)
    print(f"Frame size = {FRAME_SIZE}")

    CV_WIN_NAME: str = "window"
    cv2.namedWindow(CV_WIN_NAME, cv2.WINDOW_GUI_NORMAL)

    # Create providers
    hand_tracking_provider: HandTrackingProvider = HandTrackingProvider()

    # Control ROI is a square with a given side
    #####################################################
    # TODO: make CLI params
    # TODO: make CONTROL_ROI dataclass with fields ratio, offset, size, coords (if needed = ?)
    # Ratio of areas between control ROI (square) and full frame
    CONTROL_ROI_RATIO: float = 0.07
    # Offsets of the square by Ox and Oy axis
    CONTROL_ROI_OFFSET_X: int = 100
    CONTROL_ROI_OFFSET_Y: int = 100
    #####################################################
    CONTROL_ROI_SIZE: int = round(math.sqrt(FRAME_WIDTH * FRAME_HEIGHT * CONTROL_ROI_RATIO))
    CONTROL_ROI = [
        ((FRAME_WIDTH - CONTROL_ROI_SIZE) // 2 + CONTROL_ROI_OFFSET_X, (FRAME_HEIGHT - CONTROL_ROI_SIZE) // 2 + CONTROL_ROI_OFFSET_Y),
        ((FRAME_WIDTH + CONTROL_ROI_SIZE) // 2 + CONTROL_ROI_OFFSET_X, (FRAME_HEIGHT + CONTROL_ROI_SIZE) // 2 + CONTROL_ROI_OFFSET_Y)
    ]

    print(f"Control ROI = {CONTROL_ROI}")

    ASPECT_RATIO_X = FRAME_WIDTH / CONTROL_ROI_SIZE
    ASPECT_RATIO_Y = FRAME_HEIGHT / CONTROL_ROI_SIZE

    while cam.is_active:

        frame = cam.read()

        if frame is not None:

            right_hand: Hand = Hand(type=HandType.RIGHT)
            hand_tracking_provider.update(frame, right_hand)
            keypoints: HandLandmarkPoints | None = right_hand.data.keypoints

            cv2.rectangle(
                frame,
                pt1=CONTROL_ROI[0],
                pt2=CONTROL_ROI[1],
                color=(0, 0, 200),
                thickness=2
            )

            pointer: tuple[int, int] | None = get_pointer(keypoints, FRAME_SIZE)
            mapped_pointer: tuple[int, int] | None = None
            if (
                pointer is not None and
                (CONTROL_ROI[0][0] <= pointer[0] <= CONTROL_ROI[1][0]) and
                (CONTROL_ROI[0][1] <= pointer[1] <= CONTROL_ROI[1][1])
            ):
                mapped_pointer = (
                    int((pointer[0] - CONTROL_ROI[0][0]) * ASPECT_RATIO_X),
                    int((pointer[1] - CONTROL_ROI[0][1]) * ASPECT_RATIO_Y)
                )
                draw_pointer(frame, pointer)
                draw_pointer(frame, mapped_pointer)

            render_shapes(frame, SHAPES, mapped_pointer)
            cv2.imshow(CV_WIN_NAME, frame)

    print(cam.release_status)
    cv2.destroyWindow(CV_WIN_NAME)


if __name__ == "__main__":

    main()
