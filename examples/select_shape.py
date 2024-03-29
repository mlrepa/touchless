import cv2

from touchless.camera import Camera
from touchless.hands import HandsProvider
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
    
    # Create hands provider
    hands_provider: HandsProvider = HandsProvider()

    while cam.is_active:

        frame = cam.read()

        if frame is not None:

            hands_provider.update(frame)
            keypoints: HandLandmarkPoints | None = hands_provider.right_hand.data.keypoints
            pointer: tuple[int, int] | None = get_pointer(keypoints, FRAME_SIZE)

            draw_pointer(frame, pointer)
            render_shapes(frame, SHAPES, pointer)

            cv2.imshow(CV_WIN_NAME, frame)

    print(cam.release_status)    
    cv2.destroyWindow(CV_WIN_NAME)


if __name__ == "__main__":

    main()
