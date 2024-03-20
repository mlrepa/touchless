import cv2
import numpy as np

from touchless.hands import Hand, HandsProvider
from touchless.camera import Camera


if __name__ == "__main__":

    cam: Camera = Camera()

    FRAME_WIDTH: int = cam.resolution.width
    FRAME_HEIGHT: int = cam.resolution.height
    FRAME_SIZE: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT)
    print(f"Frame size = {FRAME_SIZE}")
    print(f"FPS = {cam.fps}")

    CV_WIN_NAME: str = "window"
    cv2.namedWindow(CV_WIN_NAME, cv2.WINDOW_GUI_NORMAL)

    # Create hands provider
    hands_provider: HandsProvider = HandsProvider()

    while cam.is_active:

        frame: np.ndarray | None = cam.read()

        if frame is not None:

            cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 100), color=(255, 255, 255), thickness=-1)
            hands_provider.update(frame, right_hand_gestures=True, left_hand_gestures=True)
            hands: list[Hand] = [hands_provider.right_hand, hands_provider.left_hand]

            for i, hand in enumerate(hands):
                
                # Filter gestures: take just detected
                detected_gestures: str = ", ".join([
                    gesture.name
                    for gesture in hand.gestures
                    if gesture.data.is_detected
                ])

                cv2.putText(
                    frame,
                    f"{hand.type} hand detected gestures: {detected_gestures}",
                    (5, 10 + i * 20),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 0),
                    thickness=1
                )
            cv2.imshow(CV_WIN_NAME, frame)
    
    cv2.destroyWindow(CV_WIN_NAME)
