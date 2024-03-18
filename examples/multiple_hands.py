import cv2
import numpy as np

from touchless.hands import HandType, Hand, HandTrackingProvider, GestureProvider
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

    # Create providers
    hand_tracking_provider: HandTrackingProvider = HandTrackingProvider()
    gesture_provider: GestureProvider = GestureProvider()

    while cam.is_active:

        frame: np.ndarray | None = cam.read()

        if frame is not None:

            # Create hand instances
            right_hand = Hand(type=HandType.RIGHT)
            left_hand = Hand(type=HandType.LEFT)

            hands: list[Hand] = [right_hand, left_hand]

            hand_tracking_provider.update(frame, right_hand, left_hand)
            gesture_provider.detect_gestures(right_hand, left_hand)

            cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 100), color=(255, 255, 255), thickness=-1)

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
