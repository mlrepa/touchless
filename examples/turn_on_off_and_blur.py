import cv2
import numpy as np

from touchless.camera import Camera
from touchless.hands import HandsProvider


def main():

    VIDEO_ON: bool = True
    BLUR_ON: bool = False

    # Start video capture and set defaults
    cam: Camera = Camera()
    cv2.namedWindow("main_window", cv2.WINDOW_GUI_NORMAL)
    
    # Define required gestures to detect
    required_gestures: list[str] = [
        "hand_up", "hand_down",
        "two_fingers_index_middle", "three_fingers_index_middle_ring"
    ]
    # Create hands provider
    hands_provider: HandsProvider = HandsProvider(right_hand_gestures=required_gestures)

    while cam.is_active:

        frame = cam.read()

        if frame is not None:

            hands_provider.update(frame, right_hand_gestures=True)

            # Create frame with a black img
            stopped_img: np.ndarray = np.zeros([100, 100, 3], dtype=np.uint8)

            for gesture in hands_provider.right_hand.gestures:

                # Check if hand is inverted or down
                if gesture.name == "hand_down" and gesture.data.is_detected:
                    VIDEO_ON = False

                # Check if three signal is given
                if gesture.name == "three_fingers_index_middle_ring" and gesture.data.is_detected:
                    BLUR_ON = True

                # Check if two signal is given
                if gesture.name == "two_fingers_index_middle" and gesture.data.is_detected:
                    BLUR_ON = False

                # Check if hand is up and continue the capture
                if gesture.name == "hand_up" and gesture.data.is_detected:
                    VIDEO_ON = True

            # If the hand was down
            if not VIDEO_ON:
                frame = stopped_img

            # If blur is on blur the image
            if BLUR_ON:
                frame = cv2.GaussianBlur(frame, (15, 15), 0)

            cv2.imshow("main_window", frame)

    print(cam.release_status)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    main()
