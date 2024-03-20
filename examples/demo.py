import cv2

from touchless.camera import Camera
from touchless.hands import HandsProvider
import touchless.gestures as gestures
from touchless.utils.landmarks import HandLandmarkPoints


def main():

    GESTURES_LIST_HEADER_POSITION: tuple[int, int] = (5, 80)
    GESTURE_NAME_POSITION_DELTA: int = 20

    cam: Camera = Camera()

    FRAME_WIDTH: int = cam.resolution.width
    FRAME_HEIGHT: int = cam.resolution.height
    FRAME_SIZE: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT)
    print(f"Frame size = {FRAME_SIZE}")

    # TODO: make a CLI param
    CHANGE_MAGNITUDE_ROI_RATIO: float = 2.5
    CHANGE_MAGNITUDE_ROI: list[tuple[int, int]] = [
        (int(FRAME_WIDTH - FRAME_WIDTH / CHANGE_MAGNITUDE_ROI_RATIO), int(FRAME_HEIGHT - FRAME_HEIGHT / CHANGE_MAGNITUDE_ROI_RATIO)),
        (FRAME_WIDTH, FRAME_HEIGHT)
    ]
    
    CV_WIN_NAME: str = "window"
    cv2.namedWindow(CV_WIN_NAME, cv2.WINDOW_GUI_NORMAL)
    
    # Create hands provider
    hands_provider: HandsProvider = HandsProvider()

    while cam.is_active:

        frame = cam.read()

        if frame is not None:

            hands_provider.update(frame, right_hand_gestures=True)

            # ROI (rectangle) where change magnitude (volume) is detected
            cv2.rectangle(frame, pt1=CHANGE_MAGNITUDE_ROI[0], pt2=CHANGE_MAGNITUDE_ROI[1], color=(0, 0, 255), thickness=2)
            cv2.putText(
                frame, "Change magnitude (volume)", (CHANGE_MAGNITUDE_ROI[0][0] + 10, CHANGE_MAGNITUDE_ROI[0][1] + 20),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.4,
                color=(0, 100, 0),
                thickness=1
            )

            keypoints: HandLandmarkPoints | None = hands_provider.right_hand.data.keypoints

            cv2.putText(
                frame,
                f"Detected gestures:",
                GESTURES_LIST_HEADER_POSITION,
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5,
                color=(0, 100, 0),
                thickness=2
            )

            if keypoints:
                # TODO: implement drawing keypoints (landmarks) without of MediaPipe function usage
                # draw_landmarks(frame, hands_results.multi_hand_landmarks[0], HAND_CONNECTIONS)
                gest_text_y: int = GESTURES_LIST_HEADER_POSITION[1] + GESTURE_NAME_POSITION_DELTA

                magnitude_ok, magnitude_value = gestures.change_magnitude(
                    points=keypoints,
                    image=frame,
                    img_size=(FRAME_WIDTH, FRAME_HEIGHT),
                    roi=CHANGE_MAGNITUDE_ROI
                )
                
                if magnitude_ok:
                    cv2.putText(
                        frame,
                        f"- volume = {magnitude_value}",
                        (10, gest_text_y),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.5,
                        color=(0, 100, 0),
                        thickness=2
                    )

                else:
                    for gesture in hands_provider.right_hand.gestures:
                        if gesture.data.is_detected:
                            # Display gesture name
                            cv2.putText(
                                frame,
                                f"- {gesture.name}",
                                (10, gest_text_y),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                fontScale=0.5,
                                color=(0, 100, 0),
                                thickness=2
                            )
                            gest_text_y += GESTURE_NAME_POSITION_DELTA

            cv2.imshow(CV_WIN_NAME, frame)

    print(cam.release_status)
    cv2.destroyWindow(CV_WIN_NAME)


if __name__ == "__main__":

    main()
