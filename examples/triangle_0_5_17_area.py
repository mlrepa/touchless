import argparse

import cv2
import numpy as np

from touchless.camera import Camera
from touchless.hands import HandType, Hand, HandTrackingProvider
from touchless.utils.landmarks import HandLandmarkPoints
from touchless.utils.math_utils import heron_area_by_points, dist_from_triangle_0_5_17_to_camera


def main(width: int = 640, height = 480) -> None:

    # Start video capture and set defaults
    cam: Camera = Camera(width=width, height=height)

    FRAME_WIDTH: int = cam.resolution.width
    FRAME_HEIGHT: int = cam.resolution.height
    FRAME_SIZE: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT)
    print(f"Frame size = {FRAME_SIZE}")
    print(f"FPS = {cam.fps}")

    CV_WIN_NAME: str = "window"
    cv2.namedWindow(CV_WIN_NAME, cv2.WINDOW_GUI_NORMAL)
    
    # Create providers
    hand_tracking_provider: HandTrackingProvider = HandTrackingProvider()

    while cam.is_active:

        frame = cam.read()
        
        if frame is not None:

            right_hand: Hand = Hand(type=HandType.RIGHT)
            hand_tracking_provider.update(frame, right_hand)
            keypoints: HandLandmarkPoints | None = right_hand.data.keypoints

            text_params: dict = {
            "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
            "fontScale": 0.8,
                "color": (0, 100, 0),
                "thickness": 2
            }
            
            if keypoints:
                # TODO: implement drawing keypoints (landmarks) without of MediaPipe function usage
                # draw_landmarks(frame, hands_results.multi_hand_landmarks[0], HAND_CONNECTIONS)
                p1: tuple[int, int] = (int(keypoints.wrist.x * FRAME_WIDTH), int(keypoints.wrist.y * FRAME_HEIGHT))
                p2: tuple[int, int] = (int(keypoints.index_mcp.x * FRAME_WIDTH), int(keypoints.index_mcp.y * FRAME_HEIGHT))
                p3: tuple[int, int] = (int(keypoints.pinky_mcp.x * FRAME_WIDTH), int(keypoints.pinky_mcp.y * FRAME_HEIGHT))

                triangle_area: float = heron_area_by_points(p1, p2, p3)
                cam_dist: float = dist_from_triangle_0_5_17_to_camera(triangle_area)

                # TODO: discuss: translate each point of triangle by the angle around z-axis
                # Note: it's assume that a palm is parallel (as well as possible) to camera
                v1 = (
                    keypoints.index_mcp.x - keypoints.wrist.x,
                    keypoints.index_mcp.y - keypoints.wrist.y,
                    keypoints.index_mcp.z - keypoints.wrist.z
                )
                v2 = (
                    keypoints.pinky_mcp.x - keypoints.wrist.x,
                    keypoints.pinky_mcp.y - keypoints.wrist.y,
                    keypoints.pinky_mcp.z - keypoints.wrist.z
                )

                normal_vector: np.ndarray = np.cross(v1, v2)
                normal_vector /= np.linalg.norm(normal_vector)
                normal_angle: float = np.degrees(np.arccos(np.dot(normal_vector, np.array([0, 0, 1]))))
                normal_angle = round(normal_angle, 3)

                cv2.putText(
                    frame,
                    text=f"triangle area = {triangle_area}",
                    org=(10, 50),
                    **text_params
                )

                cv2.putText(
                    frame,
                    text=f"distance to camera = {cam_dist}",
                    org=(10, 100),
                    **text_params
                )

                cv2.putText(
                    frame,
                    text=f"angle between z-axis = {normal_angle}",
                    org=(10, 150),
                    **text_params
                )

            cv2.imshow(CV_WIN_NAME, frame)

    print(cam.release_status)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--width", dest="width", type=int, default=640)
    args_parser.add_argument("--height", dest="height", type=int, default=480)
    args = args_parser.parse_args()

    main(width=args.width, height=args.height)
