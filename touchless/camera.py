from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Resolution:
    """A class representing the resolution of an image or video.

    Attributes:
        width (int): The width of the image or video.
        height (int): The height of the image or video.
    """

    width: int
    height: int


class Camera:
    """A class for capturing video using OpenCV."""

    def __init__(
        self,
        ocv_capture: int | str = 0,
        width: int = 640,
        height: int = 480,
        stop_capture_keys: tuple[int, ...] = (27,),
        codec_fourcc: str = "MJPG",
        flip: bool = True
    ) -> None:
        """Initializes the Camera object.

        Args:
            ocv_capture (int | str): Index of the camera or path to video file. Default is 0 (default camera).
            width (int): Width of the captured video frame. Default is 640 pixels.
            height (int): Height of the captured video frame. Default is 480 pixels.
            stop_capture_keys (tuple[int, ...]): Keys to stop the video capture. Default is (27,) for the 'Esc' key.
            codec_fourcc (str): FourCC code representing the codec for video writing. Default is "MJPG".
            flip (bool): Whether to horizontally flip the captured frames. Default is True.
        """
        self._cap = cv2.VideoCapture(ocv_capture)
        self._set_resolution(width, height)
        self._codec_fourcc: str = codec_fourcc
        self._flip: bool = flip
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self._codec_fourcc))  # depends on fourcc available camera
        self._active: bool = self._cap.isOpened()
        self._stop_capture_keys: tuple[int, ...] = stop_capture_keys
        self._release_status: str = ""

    def read(self) -> np.ndarray | None:
        """Reads a frame from the camera.

        Returns:
            np.ndarray | None: The captured frame, or None if there was an error or a stop key was pressed.
        """
        key: int = cv2.waitKey(1)

        if key in self._stop_capture_keys:
            self._release()
            self._release_status = f"Stop on key {key}"
            return None
        
        status, frame = self._cap.read()

        if status:

            if self._flip:
                frame = cv2.flip(frame, 1)
            return frame
        else:
            self._release()
            self._release_status = "Stop on frame read error."
            return None
        
    @property
    def is_active(self) -> bool:
        """Checks if the camera is active.

        Returns:
            bool: True if the camera is active, False otherwise.
        """
        return self._active

    @property
    def resolution(self) -> Resolution:
        """Gets the resolution of the camera.

        Returns:
            Resolution: An object representing the width and height of the camera resolution.
        """
        return self._resolution
    
    @property
    def release_status(self) -> str:
        """Gets the release status of the camera.

        Returns:
            str: A message indicating the release status.
        """
        return self._release_status
    
    @property
    def fps(self) -> int:
        """Gets the frames per second (FPS) of the camera.

        Returns:
            int: The FPS of the camera.
        """
        return int(self._cap.get(cv2.CAP_PROP_FPS))

    def _set_resolution(self, width: int, height) -> None:
        """Sets the resolution of the camera.

        Args:
            width (int): The width to set.
            height (int): The height to set.
        """
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution: Resolution = Resolution(width, height)

    def _release(self) -> None:
        """Releases the camera resources."""
        self._active = False
        self._cap.release()
