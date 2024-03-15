from collections.abc import Callable
import enum
import time
from typing import Any, NamedTuple

import cv2
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from mediapipe.python.solutions.hands import HandLandmark, Hands
import numpy as np
from pydantic import BaseModel

from touchless.gestures.clicks import *
from touchless.gestures.fingers import *
from touchless.gestures.hand import *
from touchless.gestures.pinches import *

from touchless.utils.landmarks import Point, HandLandmarkPoints


class HandType(enum.StrEnum):
    """An enumeration representing types of hands."""
    RIGHT = "right"
    LEFT = "left"


class HandTrackingData(BaseModel):
    """A class representing hand tracking data."""
    is_hand_detected: bool = False
    hand_confidence: float | None = None
    keypoints: HandLandmarkPoints | None = None
    timestamp_ns: int | None = None


class GestureTrackingData(BaseModel):
    """A class representing gesture tracking data."""
    is_detected: bool
    gesture_confidence: float
    timestamp_ns: int
    datapoints: Any = None


class HandGesture(BaseModel):
    """A class representing a hand gesture."""
    name: str
    data: GestureTrackingData
    provider: str


class Hand(BaseModel):
    """A class representing a hand."""
    type: HandType
    data: HandTrackingData = HandTrackingData()
    required_gestures: list[str] | None = None
    gestures: list[HandGesture] = []


class HandTrackingProvider:
    """A class for hand tracking."""
    def __init__(self) -> None:
        self._hands_processor = Hands()

    def update(self, frame: np.ndarray, right_hand: Hand, left_hand: Hand | None = None) -> None:
        """Updates hand tracking data.

        Args:
            frame (np.ndarray): The frame to process.
            right_hand (Hand): The right hand object.
            left_hand (Hand | None): The left hand object, if available.
        """
        
        assert right_hand.type == HandType.RIGHT
        if left_hand is not None: assert left_hand.type == HandType.LEFT

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results: NamedTuple = self._hands_processor.process(img)
        
        multi_hand_landmarks = hands_results.multi_hand_landmarks
        multi_handedness = hands_results.multi_handedness

        if multi_hand_landmarks is None:
            return
        
        for i, hand_landmarks in enumerate(multi_hand_landmarks):
            
            points: dict[str, Point] = {}
            landmarks: RepeatedCompositeFieldContainer = hand_landmarks.landmark

            for landmark_field_info in HandLandmarkPoints.model_fields.values():
                landmark_name: str = landmark_field_info.alias
                landmark = landmarks[getattr(HandLandmark, landmark_name).value]
                point: Point = Point(x=landmark.x, y=landmark.y, z=landmark.z)
                points[landmark_name] = point

            keypoints: HandLandmarkPoints = HandLandmarkPoints(**points)
            hand_type_name: str = multi_handedness[i].classification[0].label.lower()
            hand_confidence: float = multi_handedness[i].classification[0].score
            timestamp_ns: int = int(time.time() * 1000)

            if hand_type_name == HandType.RIGHT:
                right_hand.data = HandTrackingData(
                    is_hand_detected=True,
                    hand_confidence=hand_confidence,
                    keypoints=keypoints,
                    timestamp_ns=timestamp_ns
                )

            if left_hand is not None and hand_type_name == HandType.LEFT:
                left_hand.data = HandTrackingData(
                    is_hand_detected=True,
                    hand_confidence=hand_confidence,
                    keypoints=keypoints,
                    timestamp_ns=timestamp_ns
                )


class GestureProvider:
    """A class for detecting gestures."""
    NAME: str = "rules_defined_gesture_provider"
    GESTURES: dict[str, Callable] = {
        "click_index_middle": click_8_12,
        "click_thumb_index": click_4_6,
        "click_index_tip_down_pip": click_6_8,

        "two_fingers_thumb_index": two_fingers_4_8,
        "two_fingers_index_middle": two_fingers_8_12,
        "two_fingers_index_pinky": two_fingers_8_20,
        "three_fingers_thumb_index_middle": three_fingers_4_8_12,
        "three_fingers_index_middle_ring": three_fingers_8_12_16,
        "five_fingers": five_fingers,

        "fist_closed": fist_closed,
        "hand_down": hand_down,
        "hand_up": hand_up,

        "pinch_thumb_index": pinch_4_8,
        "pinch_thumb_middle": pinch_4_12,
        "pinch_thumb_ring": pinch_4_16,
        "pinch_thumb_pinky": pinch_4_20
    }
    
    def detect_gestures(self, right_hand: Hand, left_hand: Hand | None = None) -> None:
        """Detects gestures from hand landmarks.

        Args:
            right_hand (Hand): The right hand object.
            left_hand (Hand | None): The left hand object, if available.
        """
        hands: list[Hand] = [right_hand]
        if left_hand is not None: hands.append(left_hand)

        for hand in hands:
            keypoints: HandLandmarkPoints = hand.data.keypoints
            if keypoints is not None:
                detected_gestures: list[HandGesture] = self._detect_gestures(keypoints, hand.required_gestures)
                hand.gestures = detected_gestures

    @property
    def name(self) -> str:
        """Gets the name of the gesture provider.

        Returns:
            str: The name of the gesture provider.
        """
        return self.NAME

    def _detect_gestures(self, keypoints: HandLandmarkPoints, required_gestures: list[str] | None) -> list[HandGesture]:
        """Detects specific gestures from hand landmarks.

        Args:
            keypoints (HandLandmarkPoints): The hand landmarks.
            required_gestures (list[str] | None): List of required gestures, or None for all.

        Returns:
            list[HandGesture]: A list of detected hand gestures.
        """

        gestures_space: dict[str, Callable] = self.GESTURES

        if required_gestures is not None:
            gestures_space = {k: v for k, v in gestures_space.items() if k in required_gestures}
        
        detected_gestures: list[HandGesture] = []

        for gesture_name, gesture_callable in gestures_space.items():
            gesture: HandGesture = HandGesture(
                name=gesture_name,
                data=GestureTrackingData(
                    is_detected=gesture_callable(keypoints),
                    gesture_confidence=0.5,
                    timestamp_ns=int(time.time() * 1000)
                ),
                provider=self.name
            )
            detected_gestures.append(gesture)

        return detected_gestures
