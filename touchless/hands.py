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

    def update(self, frame: np.ndarray) -> dict[HandType, HandTrackingData]:
        """Updates hand tracking data.

        Args:
            frame (np.ndarray): The frame to process.
        
        Returns:
            dict[HandType, HandTrackingData]: Dictionary of tracking data for each hand type.
        """
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results: NamedTuple = self._hands_processor.process(img)
        
        multi_hand_landmarks = hands_results.multi_hand_landmarks
        multi_handedness = hands_results.multi_handedness

        hands_tracking_data: dict[HandType, HandTrackingData] = {
            HandType.RIGHT: HandTrackingData(),
            HandType.LEFT: HandTrackingData()
        }

        if multi_hand_landmarks is None:
            return hands_tracking_data
        
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

            hands_tracking_data[HandType(hand_type_name)] = HandTrackingData(
                is_hand_detected=True,
                hand_confidence=hand_confidence,
                keypoints=keypoints,
                timestamp_ns=timestamp_ns
            )

        return hands_tracking_data


class GestureProvider:
    """A class for detecting gestures."""

    NAME: str = "rules_defined_gesture_provider"
    GESTURE_CONFIDENCE: float = 0.5
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
    
    def detect_gestures(self, hand: Hand) -> list[HandGesture]:
        """Detects gestures from hand landmarks.

        Args:
            hand (Hand): Hand which to detect gestures for.

        Returns:
            list[HandGesture]: List of the hand gestures.
        """

        keypoints: HandLandmarkPoints = hand.data.keypoints

        return self._detect_gestures(keypoints, hand.required_gestures)

    @property
    def name(self) -> str:
        """Gets the name of the gesture provider.

        Returns:
            str: The name of the gesture provider.
        """
        return self.NAME

    def _detect_gestures(self, keypoints: HandLandmarkPoints | None, required_gestures: list[str] | None) -> list[HandGesture]:
        """Detects specific gestures from hand landmarks.

        Args:
            keypoints (HandLandmarkPoints | None): The hand landmarks.
            required_gestures (list[str] | None): List of required gestures, or None for all.

        Returns:
            list[HandGesture]: A list of detected hand gestures.
        """

        detected_gestures: list[HandGesture] = []

        if keypoints is None:
            return detected_gestures

        gestures_space: dict[str, Callable] = self.GESTURES

        if required_gestures is not None:
            gestures_space = {k: v for k, v in gestures_space.items() if k in required_gestures}

        for gesture_name, gesture_callable in gestures_space.items():
            gesture: HandGesture = HandGesture(
                name=gesture_name,
                data=GestureTrackingData(
                    is_detected=gesture_callable(keypoints),
                    gesture_confidence=self.GESTURE_CONFIDENCE,
                    timestamp_ns=int(time.time() * 1000)
                ),
                provider=self.name
            )
            detected_gestures.append(gesture)

        return detected_gestures


class HandsProvider:

    def __init__(self,
            right_hand_gestures: list[str] | None = None,
            left_hand_gestures: list[str] | None = None
        ) -> None:

        self._right_hand_gestures: list[str] | None = right_hand_gestures
        self._left_hand_gestures: list[str] | None = left_hand_gestures

        self._hand_tracking_provider = HandTrackingProvider()
        self._gesture_provider: GestureProvider = GestureProvider()

    def update(self,
            frame: np.ndarray,
            right_hand_gestures: bool = False,
            left_hand_gestures: bool = False
        ) -> None:

        self._right_hand: Hand = Hand(type=HandType.RIGHT, required_gestures=self._right_hand_gestures)
        self._left_hand: Hand = Hand(type=HandType.LEFT, required_gestures=self._left_hand_gestures)
        
        hands_tracking_data: dict[HandType, HandTrackingData] = self._hand_tracking_provider.update(frame)
        self._right_hand.data = hands_tracking_data[self._right_hand.type]
        self._left_hand.data = hands_tracking_data[self._left_hand.type]

        if right_hand_gestures:
            self._right_hand.gestures = self._gesture_provider.detect_gestures(self._right_hand)

        if left_hand_gestures:
            self._left_hand.gestures = self._gesture_provider.detect_gestures(self._left_hand)

    @property
    def right_hand(self) -> Hand:
        return self._right_hand
    
    @property
    def left_hand(self) -> Hand:
        return self._left_hand
