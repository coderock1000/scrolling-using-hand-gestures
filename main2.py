import numpy as np
import time
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands( min_detection_confidence=0.7, min_tracking_confidence=0.7 )

filters = {
    None ,
    "GRAYSCALE",
    "SEPIA",
    "NEGATIVE",
    "BLUR"
}

current_filter = 0

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

last_action_time = 0

debounce_time = 1

def apply_filter(frame, filter_type):
    if filter_type == "GRAYSCALE":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_type == "SEPIA":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        return sepia_filter.astype(np.uint)
    
    elif filter_type == "NEGATIVE":
        return cv2.bitwise_not(frame)
    elif filter_type == "BLUR":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    