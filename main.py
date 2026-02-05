import cv2
import time
import pyautogui
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

SCROLL_SPEED = 300
SCROLL_DELAY = 1
CAM_WIDTH, CAM_HEIGHT = 640, 480


def detect_gesture(landmarks, handedness):
    fingers = []

    tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    for tip in tips:
        if landmarks.landmark[tip].y < landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

    if (handedness == "Right" and thumb_tip.x > thumb_ip.x) or \
       (handedness == "Left" and thumb_tip.x < thumb_ip.x):
        fingers.append(1)
    else:
        fingers.append(0)

    if fingers == [1, 1, 1, 1, 1]:
        return "scroll_up"
    elif fingers == [0, 0, 0, 0, 0]:
        return "scroll_down"
    else:
        return "none"


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

last_scroll = 0
p_time = 0

print("Gesture scrolling started. Press 'q' to exit.")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_info in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            handedness = hand_info.classification[0].label
            gesture = detect_gesture(hand_landmarks, handedness)

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            if time.time() - last_scroll > SCROLL_DELAY:
                if gesture == "scroll_up":
                    pyautogui.scroll(SCROLL_SPEED)
                    last_scroll = time.time()
                elif gesture == "scroll_down":
                    pyautogui.scroll(-SCROLL_SPEED)
                    last_scroll = time.time()

    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
    p_time = c_time

    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Gesture Controlled Scrolling", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
