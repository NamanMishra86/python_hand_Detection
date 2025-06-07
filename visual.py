import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

# Function to control mouse
def move_mouse(x, y):
    screen_x = screen_width / cap.get(3) * x
    screen_y = screen_height / cap.get(4) * y
    pyautogui.moveTo(screen_x, screen_y)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            for id, lm in enumerate(landmarks):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Index finger tip
                if id == 8:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                    move_mouse(cx, cy)

                # Thumb tip for click
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                    thumb_x, thumb_y = cx, cy

                # If thumb tip and index tip are close, perform click
                if id == 8 and abs(cx - thumb_x) < 30 and abs(cy - thumb_y) < 30:
                    pyautogui.click()

    # Display the output
    cv2.imshow("Hand Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()