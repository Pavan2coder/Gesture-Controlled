import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller, Key
import math
import time
import winsound
import threading

keyboard = Controller()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)



# ===== KEYBOARD LAYOUT =====
keys = [
    ["1","2","3","4","5","6","7","8","9","0"],
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["Z","X","C","V","B","N","M"],
    ["SPACE","BACK"]
]

key_w, key_h = 65, 65
click_delay = 0.5
last_click = 0

# 🔊 play sound (non-blocking)
def play_click_sound():
    threading.Thread(target=lambda: winsound.PlaySound("click.wav", winsound.SND_FILENAME | winsound.SND_ASYNC), daemon=True).start()

def draw_keyboard(img):
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x = 40 + j * key_w
            y = 50 + i * key_h
            cv2.rectangle(img, (x,y), (x+key_w, y+key_h), (255,0,0), 2)
            cv2.putText(img, key, (x+10, y+45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_keyboard(img)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark

            ix = int(lm[8].x * img.shape[1])
            iy = int(lm[8].y * img.shape[0])

            tx = int(lm[4].x * img.shape[1])
            ty = int(lm[4].y * img.shape[0])

            cv2.circle(img, (ix, iy), 8, (0,255,0), -1)

            distance = math.hypot(tx - ix, ty - iy)

            for i, row in enumerate(keys):
                for j, key in enumerate(row):
                    x = 40 + j * key_w
                    y = 50 + i * key_h

                    if x < ix < x+key_w and y < iy < y+key_h:
                        cv2.rectangle(img, (x,y), (x+key_w, y+key_h), (0,255,0), -1)
                        cv2.putText(img, key, (x+10, y+45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

                        if distance < 35 and time.time() - last_click > click_delay:
                            if key == "SPACE":
                                keyboard.press(Key.space)
                            elif key == "BACK":
                                keyboard.press(Key.backspace)
                            else:
                                keyboard.press(key.lower())

                            last_click = time.time()
                            play_click_sound()

            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Keyboard", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
