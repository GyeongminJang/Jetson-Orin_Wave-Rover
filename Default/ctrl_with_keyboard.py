from base_ctrl import BaseController
import time
from pynput import keyboard
import threading

# === Constants ===
MAX_STEER = 1.0
MAX_SPEED = 0.5
STEP_STEER = 0.5
STEP_SPEED = 0.05

base = BaseController('/dev/ttyUSB0', 115200)

steering = 0
speed = 0.0

pressed_keys = set()
last_update_time = 0
update_interval = 0.1

def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({"T": 1, "L": L, "R": R})
    threading.Thread(target=worker).start()

def on_press(key):
    try:
        pressed_keys.add(key.char) # 문자 입력
    except:
        pass

def on_release(key):
    try:
        pressed_keys.discard(key.char)
        if key.char == 'p':
            return False  # 종료
    except:
        pass

def clip(val, max_val):
    return max(min(val, max_val), -max_val)

def update_vehicle_motion(steering, speed):
    steer_val = clip(steering, MAX_STEER)
    speed_val = clip(speed, MAX_STEER)

    base_speed = abs(speed_val)

    left_ratio = 1.0 - steer_val
    right_ratio = 1.0 + steer_val

    if left_ratio < 0:
        left_ratio = 0
    elif right_ratio < 0:
        right_ratio = 0

    L = base_speed * left_ratio
    R = base_speed * right_ratio

    L = clip(L, MAX_SPEED)
    R = clip(R, MAX_SPEED)

    if speed < 0:
        L, R = -L, -R

    send_control_async(-L, -R)
    print(f"[UGV] Speed: {speed_val:.2f}, Steering: {steer_val:.2f} → L: {L:.2f}, R: {R:.2f}")

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    while listener.running:
        now = time.time()

        if (now - last_update_time) >= update_interval:
            if 's' in pressed_keys:
                speed += STEP_SPEED
            elif 'w' in pressed_keys:
                speed -= STEP_SPEED
            else:
                speed *= 0.5

            if 'd' in pressed_keys:
                steering -= STEP_STEER
            elif 'a' in pressed_keys:
                steering += STEP_STEER
            else:
                steering *= 0.5

            update_vehicle_motion(steering, speed)
            last_update_time = now

except KeyboardInterrupt:
    print("\n Quit")
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
