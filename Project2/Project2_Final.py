from __future__ import annotations

import time
import numpy as np
import cv2
import torch
import torchvision
import PIL.Image

from cnn.center_dataset import TEST_TRANSFORMS
from jetcam.csi_camera import CSICamera
from base_ctrl import BaseController
from ultralytics import YOLO

# ========== 모델 및 하드웨어 초기화 ==========
yolo_model = YOLO('best.pt')
yolo_names = yolo_model.names

def get_lane_model():
    return torchvision.models.alexnet(num_classes=2, dropout=0.0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lane_model = get_lane_model().to(device)
lane_model.load_state_dict(torch.load('road_following_model.pth', map_location=device))
lane_model.eval()

def preprocess(image: PIL.Image.Image):
    return TEST_TRANSFORMS(image).to(device)[None, ...]

base = BaseController('/dev/ttyUSB0', 115200)
camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)

# ========== PID 파라미터 ==========
Kp, Kd, Ki = 1.0, 0.15, 0.095
turn_threshold = 0.7
integral_threshold = 0.1
integral_min, integral_max = -0.4 / Ki, 0.4 / Ki
cruise_speed, slow_speed = 0.5, 0.4

print("Ready... (Wave Rover Full Obstacle Avoidance Flow)")

execution, prev_err, integral = True, 0.0, 0.0
last_time = time.time()

# ========== 상태 변수 ==========
stop_start_time = None
stop_duration = 5
stop_done_once = False
vehicle_detected_recently = False
vehicle_already_avoided = False

avoid_mode = False
avoid_start_time = None
avoid_duration = 1.5

return_turn_mode = False
return_turn_start_time = None  # 좌회전 추가 단계

recovery_mode = False
recovery_start_time = None
recovery_duration = 1.25

try:
    while execution:
        frame_bgr = camera.read()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(frame_rgb)

        results = yolo_model(frame_bgr, stream=False, verbose=False)
        boxes = results[0].boxes

        vehicle_found = False
        vehicle_height = None

        for box in boxes:
            cls = int(box.cls[0].item())
            label = yolo_names[cls]
            if label == "Vehicle":
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                height = y2 - y1
                print(f"[Vehicle] Detected - Height: {height:.1f}px")
                vehicle_found = True
                vehicle_height = height
                break

        current_time = time.time()

        # ===== 회피 모드 =====
        if avoid_mode:
            if current_time - avoid_start_time < avoid_duration:
                print("[Avoidance] Executing curved avoidance...")
                base.base_json_ctrl({"T": 1, "L": 0.475, "R": 0.1})  # 우회전
                continue
            else:
                print("[Avoidance] Done. Starting left turn to return.")
                avoid_mode = False
                return_turn_mode = True
                return_turn_start_time = current_time
                continue

        # ===== 좌회전 복귀 모드 =====
        if return_turn_mode:
            if current_time - return_turn_start_time < avoid_duration:
                print("[Return Turn] Executing left turn to return...")
                base.base_json_ctrl({"T": 1, "L": 0.1, "R": 0.48})  # 우회전의 반대
                continue
            else:
                print("[Return Turn] Done. Entering recovery mode.")
                return_turn_mode = False
                recovery_mode = True
                recovery_start_time = current_time
                continue

        # ===== 복구 모드 =====
        if recovery_mode:
            if current_time - recovery_start_time < recovery_duration:
                print("[Recovery] PID lane following during recovery...")
                with torch.no_grad():
                    tensor = preprocess(pil_img)
                    out = lane_model(tensor).cpu().numpy()[0]
                    err = float(out[0])
                    now = time.time()
                    dt = max(1e-3, now - last_time)
                    last_time = now

                    if abs(err) > integral_threshold or prev_err * err < 0:
                        integral = 0.0
                    else:
                        integral = np.clip(integral + err * dt, integral_min, integral_max)

                    steering = Kp * err + Kd * (err - prev_err) / dt + Ki * integral
                    steering *= 0.7  # 70%로 감쇠
                    prev_err = err

                    throttle = slow_speed
                    steering = np.clip(steering, -0.3, 0.3)
                    L = float(np.clip(throttle + steering, -1.0, 1.0))
                    R = float(np.clip(throttle - steering, -1.0, 1.0))
                    base.base_json_ctrl({"T": 1, "L": L, "R": R})
                continue
            else:
                print("[Recovery] Recovery complete. Returning to normal driving.")
                recovery_mode = False
                stop_start_time = None
                stop_done_once = False
                vehicle_detected_recently = False
                vehicle_already_avoided = True
                continue

        # ===== 차량 정지 조건 확인 =====
        if vehicle_height is not None and vehicle_height >= 250:
            if vehicle_already_avoided:
                print("[Vehicle] Already avoided. Ignoring.")
            else:
                vehicle_detected_recently = True
                if not stop_done_once:
                    if stop_start_time is None:
                        print("[Vehicle] Stop initiated.")
                        stop_start_time = current_time
                    elif current_time - stop_start_time < stop_duration:
                        print(f"[Vehicle] Waiting... ({current_time - stop_start_time:.1f}s)")
                        base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
                        continue
                    else:
                        print("[Vehicle] Max wait reached. Starting avoidance.")
                        stop_done_once = True
                        avoid_mode = True
                        avoid_start_time = current_time
                        continue
        else:
            if not stop_done_once:
                if vehicle_detected_recently:
                    print("[Vehicle] Detection lost temporarily — maintaining stop state.")
                else:
                    stop_start_time = None
            vehicle_detected_recently = False
            vehicle_already_avoided = False

        # ===== 정상 차선 추종 =====
        with torch.no_grad():
            tensor = preprocess(pil_img)
            out = lane_model(tensor).cpu().numpy()[0]
            err = float(out[0])
            now = time.time()
            dt = max(1e-3, now - last_time)
            last_time = now

            if abs(err) > integral_threshold or prev_err * err < 0:
                integral = 0.0
            else:
                integral = np.clip(integral + err * dt, integral_min, integral_max)

            steering = Kp * err + Kd * (err - prev_err) / dt + Ki * integral
            prev_err = err

            throttle = cruise_speed if abs(steering) < turn_threshold else slow_speed
            L = float(np.clip(throttle + steering, -1.0, 1.0))
            R = float(np.clip(throttle - steering, -1.0, 1.0))
            base.base_json_ctrl({"T": 1, "L": L, "R": R})

except KeyboardInterrupt:
    print("KeyboardInterrupt - Stopping...")

finally:
    camera.release()
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
    print("Terminated")
