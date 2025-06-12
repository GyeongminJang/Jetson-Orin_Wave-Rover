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
yolo_model = YOLO('best.pt')  # YOLO 객체 탐지 모델 로드
yolo_names = yolo_model.names  # 클래스 이름 리스트

def get_lane_model():
    # 차선 추종용 CNN(AlexNet) 모델 생성
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

print("Ready... (Wave Rover with Vehicle Size Detection)")

execution, prev_err, integral = True, 0.0, 0.0
last_time = time.time()

# ========== 차량 높이 기반 정지 변수 ==========
stop_start_time = None
stop_duration = 5  # 5초 정지

try:
    while execution:
        frame_bgr = camera.read()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(frame_rgb)

        # ===== YOLO 추론 (Vehicle 탐지 및 크기 출력) =====
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
                width = x2 - x1
                height = y2 - y1
                print(f"[Vehicle] Detected - Width: {width:.1f} px, Height: {height:.1f} px")
                vehicle_found = True
                vehicle_height = height
                break  # 첫 번째 Vehicle만 처리

        if not vehicle_found:
            print("[Vehicle] Not detected")

        # ===== 차량 높이 조건: height >= 450px이면 5초간 정지 =====
        if vehicle_height is not None and vehicle_height >= 450:
            if stop_start_time is None:
                print("[Vehicle] Height >= 450px detected. Stopping for 5 seconds.")
                stop_start_time = time.time()
            elapsed = time.time() - stop_start_time
            if elapsed < stop_duration:
                # 5초 동안 정지
                base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
                continue  # 아래 제어 로직 건너뜀
            else:
                print("[Vehicle] 5 seconds stop complete. Resuming driving.")
                stop_start_time = None
        else:
            stop_start_time = None

        # ===== 차선 추종 PID 제어 =====
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
