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
yolo_names = yolo_model.names

def get_lane_model():
    return torchvision.models.alexnet(num_classes=2, dropout=0.0)  # 차선 추종용 CNN 모델

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lane_model = get_lane_model().to(device)
lane_model.load_state_dict(torch.load('road_following_model.pth', map_location=device))
lane_model.eval()

def preprocess(image: PIL.Image.Image):
    return TEST_TRANSFORMS(image).to(device)[None, ...]  # 이미지 전처리

def adjust_throttle_for_safety(throttle: float, cruise_speed: float, boxes, yolo_names, depth_map, safe_stop=1.5, safe_slow=2.5) -> float:
    # 차량(Vehicle) 객체에 대한 거리 측정 및 안전 속도 조절 함수
    vehicle_dists = []
    for box in boxes:
        cls = int(box.cls[0].item())
        label = yolo_names[cls]
        if label != 'Vehicle':
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        if center_y >= depth_map.shape[0] or center_x >= depth_map.shape[1]:
            continue
        z_dist = depth_map[center_y, center_x, 2]  # Z축(깊이) 거리 추출
        if 0 < z_dist < 2.0:
            vehicle_dists.append(z_dist)
    
    # 거리 정보 실시간 출력
    if vehicle_dists:
        min_dist = min(vehicle_dists)
        print(f"[DISTANCE] Closest vehicle: {min_dist:.2f}m")  # 가장 가까운 차량과의 거리 출력
        if min_dist < safe_stop:
            print(f"[SAFE] Vehicle very close ({min_dist:.2f}m) → STOP")
            return 0.0  # 너무 가까우면 정지
        elif min_dist < safe_slow:
            scaled_throttle = np.clip((min_dist - safe_stop) / (safe_slow - safe_stop), 0.2, cruise_speed)
            print(f"[SAFE] Vehicle ahead ({min_dist:.2f}m) → SLOW DOWN to {scaled_throttle:.2f}")
            return scaled_throttle  # 가까우면 속도 줄임
    else:
        print("[DISTANCE] No obstacles detected")  # 장애물 없을 때 출력
    
    return throttle  # 안전하면 원래 속도 유지

# ========== 카메라 및 제어 초기화 ==========
base = BaseController('/dev/ttyUSB0', 115200)  # Wave Rover 모터 제어기
camera_left = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)  # 좌측 카메라
camera_right = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30, flip=2)  # 우측 카메라 (좌우 반전)

# Stereo matcher 초기화 (스테레오 영상으로 깊이 추정)
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16*6, blockSize=9)

# Q 행렬 설정 (스테레오 보정용, 실제 환경에 맞게 조정 필요)
focal_length = 800.0
width = 640
height = 360
Q = np.float32([[1, 0, 0, -width/2],
                [0, -1, 0, height/2],
                [0, 0, 0, -focal_length],
                [0, 0, 1, 0]])

# PID 파라미터 설정 (차선 추종용)
Kp, Kd, Ki = 1.0, 0.15, 0.095
turn_threshold = 0.7
integral_threshold = 0.1
integral_min, integral_max = -0.4 / Ki, 0.4 / Ki
cruise_speed, slow_speed = 0.5, 0.4

print("Ready... (Wave Rover with Stereo Depth Safety)")
execution, prev_err, integral = True, 0.0, 0.0
last_time = time.time()

try:
    while execution:
        frameL = camera_left.read()  # 좌측 카메라 프레임 획득
        frameR = camera_right.read()  # 우측 카메라 프레임 획득

        frame_rgb = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(frame_rgb)

        # ===== YOLO 차량 탐지 =====
        results = yolo_model(frameL, stream=False, verbose=False)
        boxes = results[0].boxes

        # ===== Stereo Disparity 및 Depth Map 계산 =====
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0  # 시차 계산
        depth_map = cv2.reprojectImageTo3D(disparity, Q)  # 3D 깊이 맵 생성

        # ===== 차선 추종 PID 제어 =====
        with torch.no_grad():
            tensor = preprocess(pil_img)
            out = lane_model(tensor).cpu().numpy()[0]
        err = float(out[0])

        now = time.time()
        dt = max(1e-3, now - last_time)
        last_time = now

        if abs(err) > integral_threshold or prev_err * err < 0:
            integral = 0.0  # 적분 리셋
        else:
            integral = np.clip(integral + err * dt, integral_min, integral_max)

        steering = Kp * err + Kd * (err - prev_err) / dt + Ki * integral  # PID로 조향값 계산
        prev_err = err

        # ===== 주행 상태에 따라 안전 거리 조절 =====
        if abs(steering) < 0.2:
            safe_stop = 1.5
            safe_slow = 2.5
        elif abs(steering) < 0.5:
            safe_stop = 1.2
            safe_slow = 2.0
        else:
            safe_stop = 1.0
            safe_slow = 1.8

        throttle = cruise_speed if abs(steering) < turn_threshold else slow_speed  # 조향이 크면 감속
        throttle = adjust_throttle_for_safety(throttle, cruise_speed, boxes, yolo_names, depth_map, safe_stop, safe_slow)  # 전방 차량에 따라 속도 조절

        # ===== 모터 제어 =====
        L = float(np.clip(throttle + steering, -1.0, 1.0))  # 좌측 모터
        R = float(np.clip(throttle - steering, -1.0, 1.0))  # 우측 모터
        base.base_json_ctrl({"T": 1, "L": L, "R": R})  # 모터에 명령 전송

except KeyboardInterrupt:
    print("KeyboardInterrupt - Stopping...")

finally:
    camera_left.release()
    camera_right.release()
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})  # 정지 명령
    print("Terminated")
