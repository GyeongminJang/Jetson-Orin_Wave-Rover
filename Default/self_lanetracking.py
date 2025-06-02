from __future__ import annotations

import os
import torch
import torchvision
import cv2
import time
import PIL.Image
from cnn.center_dataset import TEST_TRANSFORMS
from jetcam.csi_camera import CSICamera
from base_ctrl import BaseController  # Wave Rover 제어용

# 칼만 필터 파라미터 (필요시 사용)
A = 1
H = 1
Q = 0.95
R = 2.38
x = 0
P = 2

def kalman_filter(z_meas, x_esti, P):
    """Kalman Filter Algorithm for One Variable."""
    x_pred = A * x_esti
    P_pred = A * P * A + Q
    K = P_pred * H / (H * P_pred * H + R)
    x_esti = x_pred + K * (z_meas - H * x_pred)
    P = P_pred - K * H * P_pred
    return x_esti, P

def get_lane_model():
    lane_model = torchvision.models.alexnet(num_classes=2, dropout=0.0)
    return lane_model

def preprocess(image: PIL.Image):
    device = torch.device('cuda')
    image = TEST_TRANSFORMS(image).to(device)
    return image[None, ...]

# Wave Rover용 BaseController 인스턴스 생성
base = BaseController('/dev/ttyUSB0', 115200)

device = torch.device('cuda')
lane_model = get_lane_model()
lane_model.load_state_dict(torch.load('road_following_model.pth'))
lane_model = lane_model.to(device)

camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)

# PID 제어 파라미터
Kp = 1.0
Kd = 0.2  #0.2~0.25범위내에서 설정하는 것이 적당
Ki = 0.1
turn_threshold = 0.7
integral_threshold = 0.2
integral_range = (-0.4/Ki, 0.4/Ki)
cruise_speed = 0.45    #0.45~0.5 범위내에서 설정
slow_speed = 0.35

# L, R 모터 속도 범위 (예시: -1.0 ~ 1.0, 필요시 수정)
LR_RANGE = (-1.0, 1.0)

execution = True
now = time.time()
previous_err = 0.0
integral = 0.0

print("Ready... (Wave Rover)")

try:
    while execution:
        image = camera.read()
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(color_coverted)
        with torch.no_grad():
            image_tensor = preprocess(image=pil_image)
            output = lane_model(image_tensor).detach().cpu().numpy()
        err, y = output[0]
        # # 칼만 필터 적용 (필요시 주석 해제)
        # x, P = kalman_filter(err, x, P)
        # err = x
        time_interval = max(0.05, time.time() - now)
        now = time.time()

        # Anti-windup
        if abs(err) > integral_threshold:
            integral = 0
        elif previous_err * err < 0:
            integral = 0
        else:
            integral += err * time_interval
            integral = max(integral_range[0], min(integral_range[1], integral))

        steering = float(Kp*err + Kd*(err-previous_err)/time_interval + Ki*integral)

        previous_err = err

        # throttle 결정
        if abs(steering) < turn_threshold:
            throttle = cruise_speed
        else:
            throttle = slow_speed


        # steering, throttle → L, R 변환
        L = throttle + steering
        R = throttle - steering
        # L, R 값 범위 제한
        L = max(LR_RANGE[0], min(LR_RANGE[1], L))
        R = max(LR_RANGE[0], min(LR_RANGE[1], R))

        # Wave Rover에 명령 전송
        base.base_json_ctrl({"T": 1, "L": L, "R": R})

        # 상태 출력 (디버깅용)
        print(f"time: {round(now)}, steering: {steering:.3f}, throttle: {throttle:.3f}, L: {L:.3f}, R: {R:.3f}")

except KeyboardInterrupt:
    print("KeyboardInterrupt - Stopping...")

finally:
    camera.release()
    # 정지 명령
    base.base_json_ctrl({"T": 1, "L": 0, "R": 0})
    print("terminated")
