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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU/CPU 자동 선택
lane_model = get_lane_model().to(device)  # 차선 추종 모델 할당
lane_model.load_state_dict(torch.load('road_following_model.pth', map_location=device))  # 가중치 로드
lane_model.eval()  # 평가 모드로 전환

def preprocess(image: PIL.Image.Image):
    # 이미지 전처리 함수 (텐서 변환 및 배치 차원 추가)
    return TEST_TRANSFORMS(image).to(device)[None, ...]

base = BaseController('/dev/ttyUSB0', 115200)  # Wave Rover 모터 제어기 시리얼 연결
camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)  # CSI 카메라 초기화

# ========== PID 파라미터 ==========
Kp, Kd, Ki = 1.0, 0.15, 0.095  # PID 제어 게인
turn_threshold = 0.7  # 조향 임계값(급커브 감속)
integral_threshold = 0.1  # 적분 항 임계값
integral_min, integral_max = -0.4 / Ki, 0.4 / Ki  # 적분 포화 범위
cruise_speed, slow_speed = 0.5, 0.4  # 순항/감속 속도

print("Ready... (Wave Rover with Vehicle Size Detection)")
execution, prev_err, integral = True, 0.0, 0.0
last_time = time.time()

try:
    while execution:
        frame_bgr = camera.read()  # 카메라 프레임 획득 (BGR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # RGB로 변환
        pil_img = PIL.Image.fromarray(frame_rgb)  # PIL 이미지 변환

        # ===== YOLO 추론 (Vehicle 탐지 및 크기 출력) =====
        results = yolo_model(frame_bgr, stream=False, verbose=False)  # YOLO 객체 탐지 실행
        boxes = results[0].boxes

        vehicle_found = False
        for box in boxes:
            cls = int(box.cls[0].item())
            label = yolo_names[cls]
            if label == "Vehicle":
                xyxy = box.xyxy[0].cpu().numpy()  # 바운딩 박스 좌표 추출
                x1, y1, x2, y2 = xyxy
                width = x2 - x1  # 가로 길이(픽셀)
                height = y2 - y1  # 세로 길이(픽셀)
                print(f"[Vehicle] Detected - Width: {width:.1f} px, Height: {height:.1f} px")
                vehicle_found = True
                # 여러 Vehicle이 있을 경우 모두 출력하려면 continue 사용
        if not vehicle_found:
            print("[Vehicle] Not detected")

        # ===== 차선 추종 PID 제어 =====
        with torch.no_grad():
            tensor = preprocess(pil_img)  # 이미지 전처리 및 텐서 변환
            out = lane_model(tensor).cpu().numpy()[0]  # 차선 추종 모델 추론
        err = float(out[0])  # 차선 중심 오차

        now = time.time()
        dt = max(1e-3, now - last_time)
        last_time = now

        # PID 적분 항 처리
        if abs(err) > integral_threshold or prev_err * err < 0:
            integral = 0.0
        else:
            integral = np.clip(integral + err * dt, integral_min, integral_max)

        steering = Kp * err + Kd * (err - prev_err) / dt + Ki * integral  # PID 조향값 계산
        prev_err = err

        throttle = cruise_speed if abs(steering) < turn_threshold else slow_speed  # 급커브 시 감속

        # ===== 모터 제어 =====
        L = float(np.clip(throttle + steering, -1.0, 1.0))  # 좌측 모터 속도
        R = float(np.clip(throttle - steering, -1.0, 1.0))  # 우측 모터 속도
        base.base_json_ctrl({"T": 1, "L": L, "R": R})  # 모터 제어 명령 전송

except KeyboardInterrupt:
    print("KeyboardInterrupt - Stopping...")

finally:
    camera.release()
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})  # 정지 명령
    print("Terminated")
