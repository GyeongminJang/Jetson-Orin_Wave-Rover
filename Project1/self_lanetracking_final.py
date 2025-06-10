from __future__ import annotations
import time
import os
import numpy as np
import cv2
import torch
import torchvision
import PIL.Image
from cnn.center_dataset import TEST_TRANSFORMS
from jetcam.csi_camera import CSICamera
from base_ctrl import BaseController  # Wave Rover 
from ultralytics import YOLO

# ==================== 설정 & 모델 로드 ====================
yolo_model = YOLO('best.pt')  # 일반 객체 탐지용 YOLO 모델 로드
yolo_intersection_model = YOLO('best_intersection.pt')  # 교차로 탐지 YOLO 모델
yolo_straight_model=YOLO('best_straight.pt')  # 직진 표지판 탐지 YOLO 모델
yolo_names = yolo_model.names  # 클래스 이름 목록

def get_lane_model():
    # 차선 추종용 CNN(AlexNet) 모델 생성
    return torchvision.models.alexnet(num_classes=2, dropout=0.0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU/CPU 선택
lane_model = get_lane_model().to(device)  # 차선 추종 모델 할당
lane_model.load_state_dict(torch.load('road_following_model.pth', map_location=device))  # 가중치 로드
lane_model.eval()  # 평가 모드로 전환

def preprocess(image: PIL.Image.Image):
    # 이미지 전처리 함수 (텐서 변환 및 배치 차원 추가)
    return TEST_TRANSFORMS(image).to(device)[None, ...]

base = BaseController('/dev/ttyUSB0', 115200)  # Wave Rover 모터 제어기 시리얼 연결
camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)  # CSI 카메라 초기화

# ==================== PID 파라미터 ====================
Kp, Kd, Ki = 1.0, 0.15, 0.095  # PID 제어 게인
turn_threshold = 0.7  # 조향 임계값(급커브 감속)
integral_threshold = 0.1  # 적분 항 임계값
integral_min, integral_max = -0.4 / Ki, 0.4 / Ki  # 적분 포화 범위
cruise_speed, slow_speed = 0.45, 0.35  # 순항/감속 속도

# ==================== 상태 머신 ====================
SLOW_DURATION, STOP_DURATION = 5.0, 3.0  # SLOW/STOP 표지판 반응 시간
SLOW_SIGN_SPEED = 0.125  # SLOW 표지판 감속 속도
slow_mode = stop_mode = False  # SLOW/STOP 상태 플래그
slow_handled = stop_handled = False  # SLOW/STOP 처리 여부
slow_start_time = stop_start_time = 0.0  # SLOW/STOP 시작 시간
turn_mode = None  # 회전 방향 ('left' 또는 'right')
turn_handled = False  # 회전 처리 여부
LEFT_TURN_DURATION = 1.6
RIGHT_TURN_DURATION = 1.5
turn_start_time = 0.0  # 회전 시작 시간
post_turn_straightening = False  # 회전 후 직진 정렬 플래그
post_turn_start_time = 0.0  # 직진 정렬 시작 시간
POST_TURN_STRAIGHTEN_DURATION = 1.0  # 회전 후 직진 정렬 시간
STRAIGHT_DURATION = 1.5  # 직진 표지판 반응 시간
straight_mode = False
straight_handled = False
straight_start_time = 0.0

# ==================== 장애물 회피 ====================
avoidance_mode = False  # 회피 모드 플래그
avoidance_timer = 0  # 회피 타이머
AVOID_DURATION = 20  # 회피 단계별 지속 시간
STRAIGHT_DURATION = 20
STRAIGHT_ALIGN_DURATION = 15
RECOVERY_DURATION = 25
RECOVERY_ALIGN_DURATION = 25
AVOID_IGNORE_DURATION = 30  # 회피 후 무시 구간
avoidance_ignore_counter = 0
avoidance_state = None  # 회피 상태 ('avoiding', 'straight', 'recovery', 'center')
avoiding_dir = 1  # 회피 방향
recovery_dir = -1  # 복구 방향

def handle_task_logic(frame: np.ndarray, throttle: float, steering: float) -> tuple[float, float]:
    """
    객체 탐지 및 표지판/신호/장애물 등 상황에 따라 주행 상태(스로틀/조향)를 결정하는 함수
    """
    global slow_mode, stop_mode, turn_mode, turn_handled
    global slow_handled, stop_handled
    global slow_start_time, stop_start_time, turn_start_time
    global avoidance_mode, avoidance_timer, avoidance_state
    global avoidance_ignore_counter, avoiding_dir, recovery_dir
    global post_turn_straightening, post_turn_start_time
    global straight_mode, straight_handled, straight_start_time

    now = time.time()

    label_counts = {k: 0 for k in ['Vehicle', 'Red_Light', 'Left', 'Right', 'SLOW', 'STOP']}
    vehicles, red_lights, signs = [], [], []

    # YOLO 모델로 객체 탐지
    results = yolo_model(frame, stream=False, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        cls = int(box.cls[0].item())
        label = yolo_names[cls]
        if label in label_counts:
            label_counts[label] += 1
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        area = (x2 - x1) * (y2 - y1)
        if label == 'Vehicle' and area >= 45000:
            print(f"[YOLO] Vehicle detected (area={area:.1f}) → AVOID")
            vehicles.append(xyxy)  # 큰 차량은 회피 대상
        elif label == 'Red_Light':
            red_lights.append((area, xyxy))
        elif label in ['STOP', 'SLOW', 'Left', 'Right']:
            signs.append((label, area))

    print(f"[YOLO] {', '.join(f'{v}x{k}' for k, v in label_counts.items() if v > 0) or 'None'}")

    # ========== STRAIGHT 표지판 YOLO 모델로 직진 표지판 탐지 ==========
    straight_results = yolo_straight_model(frame, stream=False, verbose=False)
    for box in straight_results[0].boxes:
        cls = int(box.cls[0].item())
        label_straight = yolo_straight_model.names[cls]
        if label_straight == 'Straight':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            # (직진 표지판 처리 코드 주석 처리됨)

    # STOP 표지판 감지 시 정지 상태 진입
    if stop_mode:
        if now - stop_start_time < STOP_DURATION:
            print(f"[STATE] STOP MODE ({now - stop_start_time:.1f}/{STOP_DURATION}s)")
            return 0.0, 0.0  # 정지
        else:
            stop_mode = False
            print("[STATE] STOP MODE → NORMAL")

    # SLOW 표지판 감지 시 감속 상태 진입
    if slow_mode:
        if now - slow_start_time < SLOW_DURATION:
            print(f"[STATE] SLOW MODE ({now - slow_start_time:.1f}/{SLOW_DURATION}s)")
            return SLOW_SIGN_SPEED, steering  # 저속 주행
        else:
            slow_mode = False
            print("[STATE] SLOW MODE → NORMAL")

    # 회전 상태(좌/우회전) 처리
    if turn_mode:
        if turn_mode == 'left' and now - turn_start_time < LEFT_TURN_DURATION:
            print(f"[STATE] TURN MODE ({turn_mode})")
            return 0.4, 0.65  # 좌회전(실제는 오른쪽)
        elif turn_mode == 'right' and now - turn_start_time < RIGHT_TURN_DURATION:
            print(f"[STATE] TURN MODE ({turn_mode})")
            return 0.3, -0.6  # 우회전(실제는 왼쪽)
        else:
            turn_mode = None
            turn_handled = False
            post_turn_straightening = True
            post_turn_start_time = now
            print("[STATE] TURN MODE → POST-TURN STRAIGHTENING")

    # 회전 후 직진 정렬 상태
    if post_turn_straightening:
        if now - post_turn_start_time < POST_TURN_STRAIGHTEN_DURATION:
            print("[STATE] POST-TURN STRAIGHTENING")
            return cruise_speed, 0.0  # 직진
        else:
            post_turn_straightening = False
            print("[STATE] POST-TURN STRAIGHTENING → NORMAL")

    # 회피 후 무시 구간 처리
    if avoidance_ignore_counter > 0:
        avoidance_ignore_counter -= 1
        return throttle, steering

    # 교차로 탐지 (Intersection YOLO)
    results_intersection = yolo_intersection_model(frame, stream=False, verbose=False)
    intersection_area = 0
    for box in results_intersection[0].boxes:
        inter_label = results_intersection[0].names[int(box.cls[0])]
        if inter_label == 'Intersection':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > intersection_area:
                intersection_area = area

    # 표지판(좌/우회전, STOP, SLOW) 및 교차로 상황에 따른 상태 전이
    for label, area in signs:
        if label == 'STOP' and not stop_handled and area >= 8000:
            stop_handled = True
            stop_mode = True
            stop_start_time = now
            print("[DETECT] STOP sign → ENTER STOP MODE")
            return 0.0, 0.0
        elif label == 'SLOW' and not slow_handled and area >= 8000:
            slow_handled = True
            slow_mode = True
            slow_start_time = now
            print("[DETECT] SLOW sign → ENTER SLOW MODE")
            return SLOW_SIGN_SPEED, steering
        elif label == 'Left' and not turn_handled and intersection_area >= 19000 and not red_lights:
            turn_mode = 'right' 
            turn_start_time = now
            turn_handled = True
            print("[ACTION] LEFT TURN INITIATED (intersection detected)")
            return 0.3, 0.6
        elif label == 'Right' and not turn_handled and intersection_area >= 19000 and not red_lights:
            if not straight_handled:
                straight_mode = True
                straight_handled = True
                straight_start_time = now
                print("[DETECT] STRAIGHT sign → ENTER STRAIGHT MODE")
                return cruise_speed, 0.0
            else:
                turn_mode = 'left'
                turn_start_time = now
                turn_handled = True
                print("[ACTION] RIGHT TURN INITIATED (intersection detected)")
                return 0.3, -0.6
        
    # 장애물 회피 상태 머신
    if avoidance_mode:
        if avoidance_state == 'avoiding':
            if avoidance_timer < AVOID_DURATION:
                steering = 0.4 * avoiding_dir
                throttle = 0.415
                print("avoid duration")
            else:
                avoidance_state = 'straight'
                avoidance_timer = 0
        elif avoidance_state == 'straight':
            if avoidance_timer < STRAIGHT_ALIGN_DURATION:
                steering = 0.3 * avoiding_dir
                throttle = 0.5
                print("straight align duration")
            elif avoidance_timer < STRAIGHT_DURATION:
                steering = 0.0
                throttle = 0.25
                print("straight duration")
            else:
                avoidance_state = 'recovery'
                avoidance_timer = 0
        elif avoidance_state == 'recovery':
            if avoidance_timer < RECOVERY_DURATION:
                steering = 0.45 * recovery_dir
                throttle = 0.4
                print("recovery duration")
            elif avoidance_timer >= RECOVERY_DURATION and avoidance_timer < RECOVERY_DURATION + RECOVERY_ALIGN_DURATION:
                steering = 0.3 * recovery_dir
                throttle = 0.45
                print("recovery align duration")
            else:
                avoidance_state = 'center'
                avoidance_timer = 0
                avoidance_ignore_counter = AVOID_IGNORE_DURATION
        elif avoidance_state == 'center':
            avoidance_mode = False
            avoidance_state = None
            avoidance_timer = 0
            return cruise_speed, 0.0

        avoidance_timer += 1
        return throttle, steering

    # 장애물(차량) 감지 시 회피 상태 진입
    elif vehicles and not avoidance_mode:
        bbox = vehicles[0]
        # avoiding_dir, recovery_dir 방향 설정(예시: 항상 왼쪽 회피)
        avoiding_dir=-1
        recovery_dir= 1
        avoidance_mode = True
        avoidance_timer = 0
        avoidance_state = 'avoiding'
        print(f"[AVOID] Vehicle detected → start avoidance (dir={avoiding_dir})")
        return 0.0, 0.0

    # 신호등(빨간불) 감지 시 정지
    if red_lights:
        red_lights.sort(reverse=True, key=lambda x: x[0])
        nearest_area, _ = red_lights[0]
        if nearest_area >= 400:
            print("[TRAFFIC] RED LIGHT → STOP")
            return 0.0, 0.0

    # 기본(차선 추종) 상태
    return throttle, steering

# ==================== 메인 루프 ====================
print("Ready... (Wave Rover)")
execution, prev_err, integral = True, 0.0, 0.0
last_time = time.time()

try:
    while execution:
        frame_bgr = camera.read()  # 카메라 프레임 획득
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(frame_rgb)

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

        throttle, steering = handle_task_logic(frame_bgr, throttle, steering)  # 상황별 상태 머신 처리

        L = float(np.clip(throttle + steering, -1.0, 1.0))  # 좌측 모터 속도
        R = float(np.clip(throttle - steering, -1.0, 1.0))  # 우측 모터 속도
        base.base_json_ctrl({"T": 1, "L": L, "R": R})  # 모터 제어 명령 전송

except KeyboardInterrupt:
    print("KeyboardInterrupt - Stopping...")

finally:
    camera.release()
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})  # 정지 명령
    print("Terminated")
