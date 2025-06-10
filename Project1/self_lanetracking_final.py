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
yolo_model = YOLO('best.pt')
yolo_intersection_model = YOLO('best_intersection.pt')
yolo_straight_model=YOLO('best_straight.pt')
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

# ==================== PID 파라미터 ====================
Kp, Kd, Ki = 1.0, 0.15, 0.095
turn_threshold = 0.7
integral_threshold = 0.1
integral_min, integral_max = -0.4 / Ki, 0.4 / Ki
cruise_speed, slow_speed = 0.45, 0.35

# ==================== 상태 머신 ====================
SLOW_DURATION, STOP_DURATION = 5.0, 3.0
SLOW_SIGN_SPEED = 0.125
slow_mode = stop_mode = False
slow_handled = stop_handled = False
slow_start_time = stop_start_time = 0.0
turn_mode = None  # 'left' or 'right'
turn_handled = False
LEFT_TURN_DURATION = 1.6
RIGHT_TURN_DURATION = 1.5
turn_start_time = 0.0
post_turn_straightening = False
post_turn_start_time = 0.0
POST_TURN_STRAIGHTEN_DURATION = 1.0
STRAIGHT_DURATION = 1.5
straight_mode = False
straight_handled = False
straight_start_time = 0.0

# ==================== 장애물 회피 ====================
avoidance_mode = False
avoidance_timer = 0
AVOID_DURATION = 20
STRAIGHT_DURATION = 20
STRAIGHT_ALIGN_DURATION = 15
RECOVERY_DURATION = 25
RECOVERY_ALIGN_DURATION = 25
AVOID_IGNORE_DURATION = 30
avoidance_ignore_counter = 0
avoidance_state = None
avoiding_dir = 1
recovery_dir = -1

def handle_task_logic(frame: np.ndarray, throttle: float, steering: float) -> tuple[float, float]:
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
            vehicles.append(xyxy)
        elif label == 'Red_Light':
            red_lights.append((area, xyxy))
        elif label in ['STOP', 'SLOW', 'Left', 'Right']:
            signs.append((label, area))

    print(f"[YOLO] {', '.join(f'{v}x{k}' for k, v in label_counts.items() if v > 0) or 'None'}")

    # ========== STRAIGHT 모델 인식 확인 ==========
    straight_results = yolo_straight_model(frame, stream=False, verbose=False)
    for box in straight_results[0].boxes:
        cls = int(box.cls[0].item())
        label_straight = yolo_straight_model.names[cls]
        if label_straight == 'Straight':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            # print(f"[STRAIGHT YOLO] Detected: {label_straight}, Area: {area}")
            # if not straight_handled:
            #     straight_mode = True
            #     straight_handled = True
            #     straight_start_time = now
            #     print("[DETECT] STRAIGHT sign → ENTER STRAIGHT MODE")
            #     return cruise_speed, 0.0

    if stop_mode:
        if now - stop_start_time < STOP_DURATION:
            print(f"[STATE] STOP MODE ({now - stop_start_time:.1f}/{STOP_DURATION}s)")
            return 0.0, 0.0
        else:
            stop_mode = False
            print("[STATE] STOP MODE → NORMAL")

    if slow_mode:
        if now - slow_start_time < SLOW_DURATION:
            print(f"[STATE] SLOW MODE ({now - slow_start_time:.1f}/{SLOW_DURATION}s)")
            return SLOW_SIGN_SPEED, steering
        else:
            slow_mode = False
            print("[STATE] SLOW MODE → NORMAL")

    if turn_mode:
        if turn_mode == 'right' and now - turn_start_time < LEFT_TURN_DURATION:
            print(f"[STATE] TURN MODE ({turn_mode})")
            return 0.4, 0.65 
        elif turn_mode == 'left' and now - turn_start_time < RIGHT_TURN_DURATION:
            print(f"[STATE] TURN MODE ({turn_mode})")
            return 0.3, -0.6 
        else:
            turn_mode = None
            turn_handled = False
            post_turn_straightening = True
            post_turn_start_time = now
            print("[STATE] TURN MODE → POST-TURN STRAIGHTENING")

    if post_turn_straightening:
        if now - post_turn_start_time < POST_TURN_STRAIGHTEN_DURATION:
            print("[STATE] POST-TURN STRAIGHTENING")
            return cruise_speed, 0.0
        else:
            post_turn_straightening = False
            print("[STATE] POST-TURN STRAIGHTENING → NORMAL")

    if avoidance_ignore_counter > 0:
        avoidance_ignore_counter -= 1
        return throttle, steering

    
    results_intersection = yolo_intersection_model(frame, stream=False, verbose=False)
    intersection_area = 0
    for box in results_intersection[0].boxes:
        inter_label = results_intersection[0].names[int(box.cls[0])]
        if inter_label == 'Intersection':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > intersection_area:
                intersection_area = area

    # if intersection_area > 0:
    #     print(f"[INTERSECTION] Detected with area: {intersection_area}")
    # else:
    #     print("[INTERSECTION] Not detected")

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

    elif vehicles and not avoidance_mode:
        bbox = vehicles[0]
        ##x_center = (bbox[0] + bbox[2]) / 2
        ##frame_center = frame.shape[1] / 2
        ## avoiding_dir = -1 if x_center < frame_center else 1
        avoiding_dir=-1
        recovery_dir= 1
        avoidance_mode = True
        avoidance_timer = 0
        avoidance_state = 'avoiding'
        print(f"[AVOID] Vehicle detected → start avoidance (dir={avoiding_dir})")
        return 0.0, 0.0

    if red_lights:
        red_lights.sort(reverse=True, key=lambda x: x[0])
        nearest_area, _ = red_lights[0]
        if nearest_area >= 400:
            print("[TRAFFIC] RED LIGHT → STOP")
            return 0.0, 0.0

    return throttle, steering

# ==================== 메인 루프 ====================
print("Ready... (Wave Rover)")
execution, prev_err, integral = True, 0.0, 0.0
last_time = time.time()

try:
    while execution:
        frame_bgr = camera.read()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(frame_rgb)

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

        throttle, steering = handle_task_logic(frame_bgr, throttle, steering)

        L = float(np.clip(throttle + steering, -1.0, 1.0))
        R = float(np.clip(throttle - steering, -1.0, 1.0))
        base.base_json_ctrl({"T": 1, "L": L, "R": R})

except KeyboardInterrupt:
    print("KeyboardInterrupt - Stopping...")

finally:
    camera.release()
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
    print("Terminated")
