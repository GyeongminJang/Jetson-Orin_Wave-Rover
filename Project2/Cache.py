from __future__ import annotations

from pathlib import Path
from typing import Sequence
import argparse
import cv2
import datetime
import logging
import numpy as np
import os
import time
import torch
import torchvision
import PIL.Image

from cnn.center_dataset import TEST_TRANSFORMS
from jetcam.csi_camera import CSICamera
from base_ctrl import BaseController
from ultralytics import YOLO

logging.getLogger().setLevel(logging.INFO)

def draw_boxes(image, pred, classes, colors):
    """YOLOv8 탐지 결과 시각화 (높이 정보 추가)"""
    for r in pred:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = round(float(box.conf[0]), 2)
            label = int(box.cls[0])
            height = y2 - y1  # 높이 계산
            color = colors[label].tolist()
            cls_name = classes[label]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"{cls_name} {score} H:{height}px",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

def get_lane_model():
    # 차선 추종용 CNN(AlexNet) 모델 생성
    return torchvision.models.alexnet(num_classes=2, dropout=0.0)

def preprocess(image: PIL.Image.Image, device):
    # 이미지 전처리 함수 (텐서 변환 및 배치 차원 추가)
    return TEST_TRANSFORMS(image).to(device)[None, ...]

class Camera:
    def __init__(
        self,
        sensor_id: int | Sequence[int] = 0,
        width: int = 1280,
        height: int = 720,
        _width: int = 1280,
        _height: int = 720,
        frame_rate: int = 30,
        flip_method: int = 0,
        window_title: str = "Camera",
        save_path: str = "record",
        stream: bool = False,
        save: bool = False,
        log: bool = True,
    ) -> None:
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self._width = _width
        self._height = _height
        self.frame_rate = frame_rate
        self.flip_method = flip_method
        self.window_title = window_title
        self.save_path = Path(save_path)
        self.stream = stream
        self.save = save
        self.log = log
        self.model = None

        if isinstance(sensor_id, int):
            self.sensor_id = [sensor_id]
        elif isinstance(sensor_id, Sequence) and len(sensor_id) > 1:
            raise NotImplementedError("Multiple cameras are not supported yet")

        self.cap = [cv2.VideoCapture(self.gstreamer_pipeline(sensor_id=id),
                                    cv2.CAP_GSTREAMER) for id in self.sensor_id]

        if save:
            os.makedirs(self.save_path, exist_ok=True)
            self.save_path = self.save_path / f'{len(os.listdir(self.save_path)) + 1:06d}'
            os.makedirs(self.save_path, exist_ok=True)
            logging.info(f"Save directory: {self.save_path}")

    def gstreamer_pipeline(self, sensor_id: int) -> str:
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                self.width,
                self.height,
                self.frame_rate,
                self.flip_method,
                self._width,
                self._height,
            )
        )

    def set_model(self, model: YOLO, classes: dict) -> None:
        self.model = model
        self.classes = classes
        self.colors = np.random.randn(len(self.classes), 3)
        self.colors = (self.colors * 255.0).astype(np.uint8)
        self.visualize_pred_fn = lambda img, pred: draw_boxes(img, pred, self.classes, self.colors)

    def run(self, lane_model, lane_device, lane_preprocess, base_ctrl=None) -> None:
        # ==================== PID 파라미터 (중앙선 추종) ====================
        Kp, Kd, Ki = 0.3, 0.3, 0.03

        turn_threshold = 0.7
        integral_threshold = 0.1
        integral_min, integral_max = -0.2 / Ki, 0.2 / Ki
        cruise_speed, slow_speed = 0.45, 0.35
        prev_err, integral = 0.0, 0.0
        last_time = time.time()

        # === 상태 머신 관련 변수 ===
        state = "RUNNING"  # RUNNING, STOPPED, WAITING, OVERTAKE
        last_stop_time = None
        overtake_phase = None  # None, 'LEFT', 'STRAIGHT', 'RIGHT'
        overtake_start_time = None

        # 추월 각 단계별 지속 시간(초) - 환경에 맞게 조정
        overtake_left_duration = 1.0     # 좌측 조향 (차선 이탈)
        overtake_straight_duration = 2.0 # 직진 (추월)
        overtake_right_duration = 1.0    # 우측 조향 (차선 복귀)

        if self.stream:
            cv2.namedWindow(self.window_title)

        if self.cap[0].isOpened():
            try:
                while True:
                    t0 = time.time()
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                    ret, frame = self.cap[0].read()
                    if not ret:
                        print("Camera read failed.")
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = PIL.Image.fromarray(frame_rgb)

                    # ===== 차선 추종 CNN(PID) 기반 중앙선 추종 =====
                    with torch.no_grad():
                        tensor = lane_preprocess(pil_img, lane_device)
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

                    # ===== YOLO 탐지 및 height 판별 =====
                    max_height = 0
                    if self.model is not None:
                        pred = list(self.model(frame, stream=True))
                        for r in pred:
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                height = y2 - y1
                                if height > max_height:
                                    max_height = height
                        print(f"Max height: {max_height}px")
                        self.visualize_pred_fn(frame, pred)

                    # ===== 상태 머신 기반 주행 제어 =====
                    if state == "RUNNING":
                        if max_height >= 350:
                            # 앞차가 가까워지면 정지로 전환
                            state = "STOPPED"
                            last_stop_time = now
                            overtake_phase = None
                            if base_ctrl is not None:
                                base_ctrl.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
                            print("[STOP] Object height >= 350px detected. Rover stopped.")
                        else:
                            # PID 기반 정상 주행
                            if base_ctrl is not None:
                                L = float(np.clip(throttle + steering, -1.0, 1.0))
                                R = float(np.clip(throttle - steering, -1.0, 1.0))
                                base_ctrl.base_json_ctrl({"T": 1, "L": L, "R": R})

                    elif state == "STOPPED":
                        if max_height < 350:
                            # 앞차가 움직이면 바로 RUNNING 복귀
                            state = "RUNNING"
                            if base_ctrl is not None:
                                L = float(np.clip(throttle + steering, -1.0, 1.0))
                                R = float(np.clip(throttle - steering, -1.0, 1.0))
                                base_ctrl.base_json_ctrl({"T": 1, "L": L, "R": R})
                            print("[RESUME] Front object moved. Resume driving.")
                        elif now - last_stop_time >= 5.0:
                            # 5초 경과 시 추월로 전환
                            state = "OVERTAKE"
                            overtake_phase = "LEFT"
                            overtake_start_time = now
                            print("[OVERTAKE] 5 seconds elapsed. Initiating overtake (LEFT phase).")
                        else:
                            # 5초 대기 중
                            state = "WAITING"

                    elif state == "WAITING":
                        if max_height < 350:
                            # 앞차가 움직이면 RUNNING 복귀
                            state = "RUNNING"
                            if base_ctrl is not None:
                                L = float(np.clip(throttle + steering, -1.0, 1.0))
                                R = float(np.clip(throttle - steering, -1.0, 1.0))
                                base_ctrl.base_json_ctrl({"T": 1, "L": L, "R": R})
                            print("[RESUME] Front object moved. Resume driving.")
                        elif now - last_stop_time >= 5.0:
                            # 5초 경과 시 추월로 전환
                            state = "OVERTAKE"
                            overtake_phase = "LEFT"
                            overtake_start_time = now
                            print("[OVERTAKE] 5 seconds elapsed. Initiating overtake (LEFT phase).")
                        else:
                            # 계속 정지
                            if base_ctrl is not None:
                                base_ctrl.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})

                    elif state == "OVERTAKE":
                        # 3단계 추월: LEFT(차선 이탈) → STRAIGHT(추월) → RIGHT(복귀)
                        phase_time = now - overtake_start_time
                        if overtake_phase == "LEFT":
                            if phase_time < overtake_left_duration:
                                # 좌측 조향
                                steering = 0.55
                                L = float(np.clip(slow_speed + steering, -1.0, 1.0))
                                R = float(np.clip(slow_speed - steering, -1.0, 1.0))
                                if base_ctrl is not None:
                                    base_ctrl.base_json_ctrl({"T": 1, "L": L, "R": R})
                            else:
                                overtake_phase = "STRAIGHT"
                                overtake_start_time = now
                                print("[OVERTAKE] LEFT phase complete. STRAIGHT phase start.")
                        elif overtake_phase == "STRAIGHT":
                            if phase_time < overtake_straight_duration:
                                # 직진
                                steering = 0.0
                                L = float(np.clip(slow_speed + steering, -1.0, 1.0))
                                R = float(np.clip(slow_speed - steering, -1.0, 1.0))
                                if base_ctrl is not None:
                                    base_ctrl.base_json_ctrl({"T": 1, "L": L, "R": R})
                            else:
                                overtake_phase = "RIGHT"
                                overtake_start_time = now
                                print("[OVERTAKE] STRAIGHT phase complete. RIGHT phase start.")
                        elif overtake_phase == "RIGHT":
                            if phase_time < overtake_right_duration:
                                # 우측 조향(차선 복귀)
                                steering = -0.5
                                L = float(np.clip(slow_speed + steering, -1.0, 1.0))
                                R = float(np.clip(slow_speed - steering, -1.0, 1.0))
                                if base_ctrl is not None:
                                    base_ctrl.base_json_ctrl({"T": 1, "L": L, "R": R})
                            else:
                                # 추월 완료, RUNNING 복귀
                                state = "RUNNING"
                                overtake_phase = None
                                overtake_start_time = None
                                print("[OVERTAKE] RIGHT phase complete. Resume driving.")

                    # ===== 기타 시각화/저장/종료 =====
                    if self.save:
                        cv2.imwrite(str(self.save_path / f"{timestamp}.jpg"), frame)
                    if self.log:
                        print(f"FPS: {1 / (time.time() - t0):.2f}")
                    if self.stream:
                        cv2.imshow(self.window_title, frame)
                        if cv2.waitKey(1) == ord('q'):
                            break

            except Exception as e:
                print(e)
            finally:
                self.cap[0].release()
                cv2.destroyAllWindows()
                if base_ctrl is not None:
                    base_ctrl.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
                print("Motors stopped.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_id', type=int, default=0, help='Camera ID')
    parser.add_argument('--window_title', type=str, default='Camera', help='OpenCV window title')
    parser.add_argument('--save_path', type=str, default='record', help='Image save path')
    parser.add_argument('--save', action='store_true', help='Save frames to save_path')
    parser.add_argument('--stream', action='store_true', help='Show livestream')
    parser.add_argument('--log', action='store_true', help='Print FPS')
    parser.add_argument('--yolo_model_file', type=str, default=None, help='YOLO model file')
    parser.add_argument('--lane_model_file', type=str, default='road_following_model.pth', help='Lane model file')
    parser.add_argument('--base_serial', type=str, default='/dev/ttyUSB0', help='WaveRover serial port')
    parser.add_argument('--baudrate', type=int, default=115200, help='Serial baudrate')
    args = parser.parse_args()

    # 차선 추종 모델 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lane_model = get_lane_model().to(device)
    lane_model.load_state_dict(torch.load(args.lane_model_file, map_location=device))
    lane_model.eval()

    # Wave Rover 모터 제어기 준비
    try:
        base = BaseController(args.base_serial, args.baudrate)
    except Exception as e:
        print(f"BaseController init failed: {e}")
        base = None

    # 카메라 준비
    cam = Camera(
        sensor_id=args.sensor_id,
        window_title=args.window_title,
        save_path=args.save_path,
        save=args.save,
        stream=args.stream,
        log=args.log)

    # YOLO 모델 준비 (선택)
    if args.yolo_model_file:
        classes = YOLO(args.yolo_model_file, task='detect').names
        model = YOLO(args.yolo_model_file, task='detect')
        cam.set_model(model, classes)

    # 메인 루프 실행 (차선 추종 + YOLO 시각화 + height 기반 정지 + 추월)
    cam.run(lane_model, device, preprocess, base_ctrl=base)
