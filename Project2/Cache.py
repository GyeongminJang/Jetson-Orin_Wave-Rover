# 임시 저장용

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
            height = y2 - y1
            color = colors[label].tolist()
            cls_name = classes[label]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image,
                        f"{cls_name} {score} H:{height}px",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA)

def get_lane_model():
    return torchvision.models.alexnet(num_classes=2, dropout=0.0)

def preprocess(image: PIL.Image.Image, device):
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
        Kp, Kd, Ki = 1.0, 0.15, 0.095
        turn_threshold = 0.7
        integral_threshold = 0.1
        integral_min, integral_max = -0.4 / Ki, 0.4 / Ki
        cruise_speed, slow_speed = 0.45, 0.18  # 서행 속도는 더 낮게
        prev_err, integral = 0.0, 0.0
        last_time = time.time()

        # ==== 상황 제어 변수 ====
        HEIGHT_STOP_THRESH = 500
        HEIGHT_DIFF_THRESH = 30  # height 변화량이 이 값 이하이면 "크게 변하지 않음"
        STOP_DURATION = 5.0  # 정지 유지 시간(초)
        stop_mode = False
        stop_start_time = 0.0
        prev_max_height = 0
        slow_mode = False
        reverse_mode = False
        prev_vehicle_y2 = None
        reverse_count = 0
        REVERSE_COUNT_THRESH = 3  # 몇 프레임 연속으로 y2가 감소해야 후진으로 인식

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

                    # ===== YOLO 탐지 및 상황 판단 =====
                    max_height = 0
                    max_width = 0
                    vehicle_y2 = None
                    if self.model is not None:
                        pred = list(self.model(frame, stream=True))
                        for r in pred:
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                height = y2 - y1
                                width = x2 - x1
                                print(f"[HEIGHT] {height}px [WIDTH] {width}px")
                                if height > max_height:
                                    max_height = height
                                    max_width = width
                                    vehicle_y2 = y2  # 가장 큰 height 차량의 y2 사용
                        self.visualize_pred_fn(frame, pred)
                    else:
                        vehicle_y2 = None

                    # ====== 후진 감지 ======
                    if prev_vehicle_y2 is not None and vehicle_y2 is not None:
                        if vehicle_y2 < prev_vehicle_y2 - 2:  # y2가 작아지면 bbox가 위로 이동(차량이 멀어짐/후진)
                            reverse_count += 1
                        else:
                            reverse_count = 0
                        if reverse_count >= REVERSE_COUNT_THRESH:
                            reverse_mode = True
                        else:
                            reverse_mode = False
                    prev_vehicle_y2 = vehicle_y2

                    # ====== 정지/서행/주행 상태 결정 ======
                    # 1. height가 임계값 이상이면 정지
                    if not stop_mode and max_height >= HEIGHT_STOP_THRESH:
                        stop_mode = True
                        stop_start_time = now
                        prev_max_height = max_height
                        print("[STOP] Object height >= threshold. Rover stopped.")

                    # 2. 정지 상태
                    if stop_mode:
                        # 5초간 정지
                        if now - stop_start_time < STOP_DURATION:
                            throttle = 0.0
                            L = R = 0.0
                            if base_ctrl is not None:
                                base_ctrl.base_json_ctrl({"T": 1, "L": L, "R": R})
                            print(f"[STOP] {now - stop_start_time:.1f}/{STOP_DURATION}s")
                        else:
                            # height 변화가 크지 않으면 서행
                            if abs(max_height - prev_max_height) < HEIGHT_DIFF_THRESH:
                                slow_mode = True
                                stop_mode = False
                                print("[SLOW] Height changed little after stop, switching to slow mode.")
                            else:
                                # height가 충분히 줄었으면 정상 주행 재개
                                stop_mode = False
                                print("[RESUME] Height decreased, resume normal driving.")
                        # 화면 표시, 저장, 로그 등은 아래에서 계속
                    # 3. 서행 모드 (정지 후 height 변화 적거나, 후진 감지)
                    elif slow_mode or reverse_mode:
                        throttle = slow_speed
                        L = float(np.clip(throttle + steering, -1.0, 1.0))
                        R = float(np.clip(throttle - steering, -1.0, 1.0))
                        if base_ctrl is not None:
                            base_ctrl.base_json_ctrl({"T": 1, "L": L, "R": R})
                        print("[SLOW] Slow driving mode (after stop or reverse detected).")
                        # slow_mode는 height가 다시 정상 범위로 돌아오면 해제
                        if max_height < HEIGHT_STOP_THRESH - HEIGHT_DIFF_THRESH and not reverse_mode:
                            slow_mode = False
                            print("[RESUME] Height now normal, resume normal driving.")
                    # 4. 기본: PID 차선 추종 주행
                    else:
                        throttle = cruise_speed if abs(steering) < turn_threshold else slow_speed
                        L = float(np.clip(throttle + steering, -1.0, 1.0))
                        R = float(np.clip(throttle - steering, -1.0, 1.0))
                        if base_ctrl is not None:
                            base_ctrl.base_json_ctrl({"T": 1, "L": L, "R": R})

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lane_model = get_lane_model().to(device)
    lane_model.load_state_dict(torch.load(args.lane_model_file, map_location=device))
    lane_model.eval()

    try:
        base = BaseController(args.base_serial, args.baudrate)
    except Exception as e:
        print(f"BaseController init failed: {e}")
        base = None

    cam = Camera(
        sensor_id=args.sensor_id,
        window_title=args.window_title,
        save_path=args.save_path,
        save=args.save,
        stream=args.stream,
        log=args.log)

    if args.yolo_model_file:
        classes = YOLO(args.yolo_model_file, task='detect').names
        model = YOLO(args.yolo_model_file, task='detect')
        cam.set_model(model, classes)

    cam.run(lane_model, device, preprocess, base_ctrl=base)
