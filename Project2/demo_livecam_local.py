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
            cv2.putText(image, 
                        f"{cls_name} {score} H:{height}px", 
                        (x1, max(0, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        1, 
                        cv2.LINE_AA)

class Camera:
    def __init__(
        self,
        sensor_id: int | Sequence[int] = 0,
        width: int = 1280,
        height: int = 720,
        _width: int = 640,
        _height: int = 360,
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

    def run(self) -> None:
        if self.stream:
            cv2.namedWindow(self.window_title)

        if self.cap[0].isOpened():
            try:
                while True:
                    t0 = time.time()
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                    _, frame = self.cap[0].read()

                    if self.model is not None:
                        # 모델 추론 및 결과 변환
                        pred = list(self.model(frame, stream=True))  # 제너레이터 -> 리스트 변환
                        
                        # 높이 정보 터미널 출력
                        for r in pred:
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                height = y2 - y1
                                print(f"[HEIGHT] {height}px")
                        
                        # 시각화
                        self.visualize_pred_fn(frame, pred)

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

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--sensor_id', type=int, default=0, help='Camera ID')
    args.add_argument('--window_title', type=str, default='Camera', help='OpenCV window title')
    args.add_argument('--save_path', type=str, default='record', help='Image save path')
    args.add_argument('--save', action='store_true', help='Save frames to save_path')
    args.add_argument('--stream', action='store_true', help='Show livestream')
    args.add_argument('--log', action='store_true', help='Print FPS')
    args.add_argument('--yolo_model_file', type=str, default=None, help='YOLO model file')
    args = args.parse_args()

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

    cam.run()
