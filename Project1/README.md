
# Embedded System Project: Autonomous Lane Following and Task Handling

This project implements an autonomous driving system that enables a robot vehicle to follow a lane and handle various road tasks using computer vision and control algorithms.

---

## 📚 Overview

- **Dataset Creation:**  
  - Generated a dataset of ~27,000 frames from real-time video.
  - Labeled centerline points and trained for 150 epochs (15 hours).
  - Used CVAT for further labeling and trained for 150 more epochs (20 hours).
  - Used `ffmpeg` to convert frames to video for training and evaluation.

- **Full Path Navigation:**  
  - Generated a path centered on the lane's centerline.
  - Applied PID control to minimize deviation from the lane boundaries.

---

## 🚗 Core Features & Tasks

### 1. Lane Following with PID Control

- The robot follows the centerline using a PID controller:
  - **PID Parameters:**  
    - Kp = 1.0, Kd = 0.15, Ki = 0.1
    - Integral and turn thresholds for robust performance
    - Cruise and slow speed modes depending on steering angle

### 2. Traffic Light Detection

- **Red Light:**  
  - The vehicle stops when a red traffic light is detected within a certain area.
- **Otherwise:**  
  - The vehicle continues to move forward.

### 3. SLOW/STOP Sign Handling

- **SLOW Sign:**  
  - Vehicle slows down for 5 seconds, then resumes normal speed.
- **STOP Sign:**  
  - Vehicle stops for 3 seconds, then resumes movement.

### 4. Vehicle Avoidance

- Detects other vehicles and performs a three-stage avoidance maneuver:
  1. **Avoiding:** Steers away from detected vehicle.
  2. **Straight:** Maintains a straight path for a set duration.
  3. **Recovery:** Returns to the original lane center.

### 5. Complex Intersection Handling

- Simultaneously detects traffic lights, direction signs (Left, Right, Straight), and intersections.
- If a red light is detected, the vehicle stops.
- After the light turns off, the vehicle follows the direction sign (left turn, right turn, or straight) at the intersection.

---

## 🛠️ Additional Considerations

- **Lighting Conditions:**  
  - Performance varies significantly with natural light (e.g., sunlight through windows can cause more deviations).
- **Object Detection Limitations:**  
  - Some objects may not be recognized even after labeling and training.
  - The system is designed to use as much contextual information as possible for robust driving.
  - Occasionally, unrelated objects (e.g., empty spaces, chairs, light extinguishers) may be misclassified as vehicles.

---

## 📂 File Structure

| Filename                   | Description                                                                                       |
|----------------------------|---------------------------------------------------------------------------------------------------|
| **Project1_final.py**      | The main Python script that runs the project's core logic, including model inference and control. |
| **best.pt**                | Trained PyTorch model file for general use (e.g. object detection).                               |
| **best_intersection.pt**   | Trained PyTorch model file specialized for intersection scenarios.                                |
| **best_straight.pt**       | Trained PyTorch model file specialized for straight road scenarios.                               |

---

## 👥 Team

- Gyeonngmin Jang, Junseong Kim, Yunseok Choi
