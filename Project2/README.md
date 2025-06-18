# Project2: Intelligent Vehicle Handling and Avoidance

This project implements intelligent vehicle detection, stopping, overtaking (avoidance), and lane recovery for autonomous driving scenarios using Python.

---

## 🛠️ Key Features

- **Vehicle Detection & Stopping:**  
  Detects vehicles ahead using height information and stops the rover if a vehicle is detected within a certain threshold.

- **Automatic Overtaking (Avoidance):**  
  If the detected vehicle does not move within 5 seconds, the system initiates an overtaking maneuver:
  1. **Avoid Mode:** Steers to avoid the stopped vehicle.
  2. **Return Mode:** Aligns parallel to the road after avoidance.
  3. **Recovery Mode:** Fine-tunes position to return to the lane center.

- **Cut-in Handling:**  
  Detects when another vehicle cuts in and follows or stops as needed, then overtakes if necessary.

- **Robust Logic:**  
  Ensures that after one avoidance, the same vehicle is not avoided again, preventing repeated maneuvers.

---

## 📂 File Structure

| Filename                  | Description                                                                                         |
|---------------------------|-----------------------------------------------------------------------------------------------------|
| **Project2_Final.py**     | Main script that integrates vehicle detection, stopping, overtaking (avoidance), and lane recovery. |
| **demo_livecam_local.py** | Script for running real-time live camera demonstrations and testing the system locally.              |
| **vehicle_height.py**     | Module for detecting vehicle height and determining stop/avoidance conditions.                      |

---

## 👥 Team

- Gyeongmin Jang, Junseong Kim, Yunseok Choi
