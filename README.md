# 🏋️ AI-Powered Exercise Pose Correction System

A real-time posture evaluation and repetition counter for fitness exercises using **MediaPipe**, **OpenCV**, and **Python**. The system analyzes body movements and provides feedback for form correction and rep counting using geometric analysis.

## 📽️ Demo Video & Outputs

🎥 Check all recorded exercise feedback videos and screenshots in the [Google Drive Output Folder](https://drive.google.com/drive/u/0/folders/1y_KiJy0xEEv1PO_oFlzLSNOz7F5B7nVw)

---

## ✅ Key Features

- 🔍 Detects pose landmarks using MediaPipe
- 🔁 Repetition counter with up/down movement logic
- 🧠 Real-time feedback for form corrections
- 🔧 Supports both **recorded video input** and **live webcam**
- 🖼️ Feedback box overlay showing:
  - Exercise name
  - Rep counts (left/right)
  - Elbow/shoulder/knee angles
  - Encouragement or correction messages

---

## 🧪 Supported Exercises

1. 💪 **Bicep Curls**
2. 🤸 **Lateral Raises**
3. 🧎 **Push-Ups**
4. 🦵 **Lunges**

---

## 📂 Folder Structure
```
├── main.py # Real-time / video input runner
├── exercises/
│ ├── bicep_curl.py 
│ ├── lateral_raise.py 
│ ├── lunge.py
│ ├── pushup.py 
│ └── init.py
├── utils/
│ ├── pose_extractor.py
│ ├── feedback_drawer.py 
│ └── pose_utils.py
├── requirements.txt

```
---
## ⚙️ How It Works

This system provides real-time exercise feedback using computer vision and pose estimation. Here’s an overview of the main components:

- **Pose Detection**: Utilizes MediaPipe to identify and track key body landmarks.
- **Angle Calculation**: Computes critical joint angles (elbow, knee, shoulder) for each frame.
- **Form Evaluation**: Assesses user posture, determines movement phase (e.g., up, down), and provides corrections or encouragement.
- **Counter**: Increases repetition count only when full-motion and proper transitions are detected.
- **Visual Feedback**: A dynamic feedback box overlays the video, presenting:
  - ✅ Pose name
  - 🔁 Current repetition counts
  - 💬 Real-time suggestions (like "Raise higher", "Good job", "Fix your posture")
  - 📐 Live joint angles

---

### ✨ Sample Feedback Messages

- “Left rep counted”
- “Pull right arm more”
- “Great form!”
- “Standing straight. Prepare to lunge”
- “Fix your form”

---
## 📽️ Output Preview

Watch all processed exercise feedback videos here:  
[All Output Videos – Google Drive](https://drive.google.com/drive/u/0/folders/1y_KiJy0xEEv1PO_oFlzLSNOz7F5B7nVw)
  

