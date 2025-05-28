# 🧍‍♂️ Real-Time Human Pose Classification using MoveNet Lightning

This project leverages **TensorFlow Lite MoveNet Lightning** for real-time human pose estimation and a custom-trained **MLP classifier** to classify poses like standing, sitting, and exercising from webcam input.

---

## 🚀 Features

- Real-time human pose detection using MoveNet Lightning (TFLite)
- Custom MLP classifier for pose-based action recognition
- OpenCV-based live video processing and visualization
- Dual-display: original feed and black canvas with pose skeleton
- Smooth classification of common human actions

---

## 🛠️ Tech Stack

- **Pose Estimation**: TensorFlow Lite MoveNet Lightning
- **Machine Learning Model**: MLP (Multi-Layer Perceptron)
- **Visualization**: OpenCV
- **Language**: Python
- **Libraries**: NumPy, Scikit-learn, Pickle

---

## 🧠 How It Works

1. Captures live video via webcam using OpenCV.
2. Runs each frame through the MoveNet Lightning TFLite model to extract 17 keypoints.
3. Normalizes and flattens the keypoints into a 34-dimensional feature vector.
4. Feeds the feature vector into a trained MLP classifier.
5. Displays the predicted pose label on both original and black canvas views.

---

## 🧪 Model Training

- Keypoints from labeled pose data were normalized and used as training data.
- `StandardScaler` was applied for normalization.
- Labels were encoded using `LabelEncoder`.
- Trained an MLP model using Scikit-learn with input shape `(34,)` (17 keypoints × 2).
- The trained model, scaler, and encoder were saved as `.pkl` files.

---

## 🧩 Future Enhancements

- Support for multi-person pose detection
- Pose tracking across frames
- Real-time feedback via audio or pop-up suggestions
- Web-based or mobile deployment

---

## 📷 Example Output

| Original View            | Skeleton-Only View        |
|--------------------------|---------------------------|
| ![pose](assets/pose.png) | ![skeleton](assets/skeleton.png) |

---

