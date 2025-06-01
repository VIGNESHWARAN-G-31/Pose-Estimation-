import tensorflow as tf
import numpy as np
import cv2
import pickle

# 1. Load MoveNet Lightning Model
interpreter = tf.lite.Interpreter(model_path=r'D:\PoseMate\movenet_lightning.tflite')
interpreter.allocate_tensors()

# 2. Load the pre-trained MLP model, scaler, and label encoder
mlp_model_path = "your_file_path"
scaler_path = "your_file_path"
label_encoder_path = "your_file_path"

with open(mlp_model_path, 'rb') as f:
    mlp_model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# 3. Helper Function to Normalize Keypoints
def normalize_keypoints(keypoints, img_width, img_height):
    return [(x / img_width, y / img_height) for x, y, _ in keypoints]

# 4. Helper Function to Draw Keypoints
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

# 5. Helper Function to Draw Edges
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# 6. Start Video Capture
cap = cv2.VideoCapture(0)

EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (192, 192))
    input_image = np.expand_dims(img, axis=0).astype(np.uint8)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    img_height, img_width, _ = frame.shape
    keypoints = keypoints_with_scores[0].reshape(-1, 3)
    keypoints_normalized = normalize_keypoints(keypoints, img_width, img_height)
    keypoints_flat = np.array(keypoints_normalized).flatten().reshape(1, -1)

    # Scale the input features
    keypoints_scaled = scaler.transform(keypoints_flat)

    # Predict using the MLP model
    prediction = mlp_model.predict(keypoints_scaled)
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    # Draw keypoints and edges
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

    black_frame = np.zeros_like(frame)
    draw_connections(black_frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(black_frame, keypoints_with_scores, 0.4)

    cv2.putText(frame, f"Predicted: {predicted_class}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Pose Detection', frame)
    cv2.imshow('Skeleton Only', black_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
