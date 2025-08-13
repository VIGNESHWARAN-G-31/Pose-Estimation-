import numpy as np
from utils.pose_utils import calculate_angle

def feedback_pushup(landmarks, state):
    left_shoulder = landmarks[11][:2]
    left_elbow = landmarks[13][:2]
    left_wrist = landmarks[15][:2]

    left_hip = landmarks[23][:2]

    elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)

    
    if "count" not in state:
        state["count"] = 0
        state["stage"] = "up"
        state["form"] = 0


    if elbow_angle > 160 and shoulder_angle > 40:
        state["form"] = 1

    feedback = "Fix your form"
    if state["form"] == 1:
        if elbow_angle <= 90 and state["stage"] == "up":
            state["stage"] = "down"
            feedback = "Down position reached"

        elif elbow_angle >= 160 and state["stage"] == "down":
            state["stage"] = "up"
            state["count"] += 1
            feedback = "Push-up counted!"

        else:
            feedback = "Keep going"

    return {
        "exercise": "Push-Up",
        "count": state["count"],
        "feedback": feedback,
        "angle": elbow_angle
    }
