

import numpy as np
from utils.pose_utils import calculate_angle

def feedback_lateral_raise(landmarks, state):
    
    l_hip = landmarks[23][:2]
    l_shoulder = landmarks[11][:2]
    l_elbow = landmarks[13][:2]

    
    r_hip = landmarks[24][:2]
    r_shoulder = landmarks[12][:2]
    r_elbow = landmarks[14][:2]

    
    l_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
    r_angle = calculate_angle(r_hip, r_shoulder, r_elbow)

    l_fb, r_fb = "", ""

    
    if l_angle < 30:
        l_fb = "Raise left arm higher"
        state["left_stage"] = "down"
    elif l_angle > 80 and state["left_stage"] == "down":
        l_fb = "Left rep counted"
        state["left_stage"] = "up"
        state["left_count"] += 1
    else:
        l_fb = "Hold left arm steady"

    
    if r_angle < 30:
        r_fb = "Raise right arm higher"
        state["right_stage"] = "down"
    elif r_angle > 80 and state["right_stage"] == "down":
        r_fb = "Right rep counted"
        state["right_stage"] = "up"
        state["right_count"] += 1
    else:
        r_fb = "Hold right arm steady"

    
    asymmetry = ""
    if abs(l_angle - r_angle) > 30:
        if l_angle < r_angle:
            asymmetry = "Left arm lower than right"
        else:
            asymmetry = "Right arm lower than left"

    
    feedback = f"{l_fb} | {r_fb}"
    if asymmetry:
        feedback += f" | {asymmetry}"

    return {
        "exercise": "Lateral Raise",
        "angle": (round(l_angle, 1), round(r_angle, 1)),
        "count": (state["left_count"], state["right_count"]),
        "feedback": feedback,
    }
