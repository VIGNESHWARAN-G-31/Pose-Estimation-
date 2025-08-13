import numpy as np
from utils.pose_utils import calculate_angle

def feedback_bicep_curl(landmarks, state):
    l_shoulder = landmarks[11][:2]
    l_elbow = landmarks[13][:2]
    l_wrist = landmarks[15][:2]

    r_shoulder = landmarks[12][:2]
    r_elbow = landmarks[14][:2]
    r_wrist = landmarks[16][:2]

    l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

    l_fb, r_fb = "", ""

    
    if l_angle > 160:
        l_fb = "Left arm down"
        state["left_stage"] = "down"
    elif 80 < l_angle <= 160:
        l_fb = "Pull left arm more"
    elif l_angle <= 30 and state["left_stage"] == "down":
        state["left_stage"] = "up"
        state["left_count"] += 1
        l_fb = "Left rep counted"
    else:
        l_fb = "GREAT."

    
    if r_angle > 160:
        r_fb = "Right arm down"
        state["right_stage"] = "down"
    elif 80 < r_angle <= 160:
        r_fb = "Pull right arm more"
    elif r_angle <= 30 and state["right_stage"] == "down":
        state["right_stage"] = "up"
        state["right_count"] += 1
        r_fb = "Right rep counted"
    else:
        r_fb = "GREAT."

    
    asymmetry = ""
    if abs(l_angle - r_angle) > 40:
        if l_angle < r_angle:
            asymmetry = "Left higher than right"
        else:
            asymmetry = "Right higher than left"

    feedback = f"{l_fb} | {r_fb}"
    if asymmetry:
        feedback += f" | {asymmetry}"

    return {
        "exercise": "Bicep Curl",
        "angle": (round(l_angle, 1), round(r_angle, 1)),
        "count": (state["left_count"], state["right_count"]),
        "feedback": feedback,
    }
