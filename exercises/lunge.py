

import numpy as np
from utils.pose_utils import calculate_angle

def feedback_lunge(landmarks, state):
    """
    Evaluates lunge posture using LEFT leg joints.
    Logic:
        - Standing up: Knee angle > 160°
        - Lunge down: Knee angle < 90°
        - Handles rep counting and feedback
    """

    # LEFT leg joints
    hip = landmarks[23][:2]
    knee = landmarks[25][:2]
    ankle = landmarks[27][:2]

    angle = calculate_angle(hip, knee, ankle)

    feedback = ""

    if angle > 160:
        feedback = "Standing straight. Prepare to lunge down."
        state["stage"] = "up"

    elif angle < 90 and state["stage"] == "up":
        state["stage"] = "down"
        state["count"] += 1
        feedback = "Nice lunge! Rep counted. Push back up."

    elif 90 <= angle <= 160:
        feedback = "Hold your position. Maintain form."

    else:
        feedback = "Check posture."

    return {
        "exercise": "Lunge",
        "angle": angle,
        "count": state["count"],
        "feedback": feedback,
    }
