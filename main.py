

import cv2
from utils.pose_extractor import extract_keypoints, draw_landmarks
from utils.feedback_drawer import draw_feedback_box
from exercises.bicep_curl import feedback_bicep_curl
from exercises.lateral_raise import feedback_lateral_raise
from exercises.pushup import feedback_pushup
from exercises.lunge import feedback_lunge


# Options: "bicep_curl", "lateral_raise", "pushup", "lunge"
exercise_mode = "lateral_raise"


states = {
    "bicep_curl": {
        "left_stage": None,
        "right_stage": None,
        "left_count": 0,
        "right_count": 0,
    },
    "lateral_raise": {
        "left_stage": None,
        "right_stage": None,
        "left_count": 0,
        "right_count": 0,
    },
    "pushup": {
        "stage": "up",
        "count": 0,
        "form": 0,
    },
    "lunge": {
        "stage": "up",
        "count": 0,
    }
}


cap = cv2.VideoCapture(0)  

cv2.namedWindow("Pose Feedback", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose Feedback", 1000, 700)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints, results = extract_keypoints(frame)

    if keypoints:
        if exercise_mode == "bicep_curl":
            result = feedback_bicep_curl(keypoints, states["bicep_curl"])
        elif exercise_mode == "lateral_raise":
            result = feedback_lateral_raise(keypoints, states["lateral_raise"])
        elif exercise_mode == "pushup":
            result = feedback_pushup(keypoints, states["pushup"])
        elif exercise_mode == "lunge":
            result = feedback_lunge(keypoints, states["lunge"])
        else:
            result = {
                "exercise": "Unknown",
                "feedback": "Invalid mode",
                "count": 0,
                "angle": None
            }

        
        if results and results.pose_landmarks:
            draw_landmarks(frame, results)

        
        rep_count = result["count"]
        if isinstance(rep_count, tuple):
            rep_count = f"L:{rep_count[0]} | R:{rep_count[1]}"
        else:
            rep_count = str(rep_count)

        angle = result.get("angle", None)
        if isinstance(angle, tuple):
            angle = f"L:{int(angle[0])}° | R:{int(angle[1])}°"
        elif isinstance(angle, (int, float)):
            angle = f"{int(angle)}°"
        else:
            angle = None

        draw_feedback_box(
            frame,
            pose_name=result["exercise"],
            rep_count=rep_count,
            feedback=result["feedback"],
            angle=angle
        )
    else:
        draw_feedback_box(
            frame,
            pose_name="No pose",
            rep_count="0",
            feedback="Ensure full body visible",
            angle=None
        )

    cv2.imshow("Pose Feedback", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
