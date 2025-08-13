import cv2

def draw_feedback_box(frame, pose_name, rep_count, feedback, angle=None):
    height, width, _ = frame.shape

    
    box_height = 120
    padding = 10
    cv2.rectangle(frame, (0, 0), (width, box_height), (0, 0, 0), -1)

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)
    thickness = 2
    spacing = 25

    cv2.putText(frame, f"Exercise: {pose_name}",
                (padding, padding + spacing * 1),
                font, font_scale, color, thickness, cv2.LINE_AA)

    
    cv2.putText(frame, f"Reps: {rep_count}",
                (padding, padding + spacing * 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    
    feedback_text = f"Feedback: {feedback}"
    cv2.putText(frame, feedback_text,
                (padding, padding + spacing * 3),
                font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

   
    if angle:
        cv2.putText(frame, f"Angle: {angle}",
                    (width - 200, padding + spacing * 2),
                    font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)
