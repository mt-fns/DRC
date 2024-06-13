from main import *
from servo_control import drive_servo

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    height, width, ch = frame.shape

    lane_lines = detect_lane(frame)
    if len(lane_lines) > 0:
        steering_angle = get_steering_angle(height, width, lane_lines)
        drive_servo(steering_angle)
        # display_steering(frame, lane_lines, steering_angle)

    if ret == True:
        cv2.imshow('Vincent is hot', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
