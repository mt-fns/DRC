import cv2
import numpy as np
import math

from gpiozero import PhaseEnableMotor
from gpiozero import AngularServo
from time import sleep
from gpiozero.pins.pigpio import PiGPIOFactory
import pigpio


SMOOTHING_FACTOR = 0.5

MAX_ANGLE = 15
MIN_ANGLE = -15
STRAIGHT_ANGLE = 0

pwm2_pin = 5
dir2_pin = 6
pwm1_pin = 13
dir1_pin = 19
servo_pin = 17

# drive setup
motor1 = PhaseEnableMotor(dir1_pin, pwm1_pin)
motor2 = PhaseEnableMotor(dir2_pin, pwm2_pin)
factory = PiGPIOFactory()
pi = pigpio.pi('soft', 8888)
servo = AngularServo(servo_pin, min_pulse_width=0.0005, max_pulse_width=0.00255, pin_factory=factory)

servo.angle = -3

def turn(angle) :
    # this is just a random formula to choose speed based on, linearly decreasing speed from some max to 0.1 which is real slow
    speed = 0.2
    if abs(angle) > 20:
        speed = 0.13
    angle = (MAX_ANGLE - MIN_ANGLE)/2 * angle/30 + STRAIGHT_ANGLE
    if angle > MAX_ANGLE:
        angle = MAX_ANGLE
    elif angle < MIN_ANGLE:
        angle = MIN_ANGLE
    print("angle ", angle)
    # these motor directinos might need to be swapped?
    
    motor1.backward(speed)
    motor2.forward(speed)
    servo.angle = angle



def extract_edges(frame):
    # change image to hsv for masking
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # get only the blue pixels

    lower_blue = np.array([60, 40, 40])
    upper_blue = np.array([150, 255, 255])

    # alternative hsv mask
    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])

    # terrible mask
    # lower_yellow = np.array([0, 10, 170])
    # upper_yellow = np.array([180, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # get yellow and blue lines
    mask = mask_blue #| mask_yellow

    # extract edges from lines
    edges = cv2.Canny(mask, 200, 400)    # change to contour detection later

    return edges

def crop_image(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    # wtf is this???
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 20  # minimal of votes

    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=8, maxLineGap=4)
    try:
        len(line_segments)
        #print("yes")
        return line_segments
    except:
        return []


def calculate_slope_intercept(frame, line_segments):
    height, width, ch = frame.shape
    lane_lines = []

    left_line = []
    right_line = []

    # bound is 2/3
    bound = 1/3
    left_bound = width * (1 - bound)
    right_bound = width * bound

    for line in line_segments:
        # points that make up the line segments
        x1, y1, x2, y2 = line[0]

        # skip vertical line
        if x1 == x2:
            #print("Vertical Line")
            continue

        fit = np.polyfit((x1, x2), (y1, y2), 1)
        slope = fit[0]
        intercept = fit[1]

        # check if left line is within bounds
        if slope < 0:
            if x1 < left_bound and x2 < left_bound:
                left_line.append((slope, intercept))
        # check if right line is within bounds
        else:
            if x1 > right_bound and x2 > right_bound:
                right_line.append((slope, intercept))

    left_avg = np.average(left_line, axis=0)
    right_avg = np.average(right_line, axis=0)

    if len(left_line) > 0:
        lane_lines.append(get_end_points(frame, left_avg))

    if len(right_line) > 0:
        lane_lines.append(get_end_points(frame, right_avg))

    return lane_lines

# helper function for slope intercept
def get_end_points(frame, line_fit):
    height, width, _ = frame.shape
    slope, intercept = line_fit
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def detect_lane(frame):
    edges = extract_edges(frame)
    cropped_edges = crop_image(edges)
    line_segments = detect_line_segments(cropped_edges)

    if len(line_segments) == 0:
        #print("no lane lines")
        return []

    detected_lane = calculate_slope_intercept(frame, line_segments)
    return detected_lane


def get_steering_angle(height, width, lane_lines):
    # two lane lines
    # print(lane_lines.shape)
    # print(lane_lines[1][0], lane_lines[0][0])

    if len(lane_lines) == 2:
        #print("two lines detected")
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)

    # one lane line
    elif len(lane_lines) == 1:
        #print("one line detected")
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg  # this is the steering angle needed by front wheel

    return steering_angle


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=5):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def stabilize_steering(previous_angle, current_angle, previous_weight=(1 - SMOOTHING_FACTOR), current_weight=SMOOTHING_FACTOR):
    stabilized_angle = previous_angle * previous_weight + current_angle * current_weight
    return stabilized_angle

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image

def test_video(src):
    cap = cv2.VideoCapture(src)
    previous_angle = 0

    # how many angles to output per second (camera has 60fps)
    #won't this output 60/sensitivity = 12 per second?
    frame_rate = 2 # 10 per second
    steering_rate = 8 #
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if(ret):
            frame_counter += 1
        else:
            continue
        if (frame_counter % frame_rate != 0):
            continue
        height, width, ch = frame.shape
        
        lane_lines = detect_lane(frame)
        

        if (frame_counter % frame_rate == 0):
            #print('TEST 1')

            if len(lane_lines) > 0:
                steering_angle = get_steering_angle(height, width, lane_lines)

                # TODO: output steering angles based on sensitivity
                previous_angle = stabilize_steering(previous_angle, steering_angle)

                #print("present angle", steering_angle)
                #print("smoothed angle:", previous_angle)

                angle = (MAX_ANGLE - MIN_ANGLE)/2 * previous_angle/30 + STRAIGHT_ANGLE
                if angle > MAX_ANGLE:
                    angle = MAX_ANGLE
                elif angle < MIN_ANGLE:
                    angle = MIN_ANGLE
                #print("normalised servo angle", angle)
                # lane_lines_frame = display_lines(frame, lane_lines)

                # if steering_angle is not None:
                # heading_line_frame = display_heading_line(frame, previous_angle)
                # cv2.imshow('Test v2', heading_line_frame)
            # continue
        if frame_counter % steering_rate == 0:
            
            print("present angle", steering_angle)
            print("smoothed angle:", previous_angle)
            print("frame_counter", frame_counter)
            turn(previous_angle)

        # if len(lane_lines) > 0:
        #     #print('TEST')
        #     steering_angle = get_steering_angle(height, width, lane_lines)

        #     previous_angle = stabilize_steering(previous_angle, steering_angle)

        #     #print(steering_angle)
        #     #print(previous_angle)

        if ret == True:
            # cv2.imshow('Vincent is hot', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def test_image(src):
    frame = cv2.imread(src)
    lane_lines = detect_lane(frame)
    lane_lines_image = display_lines(frame, lane_lines)

    cv2.imshow("lane lines", lane_lines_image)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()

test_video(0)