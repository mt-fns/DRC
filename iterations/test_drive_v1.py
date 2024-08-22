import cv2
import numpy as np
import math

def extract_edges(frame):
    # change image to hsv for masking
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # get only the blue pixels
    lower_blue = np.array([60, 40, 40])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

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
    min_threshold = 10  # minimal of votes

    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=8, maxLineGap=4)
    try:
        len(line_segments)
        print("yes")
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
            print("Vertical Line")
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
        print("no lane lines")
        return []

    detected_lane = calculate_slope_intercept(frame, line_segments)
    return detected_lane


def get_steering_angle(height, width, lane_lines):
    # two lane lines
    # print(lane_lines.shape)
    # print(lane_lines[1][0], lane_lines[0][0])

    if len(lane_lines) == 2:
        print("two lines detected")
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)

    # one lane line
    elif len(lane_lines) == 1:
        print("one line detected")
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

    print(steering_angle)


def display_steering(frame, lane_lines, steering_angle):
    thicc = 9
    color = (0, 255, 0)

    line_1 = lane_lines[0][0]
    start_line_1 = (line_1[0], line_1[1])
    end_line_1 = (line_1[2], line_1[3])

    if len(lane_lines) == 2:
        line_2 = lane_lines[1][0]
        start_line_2 = (line_2[0], line_2[1])
        end_line_2 = (line_2[2], line_2[3])

        image = cv2.line(frame, start_line_1, end_line_1, color, thicc)
        cv2.line(image, start_line_2, end_line_2, color, thicc)
    else:
        # one line detected
        cv2.line(frame, start_line_1, end_line_1, color, thicc)


cap = cv2.VideoCapture('../images/test1.mp4')
frame_counter = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    height, width, ch = frame.shape

    lane_lines = detect_lane(frame)

    frame_counter += 1
    print("DEBUG", frame_counter)

    if len(lane_lines) > 0:
        steering_angle = get_steering_angle(height, width, lane_lines)
        display_steering(frame, lane_lines, steering_angle)

    if ret == True:
        cv2.imshow('Vincent is hot', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

# while True:
#     frame = cv2.imread('road1_240x320.png')
#     height, width, ch = frame.shape
#
#     lane_lines = detect_lane(frame)
#     steering_angle = get_steering_angle(height, width, lane_lines)
#     display_steering(frame, lane_lines, steering_angle)
#
#     if cv2.waitKey(1) == ord('q'):
#         break