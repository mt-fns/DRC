import cv2
import numpy as np

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

    return line_segments

while True:
    frame = cv2.imread('road1_240x320.png')
    edges = extract_edges(frame)
    cropped_edges = crop_image(edges)
    line_segments = detect_line_segments(cropped_edges)

    cv2.imshow("balls", cropped_edges)

    if cv2.waitKey(1) == ord('q'):
        break

print(line_segments)
print(cropped_edges.shape)
print("Hello World")

