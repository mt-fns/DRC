import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
	ret, frame = cap.read()
	cv2.imshow("test", frame)
	
	if cv2.waitKey(1) == ord('q'):
		break
cap.release()
cap.destroyAllWindows()	
	
