from cv2 import cv2
import numpy as np

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Cannot open camera")
    exit()

while 1:
    ret, frame = video.read()

    if not ret:
        print("Can't receive frame")
        break

    output = frame.copy() # get video frame
    
    # convert to grayscale and blur for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    # detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 100, minRadius=40)

    # convert the circle parameters (x, y, r) to ints and then display them on the original frame 
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0,255,0), 4)
            
    cv2.imshow("image", output)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
