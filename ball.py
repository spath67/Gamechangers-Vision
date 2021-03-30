from cv2 import cv2
import numpy as np
import imutils
import threading
from networktables import NetworkTables

video = cv2.VideoCapture(0)

BLURFACTOR = 11
MAXRADIUS = 10
COLORL = (20, 182, 51)
COLORU = (77, 255, 255)
FOCALLENGTH = 1400
BALLRADIUS = 3

SERVERADDR = '10.11.55.2' # Put the server address here

NetworkTables.initialize(server=SERVERADDR)
sd = NetworkTables.getTable("Vision")

if not video.isOpened():
    print("Cannot open camera")
    exit()

def getRelativePos(b, x):
    return (((FOCALLENGTH*BALLRADIUS)/b)*(x - (video.get(cv2.CAP_PROP_FRAME_WIDTH)/2))/FOCALLENGTH, (FOCALLENGTH*BALLRADIUS)/b)

while 1:
    ret, frame = video.read()

    if not ret:
        print("Can't receive frame")
        break

    # Blurring the image
    blurred = cv2.medianBlur(frame, BLURFACTOR)

    # Convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Filter unwanted colors, specks, etc., etc.
    mask = cv2.inRange(hsv, COLORL, COLORU)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contour which would be ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea) # Find the one with the largest area
        ((x, y), r) = cv2.minEnclosingCircle(c) # Use minenclosingcircle to find the radius 

        M = cv2.moments(c)
        if  M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) # Calculate the center of the ball
        else:
            center = (0, 0)
        
        if r > 10:
            # cv2.circle(frame, (int(center[0]), int(center[1])), int(r),
            #    (0, 255, 255), 2) 
            rp = getRelativePos(r, center[0])
            sd.putNumber("Ball Lateral Position", rp[0])
            sd.putNumber("Ball Longitudinal Position", rp[1])
            # cv2.putText(frame, "Lateral: "+str(rp[0])+" Longtitudinal: "+str(rp[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            # cv2.circle(frame, center, 5, (0, 0, 255), -1)

    #cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
