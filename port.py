from cv2 import cv2
import numpy as np
import imutils
import math
from networktables import NetworkTables

BLURFACTOR = 5
MINPORTAREA = 800
CNTTHRESH1 = 255
CNTRHRESH2 = 255
FOCALLENGTH = 720 # can be calculated with the height in pixels and distance
PORTAPOTHEM = 3.5 # can be any unit
video = cv2.VideoCapture(0)

SERVERADDR = '10.xx.xx.2' # Put the server address here

NetworkTables.initialize(server=SERVERADDR)
sd = NetworkTables.getTable("Vision")

if not video.isOpened():
    print("Cannot open camera")
    exit()

def getRelativePos(b, x):
    return (((FOCALLENGTH*PORTAPOTHEM)/b)*(x - (video.get(cv2.CAP_PROP_FRAME_WIDTH)/2))/720, (FOCALLENGTH*PORTAPOTHEM)/b)

def yval(p):
    return p[1]

def carea(p):
    return cv2.contourArea(p[0])

while 1:
    ret, frame = video.read()

    if not ret:
        print("Can't receive frame")
        break

    # Blurring the image
    blurred = cv2.medianBlur(frame, BLURFACTOR)

    # Convert to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    imgCanny = cv2.Canny(gray, CNTTHRESH1, CNTRHRESH2)
    imgCanny = cv2.dilate(imgCanny, None, iterations=2)
    imgCanny = cv2.erode(imgCanny, None, iterations=2)

    cnts, h = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    hexagons = [] # Find hexagons in the image
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        if len(approx) == 6:
            hexagons.append((cnt, approx))

    if hexagons: # If found, outline the one with the highest area
        c = max(hexagons, key=carea)

        if cv2.contourArea(c[0]) > MINPORTAREA:
            cv2.drawContours(frame, [c[1]], 0, (0,255,0), 3)
            pts = []
            for x in c[1]:
                pts.append(x[0])

            pts.sort(key=yval)
            midpoint = (int((pts[-1][0] + pts[-2][0])/2), int((pts[-1][1] + pts[-2][1])/2))
            M = cv2.moments(c[1])
            
            if  M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) # Calculate the center of the ball
            else:
                center = (0, 0)

            #cv2.circle(frame, center, 5, (0, 0, 255), -1)
            #cv2.line(frame, center, midpoint, (255, 0, 0), 3)

            apothem_dist = math.sqrt(((center[0]-midpoint[0])**2) + ((center[1]-midpoint[1])**2))
            rp = getRelativePos(apothem_dist, center[0])
            sd.putNumber("Port Lateral Position", rp[0])
            sd.putNumber("Port Longitudinal Position", rp[1])
            #cv2.putText(frame, "Lateral: "+str(rp[0])+" Longtitudinal: "+str(rp[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    #cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()