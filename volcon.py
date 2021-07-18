import cv2
import time
import numpy as np
import math
import HandTracker as ht
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = ht.handDetector(detectionCon=0.7)
## volume control parameters
devices = AudioUtilities.GetSpeakers() #to find system speakers
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None) # building an interface
volume = cast(interface, POINTER(IAudioEndpointVolume)) # to output the speakers functionality
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPar = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img) # loacte land marks
    lmList = detector.findPosition(img, draw=False) # find position of landmarks
    if len(lmList)!=0:
        #print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1],lmList[4][2] # landmark 4 position
        x2, y2 = lmList[8][1], lmList[8][2] # landmark 8 position
        cx, cy = (x1+x2)//2, (y1+y2)//2 # midpoint of 4 and 8

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        #print(length)
        # hand range 50 - 300
        # Volume Range -65 - 0

        vol = np.interp(length, [50, 300], [minVol, maxVol]) # to convert range
        volBar = np.interp(length, [50, 300], [400, 150]) # to convert vol range into volume bar length
        volPar = np.interp(length, [50, 300], [0, 100]) # to convert vol range into vol percentage
        #print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPar)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()
