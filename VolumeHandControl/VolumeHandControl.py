from ast import Import
import cv2
import time
import numpy as np
import math

import HandTrackingModule as htm

# for volume control - pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMasterVolumeLevel()
minVol, maxVol = volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]
vol = 0
volBar = 400  # 0 at 400
volPer = 0

##########################################

wCam, hCam = 1920, 1080

########################################


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0


# hand detector object
hDetector = htm.HandDetector(detectionConfd=0.7)


while True:
    success, img = cap.read()
    img = hDetector.findHands(img)

    # landmarks list
    lmList = hDetector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # id 4 - thumb tip, id 8 = index tip

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2  # center

        # print(lmList[4], lmList[8])
        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

        # creating line between them
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # length of line
        length = math.hypot(x2-x1, y2-y1)
        print(length)

        # length range is 50-250
        # volume range is -65-0

        vol = np.interp(length, [50, 250], [minVol, maxVol])
        volBar = np.interp(length, [50, 250], [400, 150])
        volPer = np.interp(length, [50, 250], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

    # Volume Bar
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}', (50, 100),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS : {int(fps)}', (40, 70),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Video Input", img)
    cv2.waitKey(1)
