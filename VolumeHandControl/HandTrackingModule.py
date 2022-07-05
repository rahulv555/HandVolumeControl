from tkinter import font
import cv2
from cv2 import FONT_HERSHEY_COMPLEX
from cv2 import trace
import mediapipe as mp
import time

from sqlalchemy import false


class HandDetector():
    def __init__(self, mode=False, maxHands=2, model_Complexity=1, detectionConfd=0.5, trackConfd=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfd = detectionConfd
        self.trackConfd = trackConfd
        self.model_Complexity = model_Complexity

        # from stock hand detection module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.model_Complexity, self.detectionConfd, self.trackConfd)

        # for drawing the lines
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # CONVERT TO RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # processing
        self.results = self.hands.process(img_rgb)
        # print(self.results.multi_hand_landmarks) - if hand in screen

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # iterates through each hand in screen

                if draw:
                    # draws the points and the connections
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    # FOR POSITION OF THE DOTS

    def findPosition(self, img, handNo=0, draw=True):

        # list of all landmark positions
        lmList = []

        if self.results.multi_hand_landmarks:
            # selecting hand
            myHand = self.results.multi_hand_landmarks[handNo]

            # id number and position(lm info) of the points on the hand
            for id, lm in enumerate(myHand.landmark):
                # id - index of the point
                # lm - (x, y, z) = coordinates but as ratio entire width and height(fraction)

                # dimensions of whole image
                height, width, channels = img.shape

                # getting position in terms of pixels
                cx, cy = int(lm.x*width), int(lm.y*height)

                lmList.append((id, cx, cy))

        return lmList


##################################################
# IF THIS FILE IS RUN

def main():
    cap = cv2.VideoCapture(0)  # the webcam
    detector = HandDetector()

    # FOR FPS
    prevTime = 0
    currTime = 0

    #############################################
    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        # for FPS
        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)

        cv2.imshow("VideoInput", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
