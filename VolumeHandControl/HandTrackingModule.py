import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, model_Complexity=1, detectionConfd=0.5, trackConfd=0.5):
        # Ensure arguments are cast to correct types for the Mediapipe C++ backend
        self.mode = bool(mode)
        self.maxHands = int(maxHands)
        self.model_Complexity = int(model_Complexity)
        self.detectionConfd = float(detectionConfd)
        self.trackConfd = float(trackConfd)

        # Explicitly reference solutions to avoid AttributeErrors
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        
        # In 0.10.x, using keyword arguments is much safer
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.model_Complexity,
            min_detection_confidence=self.detectionConfd,
            min_tracking_confidence=self.trackConfd
        )

    def findHands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            # Check if requested hand index exists
            if len(self.results.multi_hand_landmarks) > handNo:
                myHand = self.results.multi_hand_landmarks[handNo]
                h, w, c = img.shape
                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    prevTime = 0

    while True:
        success, img = cap.read()
        if not success: break

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        
        # Example: Print position of the tip of the index finger (ID 8)
        if len(lmList) != 0:
            print(f"Index Tip: {lmList[8]}")

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()