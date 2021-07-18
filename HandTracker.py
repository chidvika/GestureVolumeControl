import cv2  # image processing
import mediapipe as mp  # to detect hand
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon) #hand tracking and landmarks inbuilt module
        self.mpDraw = mp.solutions.drawing_utils  #importing drawing package

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert bgr img into rgb
        self.results = self.hands.process(imgRGB)  # returns list of indices of landmarks
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks: # check weather hand is detected or not
            for handlms in self.results.multi_hand_landmarks: # choosing the hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS) # to detect the landmarks

        return img

    def findPosition(self, img, handNo=0, draw = True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture(0) #web cam access
    cTime = 0
    pTime = 0
    detector = handDetector() # basic variables initialization
    while True:
        success, img = cap.read()
        img = detector.findHands(img) # landmark detection
        lmList = detector.findPosition(img) # to locate desired landmarks
        if len(lmList)!=0:
            print(lmList[4])
        ## counting no.of frames per sec
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()