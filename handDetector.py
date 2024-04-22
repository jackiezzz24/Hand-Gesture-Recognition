import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        # hands object: 21 landmarks
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        self.lmList = None

    def findHands(self, frame):
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_RGB)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return self.results, frame

    def findPosition(self, frame, handNo=0):
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = frame.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        self.lmList = lmList
        return lmList

    def fingersUp(self, frame):
        lmList = self.findPosition(frame=frame)
        # landmarks for thumb, index, middle, ring, and little fingers
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []

        # Check each finger except the thumb
        if len(lmList) != 0:
            for i in range(1, 5):
                # If the tip landmark y-coordinate is less than the y-coordinate of the landmark just below the tip
                # it is considered that the finger is up
                if lmList[tip_ids[i]][2] < lmList[tip_ids[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Thumb
            if lmList[tip_ids[0]][1] < lmList[tip_ids[0] - 1][1]:
                fingers.insert(0, 1)
            else:
                fingers.insert(0, 0)

        return fingers

    
    def getCenterIndex(self):
        if not self.lmList:
            return 0, 0

        sum_x = sum_y = 0
        for lm in self.lmList:
            sum_x += lm[1] 
            sum_y += lm[2]
        
        cx = int(sum_x / len(self.lmList))
        cy = int(sum_y / len(self.lmList))

        return cx, cy
