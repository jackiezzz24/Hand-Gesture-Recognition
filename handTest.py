from handDetector import HandDetector
import cv2


# Example usage:
detector = HandDetector()
cap = cv2.VideoCapture(1)

while True:
    success, frame = cap.read()
    if not success:
        continue

    hands, frame = detector.findHands(frame)
    #print(hands)
    lmList = detector.findPosition(frame)
    #print(lmList)


    hands, frame = detector.findHands(frame)
    fingers = detector.fingersUp(frame=frame)
    #print(fingers)

    cx, cy = detector.getCenterIndex()
    # print(cx)
    # print(cy)

    if lmList:
        indexFinger = lmList[8][1], lmList[8][2]
        cv2.circle(frame, indexFinger, 12, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Frame", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


