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
    cv2.imshow("Frame", frame)
    lmList = detector.findPosition(frame)
    #print(lmList)


    hands, frame = detector.findHands(frame)
    fingers = detector.fingersUp(frame=frame)
    #print(fingers)

    center = detector.getCenterIndex()
    if center:
        cx, cy = center
        print(cx)
        print(cy)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


