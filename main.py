# Name: Jiaqi Zhao, Kexuan Chen, Zhimin Liang
# Date: April 9th
# 

# import statements
import os
import sys
import cv2
from handDetector import HandDetector

# main function
def main(argv):
    # video variables
    width, height = 1280, 720
    width2, height2 = int(128 * 2), int(72 * 2)
    pptPath = "presentation"
    threshold = 300 # line to trigger gesture detection

    # camera setup

    # for macbook, 0 will use iphone camera, 1 will use computer's camera, so 1 works for me
    cap = cv2.VideoCapture(1)
    cap.set(3,width)
    cap.set(4,height)

    # get ppt image
    pptImages = sorted(os.listdir(pptPath), key=lambda x: int(x.split('.')[0])) # sorted the images to make sure 10.jpg would not occur after 1.jpg
    print(pptImages)

    # index of image showing
    imgNumber = 0
    buttonPressed = False
    buttonCounter = 0
    #10 frame after the previous gesture, we do not accept new gestures
    #otherwise it will apply the gestures too quickly
    duration = 30 

    # get hand detector
    detector = HandDetector(min_detection_confidence = 0.8)

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # get the image path
        currentImgPath = os.path.join(pptPath, pptImages[imgNumber])
        currentImg = cv2.imread(currentImgPath)

        hands, frame = detector.findHands(frame)

        cv2.line(frame, (0, threshold), (width, threshold), (0, 255, 0), 10)

        if hands and buttonPressed is False:
            hand = hands[0]
            # check how many fingers are up
            fingers = detector.fingersUp(frame)
            lmList = detector.lmList

            if lmList:
                # get the index of index Finger
                indexFinger = lmList[8][0], lmList[8][1]

                cx, cy = detector.getCenterIndex()

                # only check the hand when it is above the threshold
                if cy <= threshold:
                    # 1. left, show previous slide
                    if fingers == [1, 0, 0, 0, 0]:
                        print("left")
                        if imgNumber > 0:
                            buttonPressed = True
                            imgNumber -= 1

                    # 2. right, show next slide
                    if (fingers == [0, 0, 0, 0, 1]):
                        print("right")
                        if imgNumber < len(pptImages) - 1:
                            buttonPressed = True
                            imgNumber += 1
                
                if fingers == [0, 1, 1, 0, 0]:
                    cv2.circle(currentImg, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        # change the statues of buttonPressed
        if buttonPressed:
            buttonCounter += 1
            if buttonCounter > duration:
                buttonCounter = 0
                buttonPressed = False


        # resize and rearrange the windows
        frameSmall = cv2.resize(frame, (width2, height2))
        h,w,_ = currentImg.shape
        currentImg[0:height2, w-width2:w] = frameSmall # put the video on the right corner of ppt

        cv2.imshow("frame", frame)
        cv2.imshow("ppt", currentImg)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break


    return

if __name__ == "__main__":
    main(sys.argv)