# Name: Jiaqi Zhao, Kexuan Chen, Zhimin Liang
# Date: April 9th
# 

# import statements
import sys
import cv2

# main function
def main(argv):
    # video variables
    width, height = 1280, 720

    # camera setup

    # for macbook, 0 will use iphone camera, 1 will use computer's camera, so 1 works for me
    cap = cv2.VideoCapture(1)
    cap.set(3,width)
    cap.set(4,height)

    while True:
        success, frame = cap.read()
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break


    return

if __name__ == "__main__":
    main(sys.argv)