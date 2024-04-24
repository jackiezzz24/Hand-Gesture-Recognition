import os
import sys
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from model import MyNetwork  # Assuming the model is in a file named model.py
import cv2
import numpy as np
import torch.nn.functional as F

def show_video(model_path, device='cpu'):
    # get the trained model
    model = torch.load('model.pth')
    model.eval()

    # get ppt image
    pptPath = "presentation"
    pptImages = sorted(os.listdir(pptPath), key=lambda x: int(x.split('.')[0])) # sorted the images to make sure 10.jpg would not occur after 1.jpg
    print(pptImages)

    # video variables
    width, height = 1280, 720
    width2, height2 = int(128 * 2), int(72 * 2)

    # index of image showing
    imgNumber = 0
    buttonPressed = False
    buttonCounter = 0
    duration = 30 

    # show video
    cap = cv2.VideoCapture(0) #specify 0 or 1, 1 works for my mac 
    cap.set(3,width)
    cap.set(4,height)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    try:
        while True:
            # get the image path
            currentImgPath = os.path.join(pptPath, pptImages[imgNumber])
            currentImg = cv2.imread(currentImgPath)
            currentImg = cv2.resize(currentImg, (width, height))



            # ret indicate whether it capture successfully
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame. Exiting ...")
                break

            # Convert image to grayscale
            img_gray = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
            
            # Calculate the aspect ratio
            aspect_ratio = img_gray.shape[1] / img_gray.shape[0]
            new_size = (128, int(128 * aspect_ratio))

            # Resize the image using bilinear interpolation
            img_intensity = cv2.resize(img_gray, new_size, interpolation=cv2.INTER_LINEAR)

            # Crop the image to make it 128x128
            crop_height = min(128, img_intensity.shape[0])
            crop_width = min(128, img_intensity.shape[1])
            
            start_height = (img_intensity.shape[0] - crop_height) // 2
            start_width = (img_intensity.shape[1] - crop_width) // 2

            end_height = start_height + crop_height
            end_width = start_width + crop_width
            img_cropped = img_intensity[start_height:end_height, start_width:end_width]
            
            # Normalize and append
            img_normalized = img_cropped.astype(np.float32) / 255.0
            tensor_frame = torch.tensor(img_normalized, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)

            classes = ('Palm', 'l', 'Fist', 'Fist_moved', 'Thumb', 'Index', 'Ok', 'Palm_moved', 'C', 'Down')

            # predict
            with torch.no_grad():
                predictions = model(tensor_frame)
                _, predicted_label = predictions.max(1)
                predicted_gesture = classes[predicted_label.item()]

            # cv2.putText(frame, f'Predicted: {predicted_gesture}', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

            if buttonPressed is False:
                # 1. show next slide
                if predicted_gesture == 'C':
                    # display prediction in the video
                    cv2.putText(frame, f'Predicted: {predicted_gesture}', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                    if imgNumber < len(pptImages) - 1:
                        buttonPressed = True
                        imgNumber += 1

                # 2. show previous slide
                if predicted_gesture == 'Palm':
                    cv2.putText(frame, f'Predicted: {predicted_gesture}', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                    if imgNumber > 0:
                        buttonPressed = True
                        imgNumber -= 1

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




            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    show_video('model.pth')