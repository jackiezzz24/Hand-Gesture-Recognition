# CS5330 - Project5
# Name: Jiaqi Zhao, Kexuan Chen
# Date: Mar 27, 2024
# Extension: identify numbers in live video

import cv2
import torch
import numpy as np
from model import MyNetwork

def show_video(model_path, device='cpu'):
    # get the trained model
    model = torch.load(model_path).to(device)
    model.eval()

    # show video
    cap = cv2.VideoCapture(1) #specify 0 or 1, 1 works for my mac 
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    try:
        while True:
            # ret indicate whether it capture successfully
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame. Exiting ...")
                break

            # convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # resize to 28 * 28 to fit the model needs
            gray_resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

            gray_normalized = gray_resized / 255.0
            tensor_frame = torch.tensor(gray_normalized, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)

            # predict
            with torch.no_grad():
                predictions = model(tensor_frame)
                _, predicted_label = predictions.max(1)

            # display prediction in the video
            cv2.putText(frame, f'Predicted Digit: {predicted_label.item()}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    show_video('model.pth')
