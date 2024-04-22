# CS5330 - Project5
# Name: Jiaqi Zhao, Kexuan Chen
# Date: Mar 24, 2024
# test the trained network

# import statements
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from model import MyNetwork
import cv2
import numpy as np


# test the model
def testModel(model):
    hand_images = []

    # get the digit images
    for i in range(10):
        # read the image and convert to grayscale
        img = cv2.imread(f'digits/digit_{i}.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # match intensity with MNIST dataset
        # img_intensity = cv2.bitwise_not(img_gray)
        img_intensity = cv2.resize(img_gray, (28, 28)) 
        img_intensity = img_intensity.astype(np.float32) / 255.0

        hand_images.append(img_intensity)

    # convert list to PyTorch tensor
    digits_tensor = torch.tensor(hand_images)
    digits_tensor = digits_tensor.unsqueeze(1)  # add channel dimension

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    predictions = []
    images = []

    with torch.no_grad():
        for i, inputs in enumerate(digits_tensor, 0):  
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            print('Outputs:', ' '.join(['%.2f' % o.item() for o in outputs[0]]))
            print('Predicted:', classes[predicted.item()], '\n')

            # append the image and prediction to lists
            images.append(inputs.squeeze().numpy())
            predictions.append(predicted.item())

            if i == 9: 
                break

    # display the results with predictions
    fig, axs = plt.subplots(4, 3)
    for i in range(4):
        for j in range(3):
            ax = axs[i, j]
            index = i * 3 + j
            if index < len(images):
                ax.imshow(images[index], cmap='gray', interpolation='none')
                ax.set_title(f'Prediction: {classes[predictions[index]]}')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
            else:
                ax.axis('off')  
    plt.tight_layout()
    plt.show()

# main function 
def main(argv):
    
    # load the model 
    model = torch.load('model.pth')
    # set model to evaluation mode
    model.eval()

    # test on first 10 examples in the test set
    testModel(model)

    return

if __name__ == "__main__":
    main(sys.argv)