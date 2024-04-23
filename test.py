# Name: Jiaqi Zhao, Kexuan Chen, Zhimin Liang
# Date: April 9th
# 
# test the trained network

# import statements
import os
import sys
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from model import MyNetwork  
import cv2
import numpy as np
import torch.nn.functional as F

def loadImages(folder_path):
    images = []
    labels = []

    # list image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort() 

    # read images
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        print(f"Trying to load: {img_path}") 
                
        try:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                class_name = os.path.splitext(filename)[0]
                labels.append(class_name)  
                print(f"Loaded image: {filename} with label: {class_name}")  
            else:
                print(f"Failed to load image: {filename}")  
        except Exception as e:
            print(f"Error loading image {filename}: {e}")  

    print(f"Number of images loaded: {len(images)}")  

    return images, labels

def testModel(model):
    hand_images = []

    # read the hand gesture images
    # folder_path = './hand_test/'
    folder_path = './hand_test_1/'

    # check if folder exists
    if os.path.exists(folder_path):
        print(f"Folder {folder_path} exists.")
        images, labels = loadImages(folder_path)
    else:
        print(f"Folder {folder_path} does not exist.")
        return

    for img in images:
        # convert image to grayscale
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        
        # calculate the aspect ratio
        aspect_ratio = img_gray.shape[1] / img_gray.shape[0]
        new_size = (128, int(128 * aspect_ratio))

        # resize the image using bilinear interpolation
        img_intensity = cv2.resize(img_gray, new_size, interpolation=cv2.INTER_LINEAR)

        # crop the image to make it 128x128
        crop_height = min(128, img_intensity.shape[0])
        crop_width = min(128, img_intensity.shape[1])
        
        start_height = (img_intensity.shape[0] - crop_height) // 2
        start_width = (img_intensity.shape[1] - crop_width) // 2

        end_height = start_height + crop_height
        end_width = start_width + crop_width
        img_cropped = img_intensity[start_height:end_height, start_width:end_width]
        
        # normalize and append
        img_normalized = img_cropped.astype(np.float32) / 255.0
        hand_images.append(img_normalized)

    # convert list to single numpy array and then to PyTorch tensor
    hand_tensor = torch.tensor(np.stack(hand_images))
    hand_tensor = hand_tensor.unsqueeze(1) 
    
    # ensure tensor has the correct dimensions
    expected_shape = (hand_tensor.size(0), 1, 128, 128)
    if hand_tensor.shape != expected_shape:
        print(f"Reshaping tensor from {hand_tensor.shape} to {expected_shape}")
        current_shape = hand_tensor.shape
        pad_height = expected_shape[2] - current_shape[2]
        pad_width = expected_shape[3] - current_shape[3]
        
        # apply padding
        hand_tensor = F.pad(hand_tensor, (0, pad_width, 0, pad_height))

    classes = ('Palm', 'l', 'Fist', 'Fist_moved', 'Thumb', 'Index', 'Ok', 'Palm_moved', 'C', 'Down')

    predictions = []
    images_to_show = []

    with torch.no_grad():
        for inputs, label in zip(hand_tensor, labels):  
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            print('Outputs:', ' '.join(['%.2f' % o.item() for o in outputs[0]]))
            print(f'Actual: {label}, Predicted: {classes[predicted.item()]}')

            # append the image and prediction to lists
            images_to_show.append(inputs.squeeze().numpy())
            predictions.append(predicted.item())

    # display the results with predictions
    fig, axs = plt.subplots(4, 3, figsize=(10, 12))
    for i in range(4):
        for j in range(3):
            ax = axs[i, j]
            index = i * 3 + j
            if index < len(images_to_show):
                ax.imshow(images_to_show[index], cmap='gray', interpolation='none')
                ax.set_title(f'Actual: {labels[index]}\n'
                             f' Prediction: {classes[predictions[index]]}')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')

            else:
                ax.axis('off')  

    plt.tight_layout()
    plt.show()

# main function
def main(argv):
    # load the trained model
    model = torch.load('model.pth')
    
    # set model to evaluation mode
    model.eval()

    # test the model
    testModel(model)

    return

if __name__ == "__main__":
    main(sys.argv)
