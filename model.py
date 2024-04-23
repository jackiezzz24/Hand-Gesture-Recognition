# Name: Jiaqi Zhao, Kexuan Chen, Zhimin Liang
# Date: April 9th
# 
# functions to get the data and save the trained model

# import statements
import os
import sys
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class LeapGestRecog(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.x = []
        self.y = []

        folders = os.listdir(root)
        for folder in folders:
            for dirpath, dirnames, filenames in os.walk(os.path.join(root, folder)):
                for filename in filenames:
                    self.x.append(os.path.join(dirpath, filename))
                    self.y.append(int(folder))  
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = Image.open(self.x[index]).convert('L')
        y = self.y[index]
        if self.transform:
            img = self.transform(img)
        return img, y


# define the neural network
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.network_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(in_features=128*12*12, out_features=128)    
        # a final fully connected linear layer with 10 nodes 
        self.fc2 = nn.Linear(in_features=128, out_features=10)   

    # computes a forward pass for the network
    def forward(self, x):
        x = self.network_layers(x)
        # a flattening operation
        x = x.view(-1, 128*12*12)
        # a ReLU function applied on the fully connect linear layer with 50 nodes
        x = F.relu(self.fc1(x))
        # log_softmax function applied to the fully connect linear layer with 10 nodes
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

# train the model for at least five epochs
def train_network( train_loader, test_loader, model, loss_func, optimizer, epochs ):
    train_losses = []
    test_losses = []
    examples_seen = []
    test_x = [i * len(train_loader.dataset) for i in range(epochs)]

    # lopp through epochs
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, examples_seen = train_loop(train_loader, model, loss_func, optimizer, examples_seen, t)
        test_loss = test_loop(test_loader, model, loss_func)

        train_losses.extend(train_loss)
        test_losses.append(test_loss)

    print("Done!")

    # plot training and testing errors and accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(examples_seen, train_losses, label='Train Loss', color='blue')
    plt.scatter(test_x, test_losses, label='Test Loss', color='red')
    plt.xlabel('number of training expamles seen')
    plt.ylabel('negative log likelihood loss')
    plt.legend()
    plt.show()

# loop over optimization code
def train_loop(dataloader, model, loss_fn, optimizer, example_size, epochId):
    size = len(dataloader.dataset)
    # set the model to training mode
    model.train()
    total_loss = []
    batch_size = 64

    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            total_loss.append(loss)
            example_size.append(batch * len(X) + epochId * size)

    return total_loss, example_size

# evaluates the model's performance against the test data
def test_loop(dataloader, model, loss_fn):
    # set the model to evaluation mode 
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

# main function 
def main(argv):
    
    # get data set 
    root = './leapGestRecog/'
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = LeapGestRecog(root, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    learning_rate = 1e-2    
    batch_size = 64
    epochs = 10

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    # instantiate the network, loss function, and optimizer
    model = MyNetwork()
    loss_func = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_network(train_loader, test_loader, model, loss_func, optimizer, epochs)

    # save the network to a file
    torch.save(model, 'model.pth')

    return

if __name__ == "__main__":
    main(sys.argv)