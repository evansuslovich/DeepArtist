import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

'''
In this file you will write end-to-end code to train a neural network to categorize fashion-mnist data
'''


'''
PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted.
'''

transform = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((224, 224), antialias=True),  # Explicitly set antialias to True
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust if images are grayscale
])

batch_size = 64

'''
PART 2:
Load the dataset. Make sure to utilize the transform and batch_size you wrote in the last section.
'''

trainset = ImageFolder(root='./Data/', transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = ImageFolder(root='./Data/', transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


'''
PART 3:
Design a multi layer perceptron. Since this is a purely Feedforward network, you mustn't use any convolutional layers
Do not directly import or copy any existing models.
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 224 * 224, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


'''
PART 4:
Choose a good loss function and optimizer
'''

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


'''
PART 5:
Train your model!
'''

num_epochs = 5
losses = []  

for epoch in range(num_epochs):
    print("EPOCH:", epoch)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader)
    print(f"Epoch {epoch+1}, Training loss: {epoch_loss}")
    losses.append(epoch_loss) 



print('Finished Training')


'''
PART 6:
Evalute your model! Accuracy should be greater or equal to 80%

'''

correctImage = None
wrongImage = None
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if correctImage is None or wrongImage is None:
            for i in range(labels.size(0)):
                if correctImage is None and predicted[i] == labels[i]:
                    correctImage = images[i]
                    correctLabel = labels[i]
                elif wrongImage is None and predicted[i] != labels[i]:
                    wrongImage = images[i]
                    wrongLabel = labels[i]
                if correctImage is not None and wrongImage is not None:
                    break
            
def imshow(img):
    img = img / 2 + 0.5    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

print('A correct example:')
imshow(torchvision.utils.make_grid(correctImage))
print('Correct Label:', correctLabel.item())

print('A wrong example:')
imshow(torchvision.utils.make_grid(wrongImage))
print('Predicted Label:', wrongLabel.item())

print('Accuracy: %f %%' % (100 * correct / total))


'''
PART 7:
Check the written portion. You need to generate some plots. 
'''
plt.plot(range(num_epochs), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()
