import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from preprocess import ImageDataManager

if torch.backends.mps.is_available():
    device=torch.device("mps")
elif torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

print("Device:", device)

transform = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((224, 224), antialias=True),  # Explicitly set antialias to True
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust if images are grayscale
])


BATCH_SIZE = 64
SQUARE_IMAGE_SIZE = 100

idm = ImageDataManager(data_root='./Data', square_image_size=SQUARE_IMAGE_SIZE)

(
    train_loader, train_size,
    validate_loader, validate_size,
    test_loader, test_size
) = idm.split_loaders(train_split=0.8, validate_split=0.1, batch_size=BATCH_SIZE, random_seed=1)

print(f"SIZES: (train={train_size}, validate={validate_size}, test={test_size})")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 5)
        self.relu = nn.ReLU()
        self.final_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = x.view(-1, 256 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.final_activation(self.fc3(x))
        return x

net = Net()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


num_epochs = 5
losses = []

for epoch in range(num_epochs):
    print("EPOCH:", epoch)
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training loss: {epoch_loss}")
    losses.append(epoch_loss) 



print('Finished Training')
torch.save(net.state_dict(), "model_letsgooo.pth")


# Test the model
net.load_state_dict(torch.load('model_letsgooo.pth', map_location="cpu"))
net.to(device)

correct=[]

net.eval()
accuracy = 0
for inputs, labels in test_loader:
	inputs, labels = inputs.to(device), labels.to(device)
	outputs = net(inputs)
	_, predicted = torch.max(outputs.data, 1)
	accuracy += (predicted == labels).sum().item()
	correct.append((predicted == labels).tolist())

print('Accuracy of the network on the test images: %d %%' % (100 * accuracy / test_size))
