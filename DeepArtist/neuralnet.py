import os
from collections import OrderedDict
from tqdm import tqdm

from matplotlib import pyplot as plt
from preprocess import ImageDataManager

import torch
import torch.nn as nn
import torch.optim as optim

MODEL_STATE_DICT_FILE = 'model3.pth'
BATCH_SIZE = 64
SQUARE_IMAGE_SIZE = 100
TRAINING_EPOCHS = 10
<<<<<<< HEAD
DEBUG_PRINT_SIZE = False

=======
>>>>>>> 506d89d3cb8eaa5793a3e67a1d087c21ece69da5

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

<<<<<<< HEAD
=======
print(f'Using device: {device}')

>>>>>>> 506d89d3cb8eaa5793a3e67a1d087c21ece69da5
idm = ImageDataManager(data_root='./Data', square_image_size=SQUARE_IMAGE_SIZE)

(
    train_loader, train_size,
    validate_loader, validate_size,
    test_loader, test_size
) = idm.split_loaders(train_split=0.8, validate_split=0.1, batch_size=BATCH_SIZE, random_seed=1)

print(f"SIZES: (train={train_size}, validate={validate_size}, test={test_size})")


<<<<<<< HEAD
class DebugPrintSize(nn.Module):

    def __init__(self, label):
        super(DebugPrintSize, self).__init__()

        self.label = label
    
    
    def forward(self, x):

        if DEBUG_PRINT_SIZE:
            print(f'{self.label}: {x.size()}')
        
        return x


=======
>>>>>>> 506d89d3cb8eaa5793a3e67a1d087c21ece69da5
class Net(nn.Module):

    def __init__(self, kernel_size=3):
        super(Net, self).__init__()

        # Initialize the convolutional kernel size and padding.
        self.kernel_size = kernel_size
        self.padding_size = self.kernel_size // 2

        # Create reusable components.
        self.pool = nn.MaxPool2d(2, 2)
        self.inner_activation = nn.ReLU()
        self.final_activation = nn.LogSoftmax(dim=1)

        # Assemble the two main sections of the network.
        self.build_convolutional_layers()
        self.build_flat_layers()

    def build_convolutional_layers(self):
        self.convolution_model = nn.Sequential(OrderedDict([

<<<<<<< HEAD
            ('print_size1',       DebugPrintSize('Step 0')),
            ('conv1',             nn.Conv2d(3, 16, self.kernel_size, padding=self.padding_size)),
            ('print_size2',       DebugPrintSize('Step 1')),
            ('inner_activation1', self.inner_activation),
            ('print_size3',       DebugPrintSize('Step 2')),
            ('pool1',             self.pool),

            ('print_size4',       DebugPrintSize('Step 3')),
            ('conv2',             nn.Conv2d(16, 32, self.kernel_size, padding=self.padding_size)),
            ('print_size5',       DebugPrintSize('Step 4')),
            ('inner_activation2', self.inner_activation),
            ('print_size6',       DebugPrintSize('Step 5')),
            ('pool2',             self.pool),
            ('print_size7',       DebugPrintSize('Step 6')),

            ('print_size8',       DebugPrintSize('Step 7')),
            ('conv3',             nn.Conv2d(32, 64, self.kernel_size, padding=self.padding_size)),
            ('print_size9',       DebugPrintSize('Step 8')),
            ('inner_activation3', self.inner_activation),
            ('print_size10',      DebugPrintSize('Step 9')),
            ('pool3',             self.pool),

            ('print_size11',      DebugPrintSize('Step 10')),
            ('conv4',             nn.Conv2d(64, 128, self.kernel_size, padding=self.padding_size)),
            ('print_size12',      DebugPrintSize('Step 11')),
            ('inner_activation4', self.inner_activation),
            ('print_size13',      DebugPrintSize('Step 12')),
            ('pool4',             self.pool),

            ('print_size14',      DebugPrintSize('Step 13')),
            ('conv5',             nn.Conv2d(128, 256, self.kernel_size, padding=self.padding_size)),
            ('print_size15',      DebugPrintSize('Step 14')),
            ('inner_activation5', self.inner_activation),
            ('print_size16',      DebugPrintSize('Step 15')),
            ('pool5',             self.pool),
            
            ('print_size17',      DebugPrintSize('Step 16'))
=======
            ('conv1', nn.Conv2d(3, 16, self.kernel_size, padding=self.padding_size)),
            ('inner_activation1', self.inner_activation),
            ('pool1', self.pool),

            ('conv2', nn.Conv2d(16, 32, self.kernel_size, padding=self.padding_size)),
            ('inner_activation2', self.inner_activation),
            ('pool2', self.pool),

            ('conv3', nn.Conv2d(32, 64, self.kernel_size, padding=self.padding_size)),
            ('inner_activation3', self.inner_activation),
            ('pool3', self.pool),

            ('conv4', nn.Conv2d(64, 128, self.kernel_size, padding=self.padding_size)),
            ('inner_activation4', self.inner_activation),
            ('pool4', self.pool),

            # Remove the last CNN layer
            # ('conv5', nn.Conv2d(128, 256, self.kernel_size, padding=self.padding_size)),
            # ('inner_activation5', self.inner_activation),
            # ('pool5', self.pool),
>>>>>>> 506d89d3cb8eaa5793a3e67a1d087c21ece69da5
        ]))


    def build_flat_layers(self):
        # The flat part of the model consists of three linear layers.
        self.flat_model = nn.Sequential(OrderedDict([
<<<<<<< HEAD
            
            ('print_size1',       DebugPrintSize('Step 17')),
            ('linear1',           nn.Linear(2304, 512)),
            ('print_size2',       DebugPrintSize('Step 18')),
            ('inner_activation1', self.inner_activation),

            ('print_size3',       DebugPrintSize('Step 19')),
            ('linear2',           nn.Linear(512, 256)),
            ('print_size4',       DebugPrintSize('Step 20')),
            ('inner_activation2', self.inner_activation),

            ('print_size5',       DebugPrintSize('Step 21')),
            ('linear3',           nn.Linear(256, 5)),
            ('print_size6',       DebugPrintSize('Step 22')),
            ('final_activation3', self.final_activation),

            ('print_size7',       DebugPrintSize('Step 23'))
=======

            # Adjust the input size based on the removed CNN layer
            ('linear1', nn.Linear(128 * 6 * 6, 512)),
            ('inner_activation1', self.inner_activation),

            ('linear2', nn.Linear(512, 256)),
            ('inner_activation2', self.inner_activation),

            ('linear3', nn.Linear(256, 5)),
            ('final_activation3', self.final_activation)
>>>>>>> 506d89d3cb8eaa5793a3e67a1d087c21ece69da5
        ]))

    def forward(self, x):
        # Perform a pass through the convolutional layers, then flatten the data, then pass through
        # the flat layers.
        x = self.convolution_model(x)
<<<<<<< HEAD
        x = x.view(-1, 2304)
=======
        x = x.view(-1, 128 * 6 * 6)  # Adjusted input size based on the removed CNN layer
>>>>>>> 506d89d3cb8eaa5793a3e67a1d087c21ece69da5
        x = self.flat_model(x)

        # Yield the result after having passed through the entire model.
        return x


net = Net(kernel_size=3)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

losses = []

def train():
    correct = []

    net.eval()
    accuracy = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        accuracy += (predicted == labels).sum().item()
        correct.append((predicted == labels).tolist())

    print('Accuracy of the network on the test images: %d %%' % (100 * accuracy / test_size))



if os.path.isfile(MODEL_STATE_DICT_FILE):
    print(f'Found model state dictionary "{MODEL_STATE_DICT_FILE}", skipping training.')
else:
    print(f'Training and saving state dictionary into "{MODEL_STATE_DICT_FILE}".')

    net.train()

    for epoch in range(1, TRAINING_EPOCHS + 1):
        print("EPOCH: ", epoch)
        running_loss = 0.0
        for data in tqdm(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training loss: {epoch_loss}")
        losses.append(epoch_loss)

    print('Finished Training')
    torch.save(net.state_dict(), MODEL_STATE_DICT_FILE)

<<<<<<< HEAD
    plt.style.use('dark_background')

    loss_plot_fig, loss_plot_ax = plt.subplots()
    loss_plot_x = torch.arange(0, TRAINING_EPOCHS, 1)
    loss_plot_ax.plot(loss_plot_x, losses, c='blue')
    loss_plot_ax.set(title='Cross-Entropy Loss By Epoch', xlabel='Epoch', ylabel='Cross-Entropy Loss')
    loss_plot_fig.savefig(f'loss_by_epoch.png')


=======
>>>>>>> 506d89d3cb8eaa5793a3e67a1d087c21ece69da5
# Test the model
net.load_state_dict(torch.load(MODEL_STATE_DICT_FILE, map_location="cpu"))
net.to(device)

correct = []

net.eval()
accuracy = 0
<<<<<<< HEAD
for inputs, labels in tqdm(test_loader):
	inputs, labels = inputs.to(device), labels.to(device)
	outputs = net(inputs)
	_, predicted = torch.max(outputs.data, 1)
	accuracy += (predicted == labels).sum().item()
	correct.append((predicted == labels).tolist())

print('Accuracy of the network on the test images: %d %%' % (100 * accuracy / test_size))

=======
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    accuracy += (predicted == labels).sum().item()
    correct.append((predicted == labels).tolist())

print('Accuracy of the network on the test images: %d %%' % (100 * accuracy / test_size))

plt.style.use('dark_background')

loss_plot_fig, loss_plot_ax = plt.subplots()
loss_plot_x = torch.arange(0, TRAINING_EPOCHS, 1)
loss_plot_ax.plot(loss_plot_x, losses, c='blue')
loss_plot_ax.set(title='Cross-Entropy Loss By Epoch', xlabel='Epoch', ylabel='Cross-Entropy Loss')
loss_plot_fig.savefig(f'loss_by_epoch.png')

>>>>>>> 506d89d3cb8eaa5793a3e67a1d087c21ece69da5
plt.show()
