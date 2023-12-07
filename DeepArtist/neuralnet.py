import os
from collections import OrderedDict
from tqdm import tqdm

import matplotlib.pyplot as plt
from preprocess import ImageDataManager

import torch
import torch.nn as nn
import torch.optim as optim

MODEL_STATE_DICT_FILE = 'model.pth'
BATCH_SIZE = 64
SQUARE_IMAGE_SIZE = 100
TRAINING_EPOCHS = 10
DEBUG_PRINT_SIZE = False


if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')


idm = ImageDataManager(data_root='./Data', square_image_size=SQUARE_IMAGE_SIZE)

(
    train_loader, train_size,
    validate_loader, validate_size,
    test_loader, test_size
) = idm.split_loaders(train_split=0.8, validate_split=0.1, batch_size=BATCH_SIZE, random_seed=1)

print(f"SIZES: (train={train_size}, validate={validate_size}, test={test_size})")


class DebugPrintSize(nn.Module):

    def __init__(self, label):
        super(DebugPrintSize, self).__init__()

        self.label = label

    def forward(self, x):

        if DEBUG_PRINT_SIZE:
            print(f'{self.label}: {x.size()}')
        
        return x


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
        self.convolution_model = self.build_convolutional_layers()
        self.flat_model = self.build_flat_layers()

    def build_convolutional_layers(self):

        # The convolutional part of the model consists of 5 layers.
        return nn.Sequential(OrderedDict([

            ('conv1',             nn.Conv2d(3, 16, self.kernel_size, padding=self.padding_size)),
            ('inner_activation1', self.inner_activation),
            ('pool1',             self.pool),

            ('conv2',             nn.Conv2d(16, 32, self.kernel_size, padding=self.padding_size)),
            ('inner_activation2', self.inner_activation),
            ('pool2',             self.pool),

            ('conv3',             nn.Conv2d(32, 64, self.kernel_size, padding=self.padding_size)),
            ('inner_activation3', self.inner_activation),
            ('pool3',             self.pool),

            ('conv4',             nn.Conv2d(64, 128, self.kernel_size, padding=self.padding_size)),
            ('inner_activation4', self.inner_activation),
            ('pool4',             self.pool),

            ('conv5',             nn.Conv2d(128, 256, self.kernel_size, padding=self.padding_size)),
            ('inner_activation5', self.inner_activation),
            ('pool5',             self.pool),
        ]))

    def build_flat_layers(self):

        # The flat part of the model consists of three layers.
        return nn.Sequential(OrderedDict([

            ('linear1',           nn.Linear(2304, 512)),
            ('inner_activation1', self.inner_activation),

            ('linear2',           nn.Linear(512, 256)),
            ('inner_activation2', self.inner_activation),

            ('linear3',           nn.Linear(256, 5)),
            ('final_activation3', self.final_activation)
        ]))

    def forward(self, x):
        # Perform a pass through the convolutional layers, then flatten the data, then pass through
        # the flat layers.
        x = self.convolution_model(x)
        x = x.view(-1, 2304)
        x = self.flat_model(x)

        # Yield the result after having passed through the entire model.
        return x


net = Net(kernel_size=3)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

losses = []


def evaluate():
    correct = []

    net.eval()
    accuracy = 0
    for inputs, labels in tqdm(test_loader):
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

        evaluate()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training loss: {epoch_loss}")
        losses.append(epoch_loss)

    print('Finished Training')
    torch.save(net.state_dict(), MODEL_STATE_DICT_FILE)

    plt.style.use('dark_background')
    loss_plot_fig, loss_plot_ax = plt.subplots()
    loss_plot_x = torch.arange(0, TRAINING_EPOCHS, 1)
    loss_plot_ax.plot(loss_plot_x, losses, c='blue')
    loss_plot_ax.set(title='Cross-Entropy Loss By Epoch', xlabel='Epoch', ylabel='Cross-Entropy Loss')
    loss_plot_fig.savefig(f'loss_by_epoch.png')


# Test the model
net.load_state_dict(torch.load(MODEL_STATE_DICT_FILE, map_location="cpu"))
net.to(device)

print("\033[0;1;34m=== FINAL RESULTS ===\033[0m")
evaluate()
