import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from preprocess import ImageDataManager

MODEL_ROOT_DIR = os.path.dirname(__file__)

DATA_ROOT_DIR = os.path.join(MODEL_ROOT_DIR, 'data')
CHART_OUTPUT_DIR = os.path.join(MODEL_ROOT_DIR, 'charts')
MODEL_STATE_DICT_FILE = os.path.join(MODEL_ROOT_DIR, 'model.pth')

BATCH_SIZE = 64
SQUARE_IMAGE_SIZE = 100
TRAINING_EPOCHS = 10
CONVOLUTION_KERNEL_SIZE = 3
DEBUG_PRINT = False


class DebugPrintSize(nn.Module):

    def __init__(self, label):
        super(DebugPrintSize, self).__init__()

        self.label = label

    def forward(self, x):

        if DEBUG_PRINT:
            print(f'{self.label}: {x.size()}')
        
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Initialize the convolutional kernel size and padding.
        self.kernel_size = CONVOLUTION_KERNEL_SIZE
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


class Model(object):

    def __init__(self):

        self.net = Net()
        self.device = self._init_device()
        self.idm = ImageDataManager(data_root=DATA_ROOT_DIR, square_image_size=SQUARE_IMAGE_SIZE)
        self.data = self._init_data()
        self.net.to(self.device)
    
    def _init_device(self):

        return torch.device('cpu')

        # Setup the device
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _init_data(self):
        (
            train_loader, train_size,
            validate_loader, validate_size,
            test_loader, test_size
        ) = self.idm.split_loaders(train_split=0.8,
                              validate_split=0.1,
                              batch_size=BATCH_SIZE,
                              random_seed=1)
        
        return {
            'train': train_loader,
            'train-size': train_size,
            'validate': validate_loader,
            'validate-size': validate_size,
            'test': test_loader,
            'test-size': test_size
        }

    def load(self, pth_file):
        self.net.load_state_dict(torch.load(pth_file, map_location="cpu"))
        self.net.to(self.device)

    def evaluate(self):
            correct = []

            self.net.eval()
            accuracy = 0
            for inputs, labels in tqdm(self.data['test']):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                accuracy += (predicted == labels).sum().item()
                correct.append((predicted == labels).tolist())

            percentage = round((100 * accuracy / self.data['test-size']), 2)

            print(f"Accuracy of the network on the test images: {percentage}%")
    
    def train(self, pth_file, make_charts):

        print(f"SPLITS: (train={self.data['train-size']}, validate={self.data['validate-size']}, test={self.data['test-size']})")

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        self.net.train()
        losses = []

        for epoch in range(1, TRAINING_EPOCHS + 1):
            print("EPOCH: ", epoch)
            running_loss = 0.0
            for data in tqdm(self.data['train']):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            self.evaluate()
            self.net.train()

            epoch_loss = running_loss / len(self.data['train'])
            print(f"Epoch {epoch + 1}, Training loss: {epoch_loss}")
            losses.append(epoch_loss)

        print('Finished Training')
        torch.save(self.net.state_dict(), pth_file)

        if make_charts:

            loss_plot_fig, loss_plot_ax = plt.subplots()
            loss_plot_x = torch.arange(0, TRAINING_EPOCHS, 1)
            loss_plot_ax.plot(loss_plot_x, losses, c='blue')
            loss_plot_ax.set(title='Cross-Entropy Loss By Epoch', xlabel='Epoch', ylabel='Cross-Entropy Loss')

            plt.style.use('default')
            loss_plot_fig.savefig(f'{CHART_OUTPUT_DIR}/loss_by_epoch_light.png')

            plt.style.use('dark_background')
            loss_plot_fig.savefig(f'{CHART_OUTPUT_DIR}/loss_by_epoch_dark.png')

    def predict(self, images, single_image=False):

        images = self.idm.transform(images)

        if single_image:
            images = images.unsqueeze(0)  # Add batch dimension
        
        images.to(self.device)
        self.net.to(self.device)

        self.net.eval()
        outputs = self.net(images)
        _, predicted = torch.max(outputs.data, 1)

        print(predicted.item(), self.idm.label_map()[predicted.item()])
        
        # Convert the predicted index to a class label
        return predicted.item(), self.idm.label_map()[predicted.item()]



if __name__ == '__main__':

    model = Model()
    print(f'Using device: {model.device}')

    if os.path.isfile(MODEL_STATE_DICT_FILE):
        print(f'Found model state dictionary "{MODEL_STATE_DICT_FILE}", skipping training.')
    else:
        print(f'Training and saving state dictionary into "{MODEL_STATE_DICT_FILE}".')
        model.train(pth_file=MODEL_STATE_DICT_FILE, make_charts=True)


    # Test the model
    model.load(pth_file=MODEL_STATE_DICT_FILE)

    print("\033[0;1;34m=== FINAL RESULTS ===\033[0m")
    model.evaluate()
