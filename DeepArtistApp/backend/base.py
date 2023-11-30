from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from preprocess import ImageDataManager

app = Flask(__name__)
CORS(app)

SQUARE_IMAGE_SIZE = 100
BATCH_SIZE = 64

idm = ImageDataManager(data_root='../../DeepArtist/Data', square_image_size=SQUARE_IMAGE_SIZE)

(
    train_loader, train_size,
    validate_loader, validate_size,
    test_loader, test_size
) = idm.split_loaders(train_split=0.8, validate_split=0.1, batch_size=BATCH_SIZE, random_seed=1)

# Specify the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
# Allow larger files to be uploaded
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

def allowed_file(filename):
    # Check if the file type is allowed
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

@app.route('/upload', methods=['POST'])
def upload_file():

    # Check if the POST request has a file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is allowed and not empty
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'})

    if file:
        # Save the uploaded image
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Open and preprocess the image for prediction
        image = Image.open(filename)
        image = idm.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = net(image)
            _, predicted = torch.max(output.data, 1)
        
        # Convert the predicted index to a class label
        class_label = str(predicted.item())

        return jsonify({'prediction': idm.label_map()[int(class_label)]})

    else:
        return jsonify({'error': 'Error'})

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

def train_model(): 
    # PART 4: Choose a good loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # PART 5: Train your model!
    num_epochs = 15

    training_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        training_losses.append(running_loss / train_size)
        print(f"Epoch {epoch + 1}, Training loss: {training_losses[-1]}")

    torch.save(net.state_dict(), "model.pth")
    print('Finished Training')

def evaluate_model():
    # PART 6: Evaluate your model!
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy: {:.2f}%'.format(accuracy))


try:
    # Test the model
    net.load_state_dict(torch.load('model.pth', map_location="cpu"))
    # net.to(device)
    print("Model Loaded from './model.pth'")
    evaluate_model()
except Exception as e: 
    print(e)
    print("Training Model")
    train_model()
    print("Evaluating Model")
    evaluate_model()

if __name__ == '__main__':
    app.run(debug=True)
