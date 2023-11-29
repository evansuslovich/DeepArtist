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

app = Flask(__name__)
CORS(app)

output_mapping = {
    0: "T-shirt/Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

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
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = net(image)
            _, predicted = torch.max(output.data, 1)
        
        # Convert the predicted index to a class label
        class_label = str(predicted.item())

        return jsonify({'prediction': output_mapping[int(class_label)]})

    else:
        return jsonify({'error': 'Error'})

# PART 1: Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Grayscale(1), # Merge all 3 channels (that should already be equal) to grayscale using mean.
    transforms.Normalize(torch.tensor([0]), torch.tensor([0.5]))
])
batch_size = 64

# PART 2: Load the dataset. Make sure to utilize the transform and batch_size.
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# PART 3: Design a multi-layer perceptron.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    # bias terms: 
    # ((784 * 512) + 512) + ((512 * 256)  + 256) + ((256 * 10) + 10)
    def forward(self, x):
        print(1, x.size)
        x = x.view(-1, 28 * 28)  # Flatten the input
        print(2, len(x))
        x = F.relu(self.fc1(x)) # run through relu 
        print(3, len(x))
        x = F.relu(self.fc2(x)) # run through relu 
        print(4, len(x))
        x = self.fc3(x)
        print(5, len(x))
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
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        training_losses.append(running_loss / len(trainloader))
        print(f"Epoch {epoch + 1}, Training loss: {training_losses[-1]}")

    torch.save(net.state_dict(), "model.pth")
    print('Finished Training')

def evaluate_model():
    # PART 6: Evaluate your model!
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
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
    print("Model Loaded from './model_training.pth'")
    evaluate_model()
except Exception as e: 
    print(e)
    print("Training Model")
    train_model()
    print("Evaluating Model")
    evaluate_model()

if __name__ == '__main__':
    app.run(debug=True)
