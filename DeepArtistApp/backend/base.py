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
from deepartist.neuralnet import Model

app = Flask(__name__)
CORS(app)

SQUARE_IMAGE_SIZE = 100
BATCH_SIZE = 64

# Specify the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
# Allow larger files to be uploaded
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB


model = Model()


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
        image = image.unsqueeze(0)  # Add batch dimension
        _, class_label = model.predict(image)

        return jsonify({'prediction': class_label})
    else:
        return jsonify({'error': 'Error'})


if __name__ == '__main__':
    app.run(debug=True)
