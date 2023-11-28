from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Specify the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
# Allow larger files to be uploaded
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

def allowed_file(filename):
    # Check if the file type is allowed
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

@app.route('/upload', methods=['POST'])
def upload_file():
    print(request.data)
    print(request.form)
    print(request)

    return jsonify({'message': 'File Uploaded Succesfully'})
    # # Check if the POST request has a file part
    # if 'file' not in request.files:
    #     return jsonify({'error': 'No file part'})

    # file = request.files['file']

    # # Check if the file is allowed and not empty
    # if file.filename == '' or not allowed_file(file.filename):
    #     return jsonify({'error': 'Invalid file'})

    # # Save the file to the upload folder
    # if file:
    #     filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #     file.save(filename)
    #     return jsonify({'message': 'File uploaded successfully'})


if __name__ == '__main__':
    app.run(debug=True)
