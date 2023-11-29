// App.tsx
import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import './App.css';
import axios from 'axios';

const App: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [predictedLabel, setPredictedLabel] = useState<string | undefined>();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    setSelectedImage(file);

    const formData = new FormData();
    formData.append('file', file);

    axios({
      url: 'http://127.0.0.1:5000/upload',
      method: 'POST',
      data: formData,
    })
      .then((res) => {
        console.log(res.data);
        setPredictedLabel(res.data.prediction);
      })
      .catch((err) => {
        console.log(err);
      });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div className="App">
      <header className="App-header">
        <p>Hi there! Welcome to DeepArtist.</p>

        {/* Input an Image */}
        <div className="dropFile" {...getRootProps()}>
          <input {...getInputProps()} />
            <p>Drop the files here ...</p>
        </div>

        {selectedImage && (
          <div>
            <h2>Selected Image</h2>
            <img
              src={URL.createObjectURL(selectedImage)}
              alt="Selected"
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        )}

        <h1>We predict this is: </h1>
        <h2>{predictedLabel}</h2>
      </header>
    </div>
  );
};

export default App;
