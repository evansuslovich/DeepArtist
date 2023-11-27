import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './App.css';

const App: React.FC = () => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];

    const formData = new FormData();
    formData.append('file', file);

    // Send the file to the Flask server using fetch or Axios
    fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData,
    })
      .then(response => response.json())
      .then(data => console.log(data))
      .catch(error => console.error('Error uploading file:', error));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div className="App">
      <header className="App-header">
        <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
          <input {...getInputProps()} />
          {isDragActive ? <p>Drop the files here...</p> : <p>Drag a file here, or click to select one</p>}
        </div>
        <p>Hi there! Welcome to DeepArtist.</p>
      </header>
    </div>
  );
};

export default App;
