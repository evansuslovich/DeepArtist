import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './App.css';

const App: React.FC = () => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    // Do something with the dropped files
    console.log('Dropped files:', acceptedFiles);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div className="App">
      <header className="App-header">
        <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
          <input {...getInputProps()} />
          {isDragActive ? <p>Drop the files here...</p> : <p>Drag a file here, or click to select one</p>}
        </div>
        <p>
          Hi there! Welcome to DeepArtist
        </p>
      </header>
    </div>
  );
};

export default App;
