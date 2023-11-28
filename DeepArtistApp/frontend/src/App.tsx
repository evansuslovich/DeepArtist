import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './App.css';
import axios from 'axios';

const App: React.FC = () => {
  
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];

    const formData = new FormData();
    formData.append('file', file);

    console.log(formData.get('file'))
    
    axios({
      url: 'http://127.0.0.1:5000/upload',
      method: "POST",
      data: formData
    }).then((res) => {console.log(res)})
      .catch((err) => {console.log(err)});
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div className="App">
      <header className="App-header">
        
        <p>Hi there! Welcome to DeepArtist.</p>

        {/* Input an Image */}
        <div className="dropFile" {...getRootProps()}>
          <input {...getInputProps()}/>
          {<p>Drag a file here, or click to select one</p>}
        </div>
      </header>
    </div>
  );
};

export default App;
