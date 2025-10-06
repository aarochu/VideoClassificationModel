import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import styled from 'styled-components';

const UploadContainer = styled.div`
  border: 2px dashed ${props => props.$isDragActive ? '#00FF88' : 'rgba(0, 255, 136, 0.3)'};
  border-radius: 20px;
  padding: 50px;
  text-align: center;
  background: ${props => props.$isDragActive 
    ? 'rgba(0, 255, 136, 0.1)' 
    : 'rgba(0, 255, 136, 0.05)'};
  transition: all 0.3s ease;
  cursor: pointer;
  backdrop-filter: blur(10px);
  position: relative;
  overflow: hidden;

  &:hover {
    border-color: #00FF88;
    background: rgba(0, 255, 136, 0.1);
    box-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
    transform: translateY(-2px);
  }

  ${props => props.$isDragActive && `
    box-shadow: 0 0 40px rgba(0, 255, 136, 0.3);
    border-color: #00FF88;
    background: rgba(0, 255, 136, 0.15);
  `}
`;

const UploadIcon = styled.div`
  font-size: 64px;
  margin-bottom: 24px;
  color: #00FF88;
  text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
  animation: float 3s ease-in-out infinite;
  
  @keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
  }
`;

const UploadText = styled.div`
  font-size: 20px;
  color: #EAEAEA;
  margin-bottom: 12px;
  font-weight: 500;
  letter-spacing: 0.5px;
`;

const UploadSubtext = styled.div`
  font-size: 16px;
  color: rgba(234, 234, 234, 0.7);
  margin-bottom: 24px;
  font-weight: 300;
`;

const FileInfo = styled.div`
  background: rgba(0, 200, 255, 0.1);
  border: 1px solid rgba(0, 200, 255, 0.3);
  border-radius: 16px;
  padding: 20px;
  margin-top: 24px;
  text-align: left;
  backdrop-filter: blur(10px);
  box-shadow: 0 0 20px rgba(0, 200, 255, 0.1);
`;

const FileName = styled.div`
  font-weight: 500;
  color: #00C8FF;
  margin-bottom: 8px;
  font-size: 16px;
  letter-spacing: 0.5px;
`;

const FileSize = styled.div`
  font-size: 14px;
  color: rgba(234, 234, 234, 0.6);
  font-weight: 300;
`;

const SupportedFormats = styled.div`
  font-size: 14px;
  color: rgba(234, 234, 234, 0.5);
  margin-top: 16px;
  font-weight: 300;
  letter-spacing: 0.3px;
`;

function VideoUploader({ onFileSelect, selectedFile }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    },
    multiple: false,
    maxSize: 100 * 1024 * 1024 // 100MB limit
  });

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div>
      <UploadContainer {...getRootProps()} $isDragActive={isDragActive}>
        <input {...getInputProps()} />
        <UploadIcon>📹</UploadIcon>
        <UploadText>
          {isDragActive ? 'Drop the video here...' : 'Drag & drop a video file here'}
        </UploadText>
        <UploadSubtext>or click to select a file</UploadSubtext>
        <SupportedFormats>
          Supported formats: MP4, AVI, MOV, MKV, WMV, FLV (max 100MB)
        </SupportedFormats>
      </UploadContainer>

      {selectedFile && (
        <FileInfo>
          <FileName>📁 {selectedFile.name}</FileName>
          <FileSize>Size: {formatFileSize(selectedFile.size)}</FileSize>
        </FileInfo>
      )}
    </div>
  );
}

export default VideoUploader;
