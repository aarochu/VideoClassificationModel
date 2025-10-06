import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import styled from 'styled-components';

const UploadContainer = styled.div`
  border: 2px dashed ${props => props.$isDragActive ? '#667eea' : '#ddd'};
  border-radius: 12px;
  padding: 40px;
  text-align: center;
  background: ${props => props.$isDragActive ? '#f8f9ff' : '#fafafa'};
  transition: all 0.3s ease;
  cursor: pointer;

  &:hover {
    border-color: #667eea;
    background: #f8f9ff;
  }
`;

const UploadIcon = styled.div`
  font-size: 48px;
  margin-bottom: 20px;
  color: #667eea;
`;

const UploadText = styled.div`
  font-size: 18px;
  color: #333;
  margin-bottom: 10px;
  font-weight: 500;
`;

const UploadSubtext = styled.div`
  font-size: 14px;
  color: #666;
  margin-bottom: 20px;
`;

const FileInfo = styled.div`
  background: #e8f2ff;
  border: 1px solid #b3d9ff;
  border-radius: 8px;
  padding: 15px;
  margin-top: 20px;
  text-align: left;
`;

const FileName = styled.div`
  font-weight: 600;
  color: #333;
  margin-bottom: 5px;
`;

const FileSize = styled.div`
  font-size: 12px;
  color: #666;
`;

const SupportedFormats = styled.div`
  font-size: 12px;
  color: #888;
  margin-top: 10px;
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
