import React, { useState, useRef, useCallback } from 'react';
import styled from 'styled-components';

const CameraContainer = styled.div`
  background: #f8f9fa;
  border-radius: 12px;
  padding: 30px;
  text-align: center;
`;

const VideoContainer = styled.div`
  position: relative;
  max-width: 600px;
  margin: 0 auto 20px;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
`;

const Video = styled.video`
  width: 100%;
  height: auto;
  display: block;
`;

const Controls = styled.div`
  display: flex;
  gap: 15px;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 20px;
`;

const Button = styled.button`
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  min-width: 120px;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const RecordButton = styled(Button)`
  background: ${props => props.isRecording ? '#dc3545' : '#28a745'};
  color: white;

  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
  }
`;

const StopButton = styled(Button)`
  background: #dc3545;
  color: white;

  &:hover:not(:disabled) {
    background: #c82333;
    transform: translateY(-2px);
  }
`;

const DownloadButton = styled(Button)`
  background: #007bff;
  color: white;

  &:hover:not(:disabled) {
    background: #0056b3;
    transform: translateY(-2px);
  }
`;

const Status = styled.div`
  font-size: 16px;
  font-weight: 500;
  color: ${props => props.$isRecording ? '#dc3545' : '#28a745'};
  margin-bottom: 15px;
`;

const Timer = styled.div`
  font-size: 24px;
  font-weight: 700;
  color: #333;
  margin-bottom: 15px;
`;

const ErrorMessage = styled.div`
  background: #f8d7da;
  color: #721c24;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 20px;
  border: 1px solid #f5c6cb;
`;

const TroubleshootingSection = styled.div`
  background: #fff3cd;
  color: #856404;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 20px;
  border: 1px solid #ffeaa7;
  text-align: left;
`;

const TroubleshootingTitle = styled.h4`
  margin: 0 0 10px 0;
  color: #856404;
  font-size: 16px;
`;

const TroubleshootingList = styled.ul`
  margin: 0;
  padding-left: 20px;
`;

const TroubleshootingItem = styled.li`
  margin-bottom: 5px;
  font-size: 14px;
`;

const SuccessMessage = styled.div`
  background: #d4edda;
  color: #155724;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 20px;
  border: 1px solid #c3e6cb;
`;

const ClassSelector = styled.div`
  margin-bottom: 20px;
`;

const Input = styled.input`
  padding: 10px 15px;
  border: 2px solid #dee2e6;
  border-radius: 8px;
  font-size: 16px;
  background: white;
  min-width: 200px;
  width: 100%;
  max-width: 300px;

  &:focus {
    outline: none;
    border-color: #667eea;
  }
`;

const Label = styled.label`
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #333;
`;

const RecordingIndicator = styled.div`
  position: absolute;
  top: 15px;
  right: 15px;
  background: #dc3545;
  color: white;
  padding: 8px 12px;
  border-radius: 20px;
  font-size: 14px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const Pulse = styled.div`
  width: 10px;
  height: 10px;
  background: white;
  border-radius: 50%;
  animation: pulse 1s infinite;

  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
`;

function CameraRecorder({ onVideoRecorded }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [recordedVideo, setRecordedVideo] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [className, setClassName] = useState('');
  const [recordingTime, setRecordingTime] = useState(0);
  const [hasPermission, setHasPermission] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      
      console.log('Starting camera...');
      
      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError('Camera access is not supported in this browser. Please use Chrome, Firefox, or Safari.');
        setIsLoading(false);
        return;
      }
      
      // Simple video constraints
      const constraints = {
        video: true,
        audio: false
      };
      
      console.log('Requesting camera access with constraints:', constraints);
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      console.log('Camera stream obtained:', stream);
      streamRef.current = stream;
      
      // Set permission first so video element renders, then set stream
      setHasPermission(true);
      setIsLoading(false);
      console.log('Camera stream obtained, setting permission to render video element');
      
    } catch (err) {
      console.error('Camera error:', err);
      
      let errorMessage = 'Camera access denied. ';
      
      if (err.name === 'NotAllowedError') {
        errorMessage += 'Please allow camera permissions in your browser and refresh the page.';
      } else if (err.name === 'NotFoundError') {
        errorMessage += 'No camera found. Please connect a camera and try again.';
      } else if (err.name === 'NotReadableError') {
        errorMessage += 'Camera is already in use by another application.';
      } else if (err.name === 'OverconstrainedError') {
        errorMessage += 'Camera constraints cannot be satisfied.';
      } else {
        errorMessage += `Error: ${err.message}`;
      }
      
      setError(errorMessage);
      setIsLoading(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setHasPermission(false);
  }, []);

  const startRecording = useCallback(() => {
    if (!streamRef.current) return;

    const mediaRecorder = new MediaRecorder(streamRef.current, {
      mimeType: 'video/webm;codecs=vp9'
    });

    mediaRecorderRef.current = mediaRecorder;
    setRecordedChunks([]);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        setRecordedChunks(prev => [...prev, event.data]);
      }
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      setRecordedVideo({ blob, url });
    };

    mediaRecorder.start();
    setIsRecording(true);
    setRecordingTime(0);

    // Start timer
    timerRef.current = setInterval(() => {
      setRecordingTime(prev => prev + 1);
    }, 1000);
  }, [recordedChunks]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
    }
  }, [isRecording]);

  const downloadVideo = useCallback(() => {
    if (recordedVideo) {
      const a = document.createElement('a');
      a.href = recordedVideo.url;
      a.download = `${className || 'video'}_${Date.now()}.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }, [recordedVideo, className]);

  const saveToDataset = useCallback(() => {
    if (recordedVideo && onVideoRecorded && className.trim()) {
      const file = new File([recordedVideo.blob], `${className.trim()}_${Date.now()}.webm`, {
        type: 'video/webm'
      });
      onVideoRecorded(file, className.trim());
      setSuccess(`Video saved to ${className.trim()} class!`);
      setRecordedVideo(null);
      setRecordedChunks([]);
      setRecordingTime(0);
    } else if (!className.trim()) {
      setError('Please enter a class name before saving the video.');
    }
  }, [recordedVideo, className, onVideoRecorded]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  React.useEffect(() => {
    return () => {
      stopCamera();
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [stopCamera]);

  // Set stream when video element is available
  React.useEffect(() => {
    if (videoRef.current && streamRef.current) {
      console.log('Setting video stream in useEffect');
      videoRef.current.srcObject = streamRef.current;
    }
  }, [hasPermission]); // This will run when hasPermission changes and video element is rendered

  return (
    <CameraContainer>
      <h3>📹 Record Video for Dataset</h3>
      
      {error && (
        <>
          <ErrorMessage>{error}</ErrorMessage>
          <TroubleshootingSection>
            <TroubleshootingTitle>🔧 Troubleshooting Camera Access:</TroubleshootingTitle>
            <TroubleshootingList>
              <TroubleshootingItem>Make sure you're using HTTPS or localhost (required for camera access)</TroubleshootingItem>
              <TroubleshootingItem>Check your browser's camera permissions in settings</TroubleshootingItem>
              <TroubleshootingItem>Close other applications that might be using the camera (Zoom, Skype, etc.)</TroubleshootingItem>
              <TroubleshootingItem>Try refreshing the page and allowing permissions again</TroubleshootingItem>
              <TroubleshootingItem>Make sure your camera is not being used by another browser tab</TroubleshootingItem>
              <TroubleshootingItem>Try using Chrome, Firefox, or Safari (best browser support)</TroubleshootingItem>
            </TroubleshootingList>
          </TroubleshootingSection>
        </>
      )}
      {success && <SuccessMessage>{success}</SuccessMessage>}

      <ClassSelector>
        <Label htmlFor="class-name">Class Name:</Label>
        <Input
          id="class-name"
          type="text"
          placeholder="Enter class name (e.g., PersonA, Walking, etc.)"
          value={className}
          onChange={(e) => setClassName(e.target.value)}
        />
      </ClassSelector>

      {!hasPermission ? (
        <div>
          <Button 
            onClick={startCamera} 
            disabled={isLoading}
            style={{ background: '#667eea', color: 'white', marginBottom: '10px' }}
          >
            {isLoading ? '🔄 Starting Camera...' : '📷 Start Camera'}
          </Button>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '10px' }}>
            Debug: {isLoading ? 'Loading...' : 'Ready to start camera'}
          </div>
        </div>
      ) : (
        <>
          <VideoContainer>
            <Video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              style={{ width: '100%', height: 'auto' }}
            />
            {isRecording && (
              <RecordingIndicator>
                <Pulse />
                REC
              </RecordingIndicator>
            )}
          </VideoContainer>

          <Status $isRecording={isRecording}>
            {isRecording ? '🔴 Recording...' : '📹 Camera Ready'}
          </Status>
          
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '15px' }}>
            Debug: Stream={streamRef.current ? '✅' : '❌'} | Video={videoRef.current ? '✅' : '❌'} | Permission={hasPermission ? '✅' : '❌'}
          </div>

          {isRecording && (
            <Timer>{formatTime(recordingTime)}</Timer>
          )}

          <Controls>
            {!isRecording ? (
              <RecordButton onClick={startRecording}>
                🔴 Start Recording
              </RecordButton>
            ) : (
              <StopButton onClick={stopRecording}>
                ⏹️ Stop Recording
              </StopButton>
            )}

            {recordedVideo && (
              <>
                <DownloadButton onClick={downloadVideo}>
                  💾 Download
                </DownloadButton>
                <Button
                  onClick={saveToDataset}
                  style={{ background: '#28a745', color: 'white' }}
                >
                  📁 Save to Dataset
                </Button>
              </>
            )}

            <Button
              onClick={stopCamera}
              style={{ background: '#6c757d', color: 'white' }}
            >
              📷 Stop Camera
            </Button>
          </Controls>
        </>
      )}
    </CameraContainer>
  );
}

export default CameraRecorder;
