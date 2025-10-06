import React, { useState, useRef, useCallback, useEffect } from 'react';
import styled from 'styled-components';
import { detectFacesInImage, getFaceDetectorStatus, resetFaceTracker } from '../services/api';

const SmartFaceContainer = styled.div`
  background: transparent;
  border-radius: 20px;
  padding: 40px;
  text-align: center;
  width: 100%;
  max-width: 1000px;
`;

const VideoContainer = styled.div`
  position: relative;
  max-width: 800px;
  margin: 0 auto 30px;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 
    0 0 40px rgba(0, 255, 136, 0.2),
    0 0 80px rgba(0, 200, 255, 0.1),
    inset 0 0 20px rgba(0, 255, 136, 0.1);
  background: #000;
  border: 2px solid rgba(0, 255, 136, 0.3);
  transition: all 0.3s ease;
  
  &:hover {
    box-shadow: 
      0 0 60px rgba(0, 255, 136, 0.3),
      0 0 120px rgba(0, 200, 255, 0.15),
      inset 0 0 30px rgba(0, 255, 136, 0.15);
    border-color: rgba(0, 255, 136, 0.5);
  }
`;

const Video = styled.video`
  width: 100%;
  height: auto;
  display: block;
  border-radius: 18px;
`;

const Canvas = styled.canvas`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  border-radius: 18px;
`;

const Controls = styled.div`
  display: flex;
  gap: 20px;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 30px;
`;

const Button = styled.button`
  padding: 16px 32px;
  border: 2px solid rgba(0, 255, 136, 0.3);
  border-radius: 12px;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  min-width: 140px;
  background: rgba(0, 255, 136, 0.1);
  color: #00FF88;
  backdrop-filter: blur(10px);
  position: relative;
  overflow: hidden;

  &:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  &:hover:not(:disabled) {
    background: rgba(0, 255, 136, 0.2);
    border-color: rgba(0, 255, 136, 0.6);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 255, 136, 0.3);
  }
`;

const StartButton = styled(Button)`
  background: ${props => props.$isActive 
    ? 'rgba(220, 53, 69, 0.2)' 
    : 'rgba(0, 255, 136, 0.2)'};
  border-color: ${props => props.$isActive 
    ? 'rgba(220, 53, 69, 0.6)' 
    : 'rgba(0, 255, 136, 0.6)'};
  color: ${props => props.$isActive ? '#FF6B6B' : '#00FF88'};
  
  ${props => props.$isActive && `
    box-shadow: 0 0 30px rgba(220, 53, 69, 0.4);
    animation: pulse-red 2s infinite;
  `}
  
  &:hover:not(:disabled) {
    background: ${props => props.$isActive 
      ? 'rgba(220, 53, 69, 0.3)' 
      : 'rgba(0, 255, 136, 0.3)'};
    box-shadow: ${props => props.$isActive 
      ? '0 8px 25px rgba(220, 53, 69, 0.4)' 
      : '0 8px 25px rgba(0, 255, 136, 0.4)'};
  }
  
  @keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 30px rgba(220, 53, 69, 0.4); }
    50% { box-shadow: 0 0 50px rgba(220, 53, 69, 0.6); }
  }
`;

const Status = styled.div`
  font-size: 18px;
  font-weight: 400;
  color: ${props => props.$isActive ? '#FF6B6B' : '#00FF88'};
  margin-bottom: 20px;
  text-transform: uppercase;
  letter-spacing: 1px;
  text-shadow: 0 0 10px ${props => props.$isActive ? 'rgba(255, 107, 107, 0.5)' : 'rgba(0, 255, 136, 0.5)'};
`;

const FaceDisplay = styled.div`
  background: rgba(0, 255, 136, 0.05);
  border: 1px solid rgba(0, 255, 136, 0.2);
  border-radius: 16px;
  padding: 30px;
  margin-top: 30px;
  box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
  backdrop-filter: blur(10px);
  text-align: left;
  width: 100%;
  max-width: 600px;
`;

const FaceItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 0;
  border-bottom: 1px solid rgba(0, 255, 136, 0.1);

  &:last-child {
    border-bottom: none;
  }
`;

const PersonName = styled.span`
  font-weight: 500;
  color: #EAEAEA;
  font-size: 18px;
  letter-spacing: 0.5px;
`;

const Confidence = styled.span`
  color: #00FF88;
  font-weight: 600;
  font-size: 16px;
  text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
`;

const ErrorMessage = styled.div`
  background: rgba(255, 107, 107, 0.1);
  color: #FF6B6B;
  padding: 20px;
  border-radius: 12px;
  margin-bottom: 20px;
  border: 1px solid rgba(255, 107, 107, 0.3);
  backdrop-filter: blur(10px);
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 2px solid rgba(0, 255, 136, 0.3);
  border-radius: 50%;
  border-top-color: #00FF88;
  animation: spin 1s ease-in-out infinite;
  margin-right: 8px;

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
`;

const LoadingText = styled.span`
  display: flex;
  align-items: center;
  justify-content: center;
`;

const StatsDisplay = styled.div`
  background: rgba(0, 200, 255, 0.05);
  border: 1px solid rgba(0, 200, 255, 0.2);
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 30px;
  text-align: center;
  backdrop-filter: blur(10px);
  box-shadow: 0 0 20px rgba(0, 200, 255, 0.1);
`;

const StatItem = styled.div`
  display: inline-block;
  margin: 0 20px;
  font-size: 14px;
  color: #00C8FF;
  font-weight: 500;
  letter-spacing: 0.5px;
`;

function SmartFaceDetector() {
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState(null);
  const [faces, setFaces] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);
  const [detectorStatus, setDetectorStatus] = useState(null);
  const [stats, setStats] = useState({
    totalFaces: 0,
    uniquePersons: 0,
    detectionCount: 0
  });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);
  const animationRef = useRef(null);

  // Check face detector status on mount
  useEffect(() => {
    const checkDetectorStatus = async () => {
      try {
        const status = await getFaceDetectorStatus();
        setDetectorStatus(status);
        if (!status.available) {
          setError('Face detector is not available. Please check the backend server.');
        }
      } catch (err) {
        console.error('Failed to check detector status:', err);
        setError('Failed to connect to face detection service.');
      }
    };
    
    checkDetectorStatus();
  }, []);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      
      console.log('Starting SmartFace camera...');
      
      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError('Camera access is not supported in this browser. Please use Chrome, Firefox, or Safari.');
        setIsLoading(false);
        return;
      }
      
      // Video constraints for better face detection
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false
      };
      
      console.log('Requesting camera access for SmartFace...');
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      console.log('Camera stream obtained for SmartFace:', stream);
      streamRef.current = stream;
      
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
    setIsActive(false);
    
    // Clear intervals
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  }, []);

  const handleResetTracker = useCallback(async () => {
    try {
      await resetFaceTracker();
      setFaces([]);
      console.log('🔄 Face tracker reset - next detection will be Person 1');
    } catch (error) {
      console.error('Failed to reset face tracker:', error);
      setError('Failed to reset face tracker');
    }
  }, []);

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return null;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to blob
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });
          resolve(file);
        } else {
          resolve(null);
        }
      }, 'image/jpeg', 0.8);
    });
  }, []);

  const detectFaces = useCallback(async () => {
    try {
      if (!videoRef.current || !detectorStatus?.available) {
        console.log('No video ref or detector not available');
        return;
      }

      const frameFile = await captureFrame();
      if (!frameFile) {
        console.log('Failed to capture frame');
        return;
      }

      // Call backend face detection API
      const result = await detectFacesInImage(frameFile);
      
      console.log('Face detection result:', result);
      
      if (result.faces && result.faces.length > 0) {
        setFaces(result.faces);
        
        // Update stats
        setStats(prev => ({
          totalFaces: prev.totalFaces + result.faces.length,
          uniquePersons: Math.max(prev.uniquePersons, result.faces.length),
          detectionCount: prev.detectionCount + 1
        }));
      } else {
        setFaces([]);
      }
      
    } catch (error) {
      console.error('Face detection error:', error);
      // Don't show error for every failed detection to avoid spam
      if (error.message.includes('Face detector not loaded')) {
        setError('Face detection service is not available. Please check the backend.');
      }
    }
  }, [captureFrame, detectorStatus]);

  const startDetection = useCallback(() => {
    if (!hasPermission || !detectorStatus?.available) return;
    
    setIsActive(true);
    
    // Make face detection every 500ms for real-time tracking
    detectionIntervalRef.current = setInterval(() => {
      detectFaces();
    }, 500);
    
    // Make initial detection
    detectFaces();
  }, [hasPermission, detectorStatus, detectFaces]);

  const stopDetection = useCallback(() => {
    setIsActive(false);
    
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    setFaces([]);
  }, []);

  const drawFaceBoxes = useCallback(() => {
    if (!canvasRef.current || !videoRef.current || !isActive) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const video = videoRef.current;
    
    // Get the actual display size of the video element
    const videoRect = video.getBoundingClientRect();
    
    // Set canvas size to match the video display size
    canvas.width = videoRect.width;
    canvas.height = videoRect.height;
    
    // Calculate scale factors for coordinate conversion
    const scaleX = videoRect.width / video.videoWidth;
    const scaleY = videoRect.height / video.videoHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw face detection boxes for each detected face
    faces.forEach((face, index) => {
      if (face.bbox) {
        const { x, y, width, height } = face.bbox;
        
        // Scale coordinates to match display size
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;
        
        // Draw futuristic face box with glow effect
        ctx.strokeStyle = '#00FF88';
        ctx.lineWidth = 3;
        ctx.shadowColor = '#00FF88';
        ctx.shadowBlur = 15;
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
        
        // Draw inner glow
        ctx.strokeStyle = 'rgba(0, 255, 136, 0.6)';
        ctx.lineWidth = 1;
        ctx.shadowBlur = 8;
        ctx.strokeRect(scaledX + 2, scaledY + 2, scaledWidth - 4, scaledHeight - 4);
        
        // Reset shadow
        ctx.shadowBlur = 0;
        
        // Draw label background with glass effect
        const labelText = `${face.label} (${(face.confidence * 100).toFixed(0)}%)`;
        const labelWidth = Math.max(scaledWidth, labelText.length * 12);
        
        // Glass background
        ctx.fillStyle = 'rgba(0, 255, 136, 0.15)';
        ctx.fillRect(scaledX, scaledY - 40, labelWidth, 35);
        
        // Border
        ctx.strokeStyle = 'rgba(0, 255, 136, 0.4)';
        ctx.lineWidth = 1;
        ctx.strokeRect(scaledX, scaledY - 40, labelWidth, 35);
        
        // Draw label text with glow
        ctx.fillStyle = '#00FF88';
        ctx.font = 'bold 16px Inter, -apple-system, sans-serif';
        ctx.shadowColor = '#00FF88';
        ctx.shadowBlur = 10;
        ctx.fillText(labelText, scaledX + 8, scaledY - 15);
        ctx.shadowBlur = 0;
      }
    });
    
    if (isActive) {
      animationRef.current = requestAnimationFrame(drawFaceBoxes);
    }
  }, [isActive, faces]);

  // Set stream when video element is available
  useEffect(() => {
    if (videoRef.current && streamRef.current) {
      console.log('Setting video stream in useEffect for SmartFace');
      videoRef.current.srcObject = streamRef.current;
    }
  }, [hasPermission]);

  // Start drawing face boxes when active
  useEffect(() => {
    if (isActive) {
      drawFaceBoxes();
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    }
  }, [isActive, drawFaceBoxes]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return (
    <SmartFaceContainer>
      <h3>🚀 SmartFace - Real-time Face Detection & Tracking</h3>
      
      {detectorStatus && (
        <StatsDisplay>
          <StatItem>Detector: {detectorStatus.type || 'Unknown'}</StatItem>
          <StatItem>Status: {detectorStatus.available ? '✅ Ready' : '❌ Unavailable'}</StatItem>
          <StatItem>Total Detections: {stats.detectionCount}</StatItem>
          <StatItem>Max Faces: {stats.uniquePersons}</StatItem>
        </StatsDisplay>
      )}
      
      {error && <ErrorMessage>{error}</ErrorMessage>}

      {!hasPermission ? (
        <div>
          <Button 
            onClick={startCamera} 
            disabled={isLoading || !detectorStatus?.available}
            style={{ background: '#667eea', color: 'white', marginBottom: '10px' }}
          >
            {isLoading ? (
              <LoadingText>
                <LoadingSpinner />
                Starting Camera...
              </LoadingText>
            ) : (
              '📷 Start Camera'
            )}
          </Button>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '10px' }}>
            {!detectorStatus?.available ? 'Face detector not available' : 'Ready to start camera'}
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
            <Canvas
              ref={canvasRef}
              style={{ 
                position: 'absolute', 
                top: 0, 
                left: 0, 
                width: '100%', 
                height: '100%',
                pointerEvents: 'none'
              }}
            />
          </VideoContainer>

          <Status $isActive={isActive}>
            {isActive ? '🔴 Live Detection Active' : '📹 Camera Ready'}
          </Status>

          <Controls>
            {!isActive ? (
              <StartButton onClick={startDetection}>
                🎯 Start Detection
              </StartButton>
            ) : (
              <StartButton onClick={stopDetection} $isActive>
                ⏹️ Stop Detection
              </StartButton>
            )}

            <Button
              onClick={stopCamera}
              style={{ background: '#6c757d', color: 'white' }}
            >
              📷 Stop Camera
            </Button>

            <Button
              onClick={handleResetTracker}
              style={{ background: 'rgba(255, 193, 7, 0.2)', borderColor: 'rgba(255, 193, 7, 0.6)', color: '#FFC107' }}
            >
              🔄 Reset Tracker
            </Button>
          </Controls>

          {isActive && (
            <FaceDisplay>
              {faces.length > 0 ? (
                <>
                  <h4>Detected Faces:</h4>
                  {faces.map((face, index) => (
                    <FaceItem key={index}>
                      <PersonName>{face.label}</PersonName>
                      <Confidence>{(face.confidence * 100).toFixed(1)}%</Confidence>
                    </FaceItem>
                  ))}
                </>
              ) : (
                <div style={{ textAlign: 'center', color: '#666', padding: '20px' }}>
                  <h4>🔍 No Faces Detected</h4>
                  <p>Move your face into the camera view</p>
                </div>
              )}
            </FaceDisplay>
          )}
        </>
      )}
    </SmartFaceContainer>
  );
}

export default SmartFaceDetector;
