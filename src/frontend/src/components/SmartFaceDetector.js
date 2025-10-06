import React, { useState, useRef, useCallback, useEffect } from 'react';
import styled from 'styled-components';
import { detectFacesInImage, getFaceDetectorStatus } from '../services/api';

const SmartFaceContainer = styled.div`
  background: #f8f9fa;
  border-radius: 12px;
  padding: 30px;
  text-align: center;
`;

const VideoContainer = styled.div`
  position: relative;
  max-width: 800px;
  margin: 0 auto 20px;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
  background: #000;
`;

const Video = styled.video`
  width: 100%;
  height: auto;
  display: block;
`;

const Canvas = styled.canvas`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
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

const StartButton = styled(Button)`
  background: ${props => props.$isActive ? '#dc3545' : '#28a745'};
  color: white;

  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
  }
`;

const Status = styled.div`
  font-size: 16px;
  font-weight: 500;
  color: ${props => props.$isActive ? '#dc3545' : '#28a745'};
  margin-bottom: 15px;
`;

const FaceDisplay = styled.div`
  background: white;
  border-radius: 8px;
  padding: 20px;
  margin-top: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  text-align: left;
`;

const FaceItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid #e9ecef;

  &:last-child {
    border-bottom: none;
  }
`;

const PersonName = styled.span`
  font-weight: 600;
  color: #333;
  font-size: 18px;
`;

const Confidence = styled.span`
  color: #667eea;
  font-weight: 600;
`;

const ErrorMessage = styled.div`
  background: #f8d7da;
  color: #721c24;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 20px;
  border: 1px solid #f5c6cb;
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: white;
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
  background: #e3f2fd;
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 20px;
  text-align: center;
`;

const StatItem = styled.div`
  display: inline-block;
  margin: 0 15px;
  font-size: 14px;
  color: #1976d2;
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
    const containerRect = canvas.parentElement.getBoundingClientRect();
    
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
        
        // Draw face box (green)
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 3;
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
        
        // Draw label background
        const labelText = `${face.label} (${(face.confidence * 100).toFixed(0)}%)`;
        ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
        ctx.fillRect(scaledX, scaledY - 35, Math.max(scaledWidth, labelText.length * 12), 30);
        
        // Draw label text
        ctx.fillStyle = '#000';
        ctx.font = 'bold 16px Arial';
        ctx.fillText(labelText, scaledX + 5, scaledY - 10);
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
