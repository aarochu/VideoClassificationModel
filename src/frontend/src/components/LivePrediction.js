import React, { useState, useRef, useCallback, useEffect } from 'react';
import styled from 'styled-components';

const LiveContainer = styled.div`
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

const PredictionDisplay = styled.div`
  background: white;
  border-radius: 8px;
  padding: 20px;
  margin-top: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  text-align: left;
`;

const PredictionItem = styled.div`
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

function LivePrediction({ onPrediction }) {
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);
  const predictionIntervalRef = useRef(null);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      
      console.log('Starting live prediction camera...');
      
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
      
      console.log('Requesting camera access for live prediction...');
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      console.log('Camera stream obtained for live prediction:', stream);
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
    setIsActive(false);
    
    // Clear intervals
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = null;
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

  // Enhanced face detection function with debugging
  const detectFacesInImage = useCallback((imageData, width, height) => {
    const faces = [];
    
    // Simple skin tone detection for face detection
    // This is a basic approach - in production you'd use a proper ML model
    const data = imageData.data;
    const skinPixels = [];
    
    // Scan for skin-colored pixels with more relaxed detection
    for (let y = 0; y < height; y += 6) { // Sample every 6th pixel for better coverage
      for (let x = 0; x < width; x += 6) {
        const index = (y * width + x) * 4;
        const r = data[index];
        const g = data[index + 1];
        const b = data[index + 2];
        
        // More relaxed skin tone detection
        const isSkinTone = (
          // Basic RGB ranges for skin tones (more inclusive)
          r > 80 && g > 30 && b > 15 &&
          // Ensure there's some color variation (not pure gray)
          Math.max(r, g, b) - Math.min(r, g, b) > 10 &&
          // Red should be somewhat dominant (typical for skin)
          r > g && r > b &&
          // Avoid very bright or very dark pixels
          r < 240 && g < 240 && b < 240 &&
          r > 40 && g > 20 && b > 10
        );
        
        if (isSkinTone) {
          skinPixels.push({ x, y });
        }
      }
    }
    
    // Debug logging
    console.log(`Found ${skinPixels.length} skin pixels in ${width}x${height} image`);
    
    if (skinPixels.length > 20) { // Lowered threshold for easier detection
      // Find the center of skin pixels
      const centerX = skinPixels.reduce((sum, p) => sum + p.x, 0) / skinPixels.length;
      const centerY = skinPixels.reduce((sum, p) => sum + p.y, 0) / skinPixels.length;
      
      // Calculate bounding box around the face
      const minX = Math.min(...skinPixels.map(p => p.x));
      const maxX = Math.max(...skinPixels.map(p => p.x));
      const minY = Math.min(...skinPixels.map(p => p.y));
      const maxY = Math.max(...skinPixels.map(p => p.y));
      
      // Make the box more face-sized
      const rawWidth = maxX - minX;
      const rawHeight = maxY - minY;
      
      // Use the smaller dimension as base and make it square-ish for face
      const baseSize = Math.min(rawWidth, rawHeight);
      
      // Set reasonable min/max sizes for face detection
      const minFaceSize = Math.min(width, height) * 0.08; // At least 8% of screen
      const maxFaceSize = Math.min(width, height) * 0.5; // At most 50% of screen
      
      const faceWidth = Math.max(minFaceSize, Math.min(maxFaceSize, baseSize * 1.2));
      const faceHeight = Math.max(minFaceSize, Math.min(maxFaceSize, baseSize * 1.4));
      
      // Center the face box on the detected skin area
      const faceX = Math.max(0, centerX - faceWidth / 2);
      const faceY = Math.max(0, centerY - faceHeight / 2);
      
      // Ensure the box stays within video bounds
      const finalWidth = Math.min(faceWidth, width - faceX);
      const finalHeight = Math.min(faceHeight, height - faceY);
      
      faces.push({
        x: faceX,
        y: faceY,
        width: finalWidth,
        height: finalHeight
      });
      
      console.log(`Detected face at (${faceX}, ${faceY}) with size ${finalWidth}x${finalHeight}`);
    } else {
      console.log('No face detected - not enough skin pixels found');
    }
    
    return { faces, skinPixels };
  }, []);

  const detectFaces = useCallback(async () => {
    try {
      if (!videoRef.current) {
        console.log('No video ref available');
        return;
      }

      // Use MediaPipe Face Detection or a simpler approach
      // For now, let's use a basic face detection approach
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      if (!canvas) {
        console.log('No canvas ref available');
        return;
      }

      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw current frame
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Get image data for face detection
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      // Simple face detection using a basic algorithm
      // This is a placeholder - in a real app you'd use a proper face detection library
      const { faces, skinPixels } = detectFacesInImage(imageData, canvas.width, canvas.height);
      
      console.log(`detectFaces: Found ${faces.length} faces, ${skinPixels.length} skin pixels`);
      
      if (faces.length > 0) {
        // Calculate confidence based on skin pixel density
        const skinDensity = skinPixels.length / (canvas.width * canvas.height / 64); // Normalize by area
        const baseConfidence = Math.min(0.9, skinDensity * 0.1); // Scale confidence
        
        console.log(`detectFaces: Skin density: ${skinDensity}, base confidence: ${baseConfidence}`);
        
        // For now, use mock predictions for detected faces
        const predictions = faces.map((face, index) => ({
          name: `Face ${index + 1}`,
          confidence: Math.max(0.3, baseConfidence - (index * 0.1)), // Ensure minimum confidence
          boundingBox: face
        }));
        
        console.log(`detectFaces: Created ${predictions.length} predictions:`, predictions);
        
        // Only show predictions with reasonable confidence
        const validPredictions = predictions.filter(p => p.confidence > 0.4);
        
        console.log(`detectFaces: ${validPredictions.length} valid predictions after filtering`);
        
        if (validPredictions.length > 0) {
          console.log('detectFaces: Setting predictions:', validPredictions);
          setPredictions(validPredictions);
          if (onPrediction) {
            onPrediction(validPredictions);
          }
        } else {
          console.log('detectFaces: No valid predictions, clearing');
          // Clear predictions if confidence is too low
          setPredictions([]);
          if (onPrediction) {
            onPrediction([]);
          }
        }
      } else {
        console.log('detectFaces: No faces detected, clearing predictions');
        // Clear predictions when no faces detected
        setPredictions([]);
        if (onPrediction) {
          onPrediction([]);
        }
      }
    } catch (error) {
      console.error('Face detection error:', error);
    }
  }, [onPrediction, detectFacesInImage]);

  const startPrediction = useCallback(() => {
    if (!hasPermission) return;
    
    setIsActive(true);
    
    // Make face detection every 200ms for real-time tracking
    predictionIntervalRef.current = setInterval(() => {
      detectFaces();
    }, 200);
    
    // Make initial detection
    detectFaces();
  }, [hasPermission, detectFaces]);

  const stopPrediction = useCallback(() => {
    setIsActive(false);
    
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = null;
    }
    
    setPredictions([]);
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
    
    // Draw current video frame to canvas first
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Debug logging
    console.log(`Drawing ${predictions.length} predictions on canvas ${canvas.width}x${canvas.height}`);
    console.log(`Video source: ${video.videoWidth}x${video.videoHeight}, Display: ${videoRect.width}x${videoRect.height}`);
    console.log(`Scale factors: x=${scaleX}, y=${scaleY}`);
    
    // Draw face detection boxes for each detected face
    predictions.forEach((prediction, index) => {
      if (prediction.confidence > 0.4 && prediction.boundingBox) { // Lowered confidence threshold
        const { x, y, width, height } = prediction.boundingBox;
        
        // Scale coordinates to match display size
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;
        
        console.log(`Drawing box ${index}: original(${x}, ${y}) scaled(${scaledX}, ${scaledY}) size ${scaledWidth}x${scaledHeight} confidence ${prediction.confidence}`);
        
        // Draw face box
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 4; // Made thicker for better visibility
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
        
        // Draw name label background
        ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
        ctx.fillRect(scaledX, scaledY - 35, Math.max(scaledWidth, prediction.name.length * 12), 30);
        
        // Draw name label
        ctx.fillStyle = '#000';
        ctx.font = 'bold 18px Arial'; // Made font bigger and bold
        ctx.fillText(
          `${prediction.name} (${(prediction.confidence * 100).toFixed(0)}%)`,
          scaledX + 5,
          scaledY - 10
        );
        
        console.log(`Box drawn successfully for ${prediction.name}`);
      } else {
        console.log(`Skipping prediction ${index}: confidence=${prediction.confidence}, hasBox=${!!prediction.boundingBox}`);
      }
    });
    
    if (isActive) {
      animationRef.current = requestAnimationFrame(drawFaceBoxes);
    }
  }, [isActive, predictions]);

  // Set stream when video element is available
  useEffect(() => {
    if (videoRef.current && streamRef.current) {
      console.log('Setting video stream in useEffect for live prediction');
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
    <LiveContainer>
      <h3>🎯 Live Face Recognition</h3>
      
      {error && <ErrorMessage>{error}</ErrorMessage>}

      {!hasPermission ? (
        <div>
          <Button 
            onClick={startCamera} 
            disabled={isLoading}
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
            {isActive ? '🔴 Live Recognition Active' : '📹 Camera Ready'}
          </Status>
          
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '15px' }}>
            Debug: Stream={streamRef.current ? '✅' : '❌'} | Video={videoRef.current ? '✅' : '❌'} | Permission={hasPermission ? '✅' : '❌'}
          </div>

          <Controls>
            {!isActive ? (
              <StartButton onClick={startPrediction}>
                🎯 Start Recognition
              </StartButton>
            ) : (
              <StartButton onClick={stopPrediction} $isActive>
                ⏹️ Stop Recognition
              </StartButton>
            )}

            <Button
              onClick={stopCamera}
              style={{ background: '#6c757d', color: 'white' }}
            >
              📷 Stop Camera
            </Button>
            
            {/* Temporary test button */}
            <Button
              onClick={() => {
                console.log('Test button clicked - setting test prediction');
                setPredictions([{
                  name: 'Test Face',
                  confidence: 0.8,
                  boundingBox: { x: 100, y: 100, width: 200, height: 200 }
                }]);
              }}
              style={{ background: '#ffc107', color: 'black' }}
            >
              🧪 Test Box
            </Button>
          </Controls>

          {isActive && (
            <PredictionDisplay>
              {predictions.length > 0 ? (
                <>
                  <h4>Current Predictions:</h4>
                  {predictions.map((pred, index) => (
                    <PredictionItem key={index}>
                      <PersonName>{pred.name}</PersonName>
                      <Confidence>{(pred.confidence * 100).toFixed(1)}%</Confidence>
                    </PredictionItem>
                  ))}
                </>
              ) : (
                <div style={{ textAlign: 'center', color: '#666', padding: '20px' }}>
                  <h4>🔍 No Face Detected</h4>
                  <p>Move your face into the camera view</p>
                </div>
              )}
            </PredictionDisplay>
          )}
        </>
      )}
    </LiveContainer>
  );
}

export default LivePrediction;
