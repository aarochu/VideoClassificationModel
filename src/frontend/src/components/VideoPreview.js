import React from 'react';
import styled from 'styled-components';
import ReactPlayer from 'react-player';

const PreviewContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 30px;
  width: 100%;
`;

const VideoWrapper = styled.div`
  width: 100%;
  max-width: 800px;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 
    0 0 40px rgba(0, 255, 136, 0.2),
    0 0 80px rgba(0, 200, 255, 0.1),
    inset 0 0 20px rgba(0, 255, 136, 0.1);
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

const ButtonContainer = styled.div`
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  justify-content: center;
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

const PredictButton = styled(Button)`
  background: rgba(0, 200, 255, 0.2);
  border-color: rgba(0, 200, 255, 0.5);
  color: #00C8FF;

  &:hover:not(:disabled) {
    background: rgba(0, 200, 255, 0.3);
    border-color: rgba(0, 200, 255, 0.7);
    box-shadow: 0 8px 25px rgba(0, 200, 255, 0.4);
  }
`;

const ResetButton = styled(Button)`
  background: rgba(234, 234, 234, 0.1);
  border-color: rgba(234, 234, 234, 0.3);
  color: #EAEAEA;

  &:hover:not(:disabled) {
    background: rgba(234, 234, 234, 0.2);
    border-color: rgba(234, 234, 234, 0.5);
    box-shadow: 0 8px 25px rgba(234, 234, 234, 0.2);
  }
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 2px solid rgba(0, 200, 255, 0.3);
  border-radius: 50%;
  border-top-color: #00C8FF;
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

function VideoPreview({ videoUrl, onPredict, onReset, loading }) {
  return (
    <PreviewContainer>
      <VideoWrapper>
        <ReactPlayer
          url={videoUrl}
          controls
          width="100%"
          height="auto"
          config={{
            file: {
              attributes: {
                style: { width: '100%', height: 'auto' }
              }
            }
          }}
        />
      </VideoWrapper>

      <ButtonContainer>
        <PredictButton onClick={onPredict} disabled={loading}>
          {loading ? (
            <LoadingText>
              <LoadingSpinner />
              Analyzing...
            </LoadingText>
          ) : (
            '🔍 Predict'
          )}
        </PredictButton>

        <ResetButton onClick={onReset} disabled={loading}>
          🔄 Reset
        </ResetButton>
      </ButtonContainer>
    </PreviewContainer>
  );
}

export default VideoPreview;
