import React from 'react';
import styled from 'styled-components';
import ReactPlayer from 'react-player';

const PreviewContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
`;

const VideoWrapper = styled.div`
  width: 100%;
  max-width: 600px;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
`;

const ButtonContainer = styled.div`
  display: flex;
  gap: 15px;
  flex-wrap: wrap;
  justify-content: center;
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

const PredictButton = styled(Button)`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;

  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
  }

  &:active:not(:disabled) {
    transform: translateY(0);
  }
`;

const ResetButton = styled(Button)`
  background: #f8f9fa;
  color: #6c757d;
  border: 2px solid #dee2e6;

  &:hover:not(:disabled) {
    background: #e9ecef;
    border-color: #adb5bd;
  }
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
