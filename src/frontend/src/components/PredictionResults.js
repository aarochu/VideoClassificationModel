import React from 'react';
import styled from 'styled-components';

const ResultsContainer = styled.div`
  background: rgba(0, 255, 136, 0.05);
  border: 1px solid rgba(0, 255, 136, 0.2);
  border-radius: 20px;
  padding: 40px;
  backdrop-filter: blur(10px);
  box-shadow: 0 0 30px rgba(0, 255, 136, 0.1);
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #00FF88, #00C8FF);
  }
`;

const ErrorContainer = styled.div`
  background: rgba(255, 107, 107, 0.1);
  border: 1px solid rgba(255, 107, 107, 0.3);
  border-radius: 20px;
  padding: 40px;
  backdrop-filter: blur(10px);
  box-shadow: 0 0 30px rgba(255, 107, 107, 0.1);
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #FF6B6B, #FF8E8E);
  }
`;

const Title = styled.h3`
  margin: 0 0 30px 0;
  color: ${props => props.isError ? '#FF6B6B' : '#00FF88'};
  font-size: 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  font-weight: 500;
  letter-spacing: 0.5px;
  text-shadow: 0 0 10px ${props => props.isError ? 'rgba(255, 107, 107, 0.5)' : 'rgba(0, 255, 136, 0.5)'};
`;

const PredictionCard = styled.div`
  background: rgba(0, 200, 255, 0.05);
  border: 1px solid rgba(0, 200, 255, 0.2);
  border-radius: 16px;
  padding: 30px;
  margin-bottom: 30px;
  backdrop-filter: blur(10px);
  box-shadow: 0 0 20px rgba(0, 200, 255, 0.1);
`;

const MainPrediction = styled.div`
  text-align: center;
  margin-bottom: 30px;
`;

const PredictedClass = styled.div`
  font-size: 28px;
  font-weight: 500;
  color: #EAEAEA;
  margin-bottom: 16px;
  letter-spacing: 1px;
`;

const Confidence = styled.div`
  font-size: 20px;
  color: #00C8FF;
  font-weight: 500;
  text-shadow: 0 0 10px rgba(0, 200, 255, 0.5);
`;

const ConfidenceBar = styled.div`
  width: 100%;
  height: 12px;
  background: rgba(0, 200, 255, 0.1);
  border-radius: 8px;
  overflow: hidden;
  margin-top: 16px;
  border: 1px solid rgba(0, 200, 255, 0.2);
`;

const ConfidenceFill = styled.div`
  height: 100%;
  background: linear-gradient(90deg, #00FF88 0%, #00C8FF 100%);
  width: ${props => props.confidence * 100}%;
  transition: width 0.8s ease;
  box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
`;

const AllPredictions = styled.div`
  margin-top: 30px;
`;

const PredictionsTitle = styled.h4`
  margin: 0 0 20px 0;
  color: #EAEAEA;
  font-size: 18px;
  font-weight: 500;
  letter-spacing: 0.5px;
`;

const PredictionItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 0;
  border-bottom: 1px solid rgba(0, 200, 255, 0.1);

  &:last-child {
    border-bottom: none;
  }
`;

const ClassName = styled.span`
  font-weight: 500;
  color: #EAEAEA;
  font-size: 16px;
  letter-spacing: 0.3px;
`;

const Probability = styled.span`
  color: #00C8FF;
  font-weight: 600;
  font-size: 16px;
  text-shadow: 0 0 10px rgba(0, 200, 255, 0.5);
`;

const Timestamp = styled.div`
  font-size: 14px;
  color: rgba(234, 234, 234, 0.6);
  margin-top: 20px;
  text-align: center;
  font-weight: 300;
  letter-spacing: 0.3px;
`;

const ErrorMessage = styled.div`
  color: #FF6B6B;
  font-size: 18px;
  line-height: 1.6;
  font-weight: 400;
  letter-spacing: 0.3px;
`;

function PredictionResults({ prediction, error }) {
  if (error) {
    return (
      <ErrorContainer isError>
        <Title isError>❌ Prediction Failed</Title>
        <ErrorMessage>{error}</ErrorMessage>
      </ErrorContainer>
    );
  }

  if (!prediction) {
    return null;
  }

  const { predicted_class, confidence, all_probabilities, timestamp } = prediction;

  // Sort probabilities for display
  const sortedProbabilities = Object.entries(all_probabilities)
    .sort(([, a], [, b]) => b - a);

  return (
    <ResultsContainer>
      <Title>✅ Prediction Results</Title>
      
      <PredictionCard>
        <MainPrediction>
          <PredictedClass>🎯 {predicted_class}</PredictedClass>
          <Confidence>
            {(confidence * 100).toFixed(1)}% confidence
          </Confidence>
          <ConfidenceBar>
            <ConfidenceFill confidence={confidence} />
          </ConfidenceBar>
        </MainPrediction>

        <AllPredictions>
          <PredictionsTitle>All Class Probabilities:</PredictionsTitle>
          {sortedProbabilities.map(([className, probability]) => (
            <PredictionItem key={className}>
              <ClassName>{className}</ClassName>
              <Probability>{(probability * 100).toFixed(1)}%</Probability>
            </PredictionItem>
          ))}
        </AllPredictions>
      </PredictionCard>

      {timestamp && (
        <Timestamp>
          Prediction made at: {new Date(timestamp).toLocaleString()}
        </Timestamp>
      )}
    </ResultsContainer>
  );
}

export default PredictionResults;
