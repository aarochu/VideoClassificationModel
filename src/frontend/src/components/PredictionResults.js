import React from 'react';
import styled from 'styled-components';

const ResultsContainer = styled.div`
  background: #f8f9fa;
  border-radius: 12px;
  padding: 30px;
  border-left: 5px solid ${props => props.isError ? '#dc3545' : '#28a745'};
`;

const ErrorContainer = styled(ResultsContainer)`
  background: #f8d7da;
  border-left-color: #dc3545;
`;

const Title = styled.h3`
  margin: 0 0 20px 0;
  color: ${props => props.isError ? '#721c24' : '#155724'};
  font-size: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
`;

const PredictionCard = styled.div`
  background: white;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
`;

const MainPrediction = styled.div`
  text-align: center;
  margin-bottom: 20px;
`;

const PredictedClass = styled.div`
  font-size: 24px;
  font-weight: 700;
  color: #333;
  margin-bottom: 10px;
`;

const Confidence = styled.div`
  font-size: 18px;
  color: #667eea;
  font-weight: 600;
`;

const ConfidenceBar = styled.div`
  width: 100%;
  height: 8px;
  background: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
  margin-top: 10px;
`;

const ConfidenceFill = styled.div`
  height: 100%;
  background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
  width: ${props => props.confidence * 100}%;
  transition: width 0.5s ease;
`;

const AllPredictions = styled.div`
  margin-top: 20px;
`;

const PredictionsTitle = styled.h4`
  margin: 0 0 15px 0;
  color: #333;
  font-size: 16px;
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

const ClassName = styled.span`
  font-weight: 500;
  color: #333;
`;

const Probability = styled.span`
  color: #667eea;
  font-weight: 600;
`;

const Timestamp = styled.div`
  font-size: 12px;
  color: #6c757d;
  margin-top: 15px;
  text-align: center;
`;

const ErrorMessage = styled.div`
  color: #721c24;
  font-size: 16px;
  line-height: 1.5;
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
