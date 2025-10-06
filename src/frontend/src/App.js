import React, { useState, useCallback } from 'react';
import styled from 'styled-components';
import VideoUploader from './components/VideoUploader';
import VideoPreview from './components/VideoPreview';
import PredictionResults from './components/PredictionResults';
import CameraRecorder from './components/CameraRecorder';
import LivePrediction from './components/LivePrediction';
import SmartFaceDetector from './components/SmartFaceDetector';
import Header from './components/Header';
import { predictVideo, saveVideo } from './services/api';

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
`;

const MainContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  background: white;
  border-radius: 20px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  overflow: hidden;
`;

const ContentArea = styled.div`
  padding: 40px;
`;

const Section = styled.div`
  margin-bottom: 40px;
`;

const SectionTitle = styled.h2`
  color: #333;
  margin-bottom: 20px;
  font-size: 24px;
  font-weight: 600;
`;

const TabContainer = styled.div`
  display: flex;
  margin-bottom: 30px;
  background: #f8f9fa;
  border-radius: 12px;
  padding: 5px;
`;

const Tab = styled.button`
  flex: 1;
  padding: 15px 20px;
  border: none;
  background: ${props => props.$active ? 'white' : 'transparent'};
  color: ${props => props.$active ? '#667eea' : '#6c757d'};
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: ${props => props.$active ? '0 2px 10px rgba(0,0,0,0.1)' : 'none'};

  &:hover {
    color: #667eea;
  }
`;

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('smartface');

  const handleFileSelect = useCallback((file) => {
    setSelectedFile(file);
    setVideoUrl(URL.createObjectURL(file));
    setPrediction(null);
    setError(null);
  }, []);

  const handlePredict = useCallback(async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const result = await predictVideo(selectedFile);
      setPrediction(result);
    } catch (err) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  }, [selectedFile]);

  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setVideoUrl(null);
    setPrediction(null);
    setError(null);
  }, []);

  const handleVideoRecorded = useCallback(async (file, className) => {
    try {
      const result = await saveVideo(file, className);
      alert(`✅ Video saved successfully to ${className} class!\n\nFilename: ${result.filename}\nYou can now use it for training.`);
    } catch (error) {
      alert(`❌ Failed to save video: ${error.message}`);
    }
  }, []);

  return (
    <AppContainer>
      <MainContent>
        <Header />
        <ContentArea>
          <TabContainer>
            <Tab 
              $active={activeTab === 'smartface'} 
              onClick={() => setActiveTab('smartface')}
            >
              🚀 SmartFace Detection
            </Tab>
            <Tab 
              $active={activeTab === 'predict'} 
              onClick={() => setActiveTab('predict')}
            >
              🔍 Predict Video
            </Tab>
            <Tab 
              $active={activeTab === 'live'} 
              onClick={() => setActiveTab('live')}
            >
              🎯 Live Recognition
            </Tab>
            <Tab 
              $active={activeTab === 'record'} 
              onClick={() => setActiveTab('record')}
            >
              📹 Record for Dataset
            </Tab>
          </TabContainer>

          {activeTab === 'smartface' && (
            <Section>
              <SectionTitle>🚀 SmartFace - Real-time Face Detection & Tracking</SectionTitle>
              <SmartFaceDetector />
            </Section>
          )}

          {activeTab === 'predict' && (
            <>
              <Section>
                <SectionTitle>Upload Video for Prediction</SectionTitle>
                <VideoUploader 
                  onFileSelect={handleFileSelect}
                  selectedFile={selectedFile}
                />
              </Section>

              {videoUrl && (
                <Section>
                  <SectionTitle>Video Preview</SectionTitle>
                  <VideoPreview 
                    videoUrl={videoUrl}
                    onPredict={handlePredict}
                    onReset={handleReset}
                    loading={loading}
                  />
                </Section>
              )}

              {(prediction || error) && (
                <Section>
                  <SectionTitle>Prediction Results</SectionTitle>
                  <PredictionResults 
                    prediction={prediction}
                    error={error}
                  />
                </Section>
              )}
            </>
          )}

          {activeTab === 'live' && (
            <Section>
              <SectionTitle>Live Face Recognition</SectionTitle>
              <LivePrediction 
                onPrediction={(predictions) => {
                  // Only log if there are predictions to avoid spam
                  if (predictions && predictions.length > 0) {
                    console.log('Live predictions:', predictions);
                  }
                }}
              />
            </Section>
          )}

          {activeTab === 'record' && (
            <Section>
              <SectionTitle>Record Video for Training Dataset</SectionTitle>
              <CameraRecorder 
                onVideoRecorded={handleVideoRecorded}
              />
            </Section>
          )}
        </ContentArea>
      </MainContent>
    </AppContainer>
  );
}

export default App;
