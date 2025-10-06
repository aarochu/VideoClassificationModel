import React, { useState, useCallback } from 'react';
import styled from 'styled-components';
import SmartFaceDetector from './components/SmartFaceDetector';
import VideoUploader from './components/VideoUploader';
import VideoPreview from './components/VideoPreview';
import PredictionResults from './components/PredictionResults';
import Header from './components/Header';
import { predictVideo } from './services/api';

const AppContainer = styled.div`
  min-height: 100vh;
  background: #0A0F14;
  color: #EAEAEA;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  overflow-x: hidden;
`;

const MainContent = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-height: 100vh;
`;

const ContentArea = styled.div`
  width: 100%;
  max-width: 1200px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const Section = styled.div`
  width: 100%;
  margin-bottom: 40px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const SectionTitle = styled.h2`
  color: #EAEAEA;
  margin-bottom: 30px;
  font-size: 28px;
  font-weight: 300;
  letter-spacing: 1px;
  text-align: center;
`;

const TabContainer = styled.div`
  display: flex;
  margin-bottom: 40px;
  background: rgba(0, 255, 136, 0.1);
  border: 1px solid rgba(0, 255, 136, 0.3);
  border-radius: 16px;
  padding: 8px;
  backdrop-filter: blur(10px);
`;

const Tab = styled.button`
  flex: 1;
  padding: 16px 24px;
  border: none;
  background: ${props => props.$active ? 'rgba(0, 255, 136, 0.2)' : 'transparent'};
  color: ${props => props.$active ? '#00FF88' : '#EAEAEA'};
  border-radius: 12px;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;

  &:hover {
    background: rgba(0, 255, 136, 0.15);
    color: #00FF88;
    transform: translateY(-1px);
  }

  ${props => props.$active && `
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    &::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(45deg, transparent, rgba(0, 255, 136, 0.1), transparent);
      animation: shimmer 2s infinite;
    }
  `}

  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
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
              $active={activeTab === 'upload'} 
              onClick={() => setActiveTab('upload')}
            >
              📁 Upload Video
            </Tab>
          </TabContainer>

          {activeTab === 'smartface' && (
            <Section>
              <SectionTitle>🚀 SmartFace - Real-time Face Detection & Tracking</SectionTitle>
              <SmartFaceDetector />
            </Section>
          )}

          {activeTab === 'upload' && (
            <>
              <Section>
                <SectionTitle>📁 Upload Video for Face Detection</SectionTitle>
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
                  <SectionTitle>Detection Results</SectionTitle>
                  <PredictionResults 
                    prediction={prediction}
                    error={error}
                  />
                </Section>
              )}
            </>
          )}
        </ContentArea>
      </MainContent>
    </AppContainer>
  );
}

export default App;
