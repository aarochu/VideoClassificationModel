import React from 'react';
import styled from 'styled-components';

const HeaderContainer = styled.header`
  background: transparent;
  color: #EAEAEA;
  padding: 40px 20px;
  text-align: center;
  position: relative;
  margin-bottom: 20px;
`;

const Title = styled.h1`
  margin: 0;
  font-size: 48px;
  font-weight: 300;
  margin-bottom: 16px;
  letter-spacing: 2px;
  background: linear-gradient(45deg, #00FF88, #00C8FF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00FF88, transparent);
    animation: pulse 2s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
  }
`;

const Subtitle = styled.p`
  margin: 0;
  font-size: 20px;
  opacity: 0.8;
  font-weight: 300;
  letter-spacing: 1px;
  margin-bottom: 20px;
`;

const Description = styled.p`
  margin: 0;
  font-size: 16px;
  opacity: 0.7;
  max-width: 700px;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.6;
  font-weight: 300;
`;

function Header() {
  return (
    <HeaderContainer>
      <Title>🚀 SmartFace</Title>
      <Subtitle>Real-time Face Detection & Tracking</Subtitle>
      <Description>
        Detect faces in real-time with consistent labeling. Upload videos or use your webcam 
        to see green bounding boxes around detected faces with persistent "Person 1, 2, 3..." labels.
      </Description>
    </HeaderContainer>
  );
}

export default Header;
