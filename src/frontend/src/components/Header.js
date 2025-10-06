import React from 'react';
import styled from 'styled-components';

const HeaderContainer = styled.header`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 30px 40px;
  text-align: center;
`;

const Title = styled.h1`
  margin: 0;
  font-size: 36px;
  font-weight: 700;
  margin-bottom: 10px;
`;

const Subtitle = styled.p`
  margin: 0;
  font-size: 18px;
  opacity: 0.9;
  font-weight: 300;
`;

const Description = styled.p`
  margin: 15px 0 0 0;
  font-size: 14px;
  opacity: 0.8;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.5;
`;

function Header() {
  return (
    <HeaderContainer>
      <Title>🎯 Human Classifier</Title>
      <Subtitle>AI-Powered Video Classification</Subtitle>
      <Description>
        Upload a short video clip and our AI model will classify who the person is 
        or what kind of activity they're performing.
      </Description>
    </HeaderContainer>
  );
}

export default Header;
