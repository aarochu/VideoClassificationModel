import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout for video processing
});

export const predictVideo = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      // Server responded with error status
      throw new Error(error.response.data.detail || 'Server error occurred');
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('No response from server. Please check if the backend is running.');
    } else {
      // Something else happened
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
};

export const getClasses = async () => {
  try {
    const response = await api.get('/classes');
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Failed to fetch classes');
    } else if (error.request) {
      throw new Error('No response from server. Please check if the backend is running.');
    } else {
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
};

export const healthCheck = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    return { status: 'unhealthy', model_loaded: false };
  }
};

export const saveVideo = async (file, className) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('class_name', className);

  try {
    const response = await api.post('/save_video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Failed to save video');
    } else if (error.request) {
      throw new Error('No response from server. Please check if the backend is running.');
    } else {
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
};

export const getDatasetInfo = async () => {
  try {
    const response = await api.get('/dataset_info');
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Failed to fetch dataset info');
    } else if (error.request) {
      throw new Error('No response from server. Please check if the backend is running.');
    } else {
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
};

export const detectFacesInImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await api.post('/detect_faces', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Face detection failed');
    } else if (error.request) {
      throw new Error('No response from server. Please check if the backend is running.');
    } else {
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
};

export const detectFacesInVideo = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await api.post('/detect_faces_video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // 60 seconds for video processing
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Video face detection failed');
    } else if (error.request) {
      throw new Error('No response from server. Please check if the backend is running.');
    } else {
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
};

export const getFaceDetectorStatus = async () => {
  try {
    const response = await api.get('/face_detector_status');
    return response.data;
  } catch (error) {
    return { available: false, type: null, capabilities: {} };
  }
};

export const resetFaceTracker = async () => {
  try {
    const response = await api.post('/reset_face_tracker');
    return response.data;
  } catch (error) {
    console.error('Error resetting face tracker:', error);
    throw error;
  }
};

export const switchDetector = async (method) => {
  try {
    const response = await api.post('/switch_detector', { method });
    return response.data;
  } catch (error) {
    console.error('Error switching detector:', error);
    throw error;
  }
};

export const getAvailableDetectors = async () => {
  try {
    const response = await api.get('/available_detectors');
    return response.data;
  } catch (error) {
    console.error('Error getting available detectors:', error);
    return { available_detectors: [], current_detector: null };
  }
};

export const getDetectionDebug = async () => {
  try {
    const response = await api.get('/detection_debug');
    return response.data;
  } catch (error) {
    console.error('Error getting detection debug:', error);
    return null;
  }
};
