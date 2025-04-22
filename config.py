import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Base configuration."""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-for-testing-only')
    FLASK_APP = os.environ.get('FLASK_APP', 'app.py')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    
    # Application settings
    CAPTION_INTERVAL = int(os.environ.get('CAPTION_INTERVAL', '4'))
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Model settings
    CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
    
    # Speech recognition settings
    ENERGY_THRESHOLD = 400
    PAUSE_THRESHOLD = 0.7
    NON_SPEAKING_DURATION = 0.5
    
    # TTS settings
    TTS_RATE = 150
    TTS_VOLUME = 0.9
    
    # Webcam settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 360
    CAMERA_FPS = 15
    
class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    
# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Returns the appropriate configuration object based on environment."""
    config_name = os.environ.get('FLASK_ENV', 'default')
    return config.get(config_name, config['default']) 