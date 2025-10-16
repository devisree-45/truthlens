
import os
from pathlib import Path
from dotenv import load_dotenv
from src.logger import setup_logger

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize logger
logger = setup_logger(__name__, os.getenv('LOG_LEVEL', 'INFO'))

class Config:
    """Configuration class for the Fake News Detection system."""
    
    # Ollama API Configuration
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3:8b')
    OLLAMA_API_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"
    
    # API Timeout and Retry Configuration
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '120'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    
    # Model Parameters
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.1'))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '500'))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Application Settings
    APP_TITLE = "🔍 Fake News Detection System"
    APP_DESCRIPTION = """
    This AI-powered system analyzes news articles and headlines to detect potential misinformation.
    Powered by Ollama LLM for intelligent reasoning and classification.
    """
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            assert cls.OLLAMA_BASE_URL, "OLLAMA_BASE_URL must be set"
            assert cls.OLLAMA_MODEL, "OLLAMA_MODEL must be set"
            logger.info("Configuration validated successfully")
            logger.info(f"Using model: {cls.OLLAMA_MODEL}")
            logger.info(f"Ollama endpoint: {cls.OLLAMA_API_ENDPOINT}")
            return True
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

# Validate configuration on import
Config.validate()
