
import requests
import json
from typing import Optional, Dict, Any
from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)

class OllamaHandler:
    """Handle interactions with Ollama LLM API."""
    
    def __init__(self):
        """Initialize Ollama handler with configuration."""
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.endpoint = Config.OLLAMA_API_ENDPOINT
        self.timeout = Config.REQUEST_TIMEOUT
        self.max_retries = Config.MAX_RETRIES
        
        logger.info(f"Initialized OllamaHandler with model: {self.model}")
    
    def generate_response(self, prompt: str, temperature: float = None) -> Optional[Dict[str, Any]]:
        """
        Generate response from Ollama model.
        
        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Dictionary containing response and metadata, or None on failure
        """
        if temperature is None:
            temperature = Config.TEMPERATURE
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "options": {
                "num_predict": Config.MAX_TOKENS
            }
        }
        
        logger.info(f"Sending request to Ollama API: {self.endpoint}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info("Successfully received response from Ollama")
                logger.debug(f"Response: {result.get('response', '')[:100]}...")
                
                return {
                    'response': result.get('response', ''),
                    'model': result.get('model', self.model),
                    'total_duration': result.get('total_duration', 0),
                    'load_duration': result.get('load_duration', 0),
                    'prompt_eval_count': result.get('prompt_eval_count', 0),
                    'eval_count': result.get('eval_count', 0)
                }
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    logger.error("Max retries reached. Request failed.")
                    return None
                    
            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error: Cannot connect to Ollama at {self.base_url}")
                logger.error("Please ensure Ollama is running: 'ollama serve'")
                return None
                
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error: {e}")
                logger.error(f"Response: {response.text if response else 'No response'}")
                return None
                
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON response from Ollama")
                return None
                
            except Exception as e:
                logger.error(f"Unexpected error: {type(e).__name__}: {e}")
                return None
        
        return None
    
    def check_health(self) -> bool:
        """
        Check if Ollama service is available.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Ollama service is healthy")
            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False