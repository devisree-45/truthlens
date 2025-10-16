import regex as re # <-- MODIFIED: Use 'regex' for advanced Unicode support
from typing import Optional
from src.logger import setup_logger

logger = setup_logger(__name__)

class TextPreprocessor:
    """Handle text preprocessing and cleaning for news articles."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize input text for multilingual support.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for cleaning")
            return ""
        
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # --- MODIFICATION 1: Multilingual Character Filtering ---
            # Remove special characters but keep basic punctuation.
            # We use re.UNICODE to ensure \w includes word characters from all languages.
            # The pattern is now less restrictive to preserve Hindi/Telugu characters.
            # We keep only only Unicode word chars (\w), whitespace (\s), and basic English punctuation.
            text = re.sub(r'[^\w\s.,!?;:\'-]', '', text, flags=re.UNICODE)
            
            # Remove multiple punctuation marks
            text = re.sub(r'([.,!?;:])\1+', r'\1', text)
            
            # (English-specific capitalization rule was removed)
            
            logger.debug(f"Text cleaned successfully. Length: {len(text)}")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text
    
    @staticmethod
    def validate_input(text: str, min_length: int = 10, max_length: int = 10000) -> tuple[bool, Optional[str]]:
        """
        Validate input text length and content for multilingual support.
        
        Args:
            text: Input text to validate
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text or not text.strip():
            return False, "Text cannot be empty"
        
        cleaned_text = text.strip()
        
        if len(cleaned_text) < min_length:
            return False, f"Text too short. Minimum {min_length} characters required"
        
        if len(cleaned_text) > max_length:
            return False, f"Text too long. Maximum {max_length} characters allowed"
        
        # --- MODIFICATION 3: Multilingual Alphabetic Check (Now works with 'regex') ---
        # Check if text contains at least some alphabetic characters using Unicode letter property (\p{L})
        if not re.search(r'\p{L}', cleaned_text, flags=re.UNICODE):
            return False, "Text must contain alphabetic characters"
        
        logger.debug("Input validation passed")
        return True, None
    
    @staticmethod
    def extract_features(text: str) -> dict:
        """
        Extract basic features from text. Clickbait feature is disabled for multilingual support.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            # The uppercase ratio feature is only meaningful for Latin scripts, kept for English context
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            # --- MODIFICATION 4: Disable English-specific clickbait check ---
            'has_clickbait_words': False 
        }
        
        logger.debug(f"Extracted features: {features}")
        return features