import re
from typing import Dict, Any, Optional
from src.model_handler import OllamaHandler
from src.data_preprocessing import TextPreprocessor
from src.logger import setup_logger

logger = setup_logger(__name__)

class FakeNewsClassifier:
    """Classify news articles as Real or Fake using Ollama LLM."""

    def __init__(self):
        """Initialize classifier with model handler and preprocessor."""
        self.model_handler = OllamaHandler()
        self.preprocessor = TextPreprocessor()
        logger.info("FakeNewsClassifier initialized")

    def _create_classification_prompt(self, text: str) -> str:
        """
        Create a structured prompt for fake news classification.

        Args:
            text: News article or headline to classify

        Returns:
            Formatted prompt for the LLM
        """
        prompt = f"""You are an expert fact-checker and misinformation analyst. Analyze the following news article or headline and determine if it is REAL or FAKE news.

Consider the following factors:
1. Sensationalism and emotional manipulation
2. Factual claims that can be verified
3. Source credibility indicators
4. Logical consistency and coherence
5. Use of clickbait language
6. Presence of verifiable facts vs opinions

News Article/Headline:
"{text}"

Provide your analysis in the following format:

CLASSIFICATION: [REAL or FAKE]
CONFIDENCE: [A percentage from 0-100]
REASONING: [Detailed explanation of your decision, highlighting specific indicators]

Be thorough and analytical in your reasoning."""
        return prompt

    def _parse_model_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Robustly parse Ollama's response to extract classification, confidence, and reasoning.
        """
        try:
            # Normalize response for easier regex
            response_upper = response.upper()

            # Extract classification
            classification_match = re.search(r'CLASSIFICATION\s*[:\-]\s*(REAL|FAKE)', response_upper)
            classification = classification_match.group(1) if classification_match else "UNKNOWN"

            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE\s*[:\-]\s*(\d{1,3})', response_upper)
            confidence = int(confidence_match.group(1)) if confidence_match else 50
            confidence = max(0, min(100, confidence))

            # Extract reasoning
            reasoning_match = re.search(r'REASONING\s*[:\-]\s*(.*)', response, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else response.strip()

            logger.info(f"Parsed classification: {classification} ({confidence}%)")
            return {
                'classification': classification,
                'confidence': confidence,
                'reasoning': reasoning,
                'raw_response': response
            }

        except Exception as e:
            logger.error(f"Error parsing model response: {e}")
            return {
                'classification': "UNKNOWN",
                'confidence': 0,
                'reasoning': response.strip(),
                'raw_response': response
            }

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify a news article as Real or Fake.

        Args:
            text: News article or headline text

        Returns:
            Dictionary containing classification results and metadata
        """
        logger.info("Starting classification process")

        # Validate input
        is_valid, error_msg = self.preprocessor.validate_input(text)
        if not is_valid:
            logger.warning(f"Input validation failed: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'classification': None,
                'confidence': 0,
                'reasoning': None
            }

        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)

        # Extract features (for logging/analysis)
        features = self.preprocessor.extract_features(cleaned_text)
        logger.debug(f"Text features: {features}")

        # Create prompt
        prompt = self._create_classification_prompt(cleaned_text)

        # Get model response
        model_output = self.model_handler.generate_response(prompt)

        if not model_output:
            logger.error("Failed to get response from model")
            return {
                'success': False,
                'error': 'Failed to communicate with Ollama. Please ensure the service is running.',
                'classification': None,
                'confidence': 0,
                'reasoning': None
            }

        # Parse response
        parsed_result = self._parse_model_response(model_output['response'])

        if not parsed_result:
            logger.error("Failed to parse model response")
            return {
                'success': False,
                'error': 'Failed to parse model response',
                'classification': None,
                'confidence': 0,
                'reasoning': model_output['response']
            }

        logger.info("Classification completed successfully")
        return {
            'success': True,
            'classification': parsed_result['classification'],
            'confidence': parsed_result['confidence'],
            'reasoning': parsed_result['reasoning'],
            'features': features,
            'model_info': {
                'model': model_output['model'],
                'eval_count': model_output.get('eval_count', 0)
            }
        }
