"""
Unit tests for Fake News Detection System
Run with: python -m pytest tests/test_classifier.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fake_news_classifier import FakeNewsClassifier
from src.data_preprocessing import TextPreprocessor

# ============================================================================
# 1️⃣ Text Preprocessing Tests
# ============================================================================

def test_text_cleaning():
    preprocessor = TextPreprocessor()
    
    dirty_text = "This  is    a TEST!!!! With @@special $$characters??"
    clean_text = preprocessor.clean_text(dirty_text)
    
    assert "  " not in clean_text
    assert "@" not in clean_text and "$" not in clean_text
    assert clean_text.count("!") == 1
    assert clean_text.count("?") == 1
    
    print("✅ Text cleaning test passed")

def test_input_validation():
    preprocessor = TextPreprocessor()
    
    # Empty input
    is_valid, error = preprocessor.validate_input("")
    assert not is_valid
    assert error == "Text cannot be empty"
    
    # Too short
    is_valid, error = preprocessor.validate_input("Hi")
    assert not is_valid
    
    # Valid input
    is_valid, error = preprocessor.validate_input(
        "This is a proper news article text with enough length."
    )
    assert is_valid
    assert error is None
    
    print("✅ Input validation test passed")

def test_feature_extraction():
    preprocessor = TextPreprocessor()
    
    text = "SHOCKING NEWS!!! You won't believe this amazing discovery!"
    features = preprocessor.extract_features(text)
    
    assert features['word_count'] > 0
    assert features['exclamation_count'] >= 1
    assert features['has_clickbait_words'] == True
    
    print("✅ Feature extraction test passed")

# ============================================================================
# 2️⃣ Fake News Classification Tests (with Mock)
# ============================================================================

# Mock LLM response for real news
mock_real_response = """
CLASSIFICATION: REAL
CONFIDENCE: 95
REASONING: The news article contains verified facts and credible sources.
"""

# Mock LLM response for fake news
mock_fake_response = """
CLASSIFICATION: FAKE
CONFIDENCE: 90
REASONING: The news contains sensational language, clickbait phrases, and unverified claims.
"""

@patch('src.model_handler.OllamaHandler.generate_response')
def test_classification_real_news(mock_generate):
    mock_generate.return_value = {'response': mock_real_response, 'model': 'gemma:latest'}
    classifier = FakeNewsClassifier()
    
    real_news = """
    NASA successfully landed the Perseverance rover on Mars in 2021, conducting experiments and sending images back to Earth.
    """
    
    result = classifier.classify(real_news)
    
    assert result['success'] == True
    assert result['classification'] == "REAL"
    assert 0 <= result['confidence'] <= 100
    assert result['reasoning'] is not None
    
    print(f"✅ Real news classified as: {result['classification']} ({result['confidence']}%)")

@patch('src.model_handler.OllamaHandler.generate_response')
def test_classification_fake_news(mock_generate):
    mock_generate.return_value = {'response': mock_fake_response, 'model': 'gemma:latest'}
    classifier = FakeNewsClassifier()
    
    fake_news = """
    SHOCKING: Aliens are secretly controlling the government! You won't believe what happens next!!!
    """
    
    result = classifier.classify(fake_news)
    
    assert result['success'] == True
    assert result['classification'] == "FAKE"
    assert 0 <= result['confidence'] <= 100
    
    print(f"✅ Fake news classified as: {result['classification']} ({result['confidence']}%)")

def test_classification_empty_input():
    classifier = FakeNewsClassifier()
    
    result = classifier.classify("")
    
    assert result['success'] == False
    assert "error" in result
    
    print("✅ Empty input test passed")

# ============================================================================
# 3️⃣ Run All Tests
# ============================================================================

if __name__ == "__main__":
    print("Running Fake News Detector Tests...\n")
    
    test_text_cleaning()
    test_input_validation()
    test_feature_extraction()
    
    print("\n⚠️ Classification tests will use mocked LLM responses (no Ollama required)\n")
    
    test_classification_real_news()
    test_classification_fake_news()
    test_classification_empty_input()
    
    print("\n✅ All tests completed successfully!")
