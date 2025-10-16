# ============================================================================ 
# FILE: tests/test_input_cases.py
# ============================================================================

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fake_news_classifier import FakeNewsClassifier

def run_input_tests():
    classifier = FakeNewsClassifier()
    
    test_cases = [
        {
            "text": "Apple announced its new M3 chip with improved performance and energy efficiency.",
            "expected": ["REAL", "FAKE"]
        },
        {
            "text": "SHOCKING: Aliens have been living among us for decades! Government officials FINALLY admit the truth!",
            "expected": ["REAL", "FAKE"]
        },
        {
            "text": "",  # Empty input
            "expected_error": "Text cannot be empty"
        },
        {
            "text": "Hi",  # Too short
            "expected_error": "Text too short"
        },
        {
            "text": "1234567890!!!@@@###",  # Non-alphabetic
            "expected_error": "Text must contain alphabetic characters"
        },
        {
            "text": "Scientists at MIT have developed a new method for detecting gravitational waves using quantum sensors, which could revolutionize our understanding of the universe.",
            "expected": ["REAL", "FAKE"]
        },
        {
            "text": "BREAKING: One weird trick to make millions overnight revealed by unknown source!!!",
            "expected": ["REAL", "FAKE"]
        }
    ]
    
    for idx, case in enumerate(test_cases):
        print(f"\nTest Case {idx + 1}:")
        text = case["text"]
        result = classifier.classify(text)
        
        if 'expected_error' in case:
            assert not result['success'], f"Expected failure but got success for input: {text}"
            assert case['expected_error'].lower() in result['error'].lower(), f"Error mismatch: {result['error']}"
            print(f"✅ Passed (Error correctly raised: {result['error']})")
        else:
            assert result['success'], f"Classification failed unexpectedly for input: {text}"
            assert result['classification'] in case['expected'], f"Unexpected classification: {result['classification']}"
            assert 0 <= result['confidence'] <= 100, "Confidence out of range"
            print(f"✅ Passed (Classification: {result['classification']}, Confidence: {result['confidence']}%)")

if __name__ == "__main__":
    print("Running user input test cases...")
    run_input_tests()
    print("\n✅ All input test cases completed!")
