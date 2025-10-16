# 🔎 TruthLens: AI News Verifier

**Bringing Clarity to the Digital Noise: Instant Verification, Trusted Reasoning.**

TruthLens is a Streamlit web application that uses a local Large Language Model (LLM) via **Ollama** to analyze news articles and headlines, classifying them as **REAL** or **FAKE** with a confidence score and detailed reasoning.

## ✨ Features

* **AI-Powered Classification:** Uses a sophisticated LLM (defaulting to `llama3:8b`) to determine authenticity.
* **Structured Reasoning:** The LLM is prompted to act as an expert fact-checker, analyzing the text based on six critical factors, including sensationalism, factual claims, and source credibility.
* **Confidence Score:** Provides a numerical percentage (0-100%) indicating the AI's certainty in its verdict.
* **Multilingual Text Handling:** Uses the `regex` library and Unicode properties (`\p{L}`) for robust cleaning and validation of multilingual inputs.
* **Resilient API Communication:** Implements retries (`MAX_RETRIES=3`) and specific error handling for timeouts and connection failures to the Ollama service.
* **Clean User Interface:** Built with Streamlit and enhanced with custom CSS (`assets/styles.css`) for a polished, modern dark theme.

---

## 🚀 Getting Started

Follow these steps to set up and run the TruthLens application locally.

### Prerequisites

1.  **Python:** Python 3.9+ is required. (Testing confirmed on a Python 3.13 environment)
2.  **Ollama:** You must have the [Ollama software](https://ollama.com/) installed and running.
3.  **Model:** Pull the required model:
    ```bash
    ollama pull llama3:8b
    ```

### 1. Project Setup

Clone this repository and navigate to the project directory.

### 2. Install Python Dependencies

The project uses the following libraries: `streamlit`, `requests`, `python-dotenv`, and `regex`.

```bash
pip install -r requirements.txt
```
### 3. Configuration (.env)
Create a file named .env in the root directory and populate it with your configuration settings. The key variables are:
```bash 
# AI-FakeNews-Detector/.env file content
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b
LOG_LEVEL=INFO
# Optional: REQUEST_TIMEOUT=120
# Optional: MAX_RETRIES=3
```
Note: The OLLAMA_BASE_URL and OLLAMA_MODEL values are used by src/config.py and src/model_handler.py.
### 4. Run the Application
 ## Step 4.1: Start Ollama
 Ensure the Ollama service is running in a separate terminal:
 ```bash 
  ollama serve
 ```
 ## Step 4.2: Launch Streamlit App
Run the main application file:
```bash
streamlit run app.py
```
The application will launch in your web browser, ready for testing.

### 🛠 Project Structure
The core logic is modularized across the src/ directory.
```bash
AI-FakeNews-Detector/
├── .env                  # Environment variables for configuration
├── app.py                # Main Streamlit application UI
├── requirements.txt      # Python dependencies
├── assets/
│   ├── styles.css        # Custom CSS for the Streamlit UI
│   └── logo.png          # App logo/icon (if used)
└── src/
    ├── config.py         # Loads .env vars and defines Config class
    ├── logger.py         # Setup and configuration of the standard logger
    ├── data_preprocessing.py # Text cleaning, validation, and feature extraction
    ├── model_handler.py  # Handles API requests and connection checks with Ollama
    └── fake_news_classifier.py # Core classification logic, prompt creation, and response parsing
```
### 🧪 Testing
Unit tests are included to ensure the core functionalities are reliable.

## Running Tests
The tests are set up to be run using pytest.

Classification Tests: These use the unittest.mock.patch decorator to simulate (mock) the Ollama response, allowing tests to run without an active Ollama service.

```bash
python -m pytest tests/test_classifier.py -v
```
Input Validation Tests: These explicitly check the TextPreprocessor's handling of various inputs (empty, too short, non-alphabetic, valid length).

```bash
python tests/test_app.py
```
### 🔒 Privacy Note
The TruthLens system is designed to run the AI model locally via Ollama. This means the content you submit for analysis is processed entirely on your computer and is not sent to any external cloud service, ensuring your privacy.