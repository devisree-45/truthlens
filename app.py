import streamlit as st
from pathlib import Path
from src.fake_news_classifier import FakeNewsClassifier
from src.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="TruthLens - AI News Verifier",
    page_icon="🔎",
    layout="wide"
)

# Load external CSS if available
styles_path = Path("assets/styles.css")
if styles_path.exists():
    try:
        css = styles_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:
        st.warning("Failed to load stylesheet. UI may appear unstyled.")
else:
    st.warning("Stylesheet not found. UI may appear unstyled.")


# Inline CSS (loader + result)
st.markdown("""
<style>
.loader-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 250px;
    text-align: center;
    animation: fadeIn 0.4s ease-in;
}
.loader {
    border: 6px solid rgba(255, 255, 255, 0.2);
    border-top: 6px solid #00b4d8;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    animation: spin 1.2s linear infinite;
    margin-bottom: 15px;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.loading-text {
    font-size: 18px;
    font-weight: 500;
    color: #fff;
    letter-spacing: 0.5px;
}
.result-box {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 25px;
    margin-top: 15px;
    box-shadow: 0 4px 25px rgba(0, 0, 0, 0.15);
    animation: fadeIn 0.5s ease-in-out;
    color: #fff;
}
.result-title {
    font-size: 22px;
    font-weight: 600;
    text-align: center;
    padding-bottom: 10px;
}
.result-real {
    color: #22c55e;
    font-weight: bold;
    font-size: 24px;
}
.result-fake {
    color: #ef4444;
    font-weight: bold;
    font-size: 24px;
}
.confidence {
    font-size: 16px;
    margin-top: 6px;
    color: #ccc;
}
.explanation-block {
    margin-top: 18px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 15px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'latest_result' not in st.session_state:
        st.session_state.latest_result = None
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False


def render_hero():
    st.title("TruthLens")
    st.caption("Simple AI verification for news text")


def render_result_card(result: dict):
    """Render the simplified result block UI (no metrics)."""
    if not result or not result.get('success'):
        return

    classification = result['classification']
    confidence = result['confidence']
    label_class = "result-real" if classification == "REAL" else "result-fake"
    label_text = "✅ Real News" if classification == "REAL" else "⚠️ Fake News"

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-title">Analysis Result</div>
            <div class="{label_class}">{label_text}</div>
            <div class="confidence">Confidence: <strong>{confidence}%</strong></div>
        """,
        unsafe_allow_html=True
    )

    # Explanation block
    if result.get("reasoning"):
        st.markdown(
            f"""
            <div class="explanation-block">
                <strong>Explanation:</strong><br>{result['reasoning']}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    initialize_session_state()
    render_hero()

    if st.session_state.classifier is None:
        with st.spinner("Initializing classifier..."):
            st.session_state.classifier = FakeNewsClassifier()
            logger.info("Classifier initialized successfully")

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("Enter News Content")
        news_text = st.text_area("", height=220, placeholder="Paste your news article or headline...", label_visibility="collapsed")
        analyze_button = st.button("Analyze", type="primary", use_container_width=True)

    with right_col:
        st.subheader("Analysis Results")

        if analyze_button:
            if not news_text.strip():
                st.warning("Please enter text to analyze.")
            else:
                st.session_state.is_loading = True
                placeholder = st.empty()

                # Show animated loader
                placeholder.markdown(
                    """
                    <div class='card loader-container'>
                        <div class='loader'></div>
                        <div class='loading-text'>Analyzing your content...</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                try:
                    result = st.session_state.classifier.classify(news_text)
                    st.session_state.latest_result = result
                except Exception as e:
                    st.error(f"❌ **Analysis Error:** {str(e)}")
                    logger.error(f"Classification error: {e}")
                finally:
                    st.session_state.is_loading = False
                    placeholder.empty()

        if st.session_state.is_loading:
            st.markdown(
                """
                <div class='card loader-container'>
                    <div class='loader'></div>
                    <div class='loading-text'>Analyzing your content...</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif st.session_state.latest_result:
            render_result_card(st.session_state.latest_result)
        else:
            st.markdown(
                """
                <div class='card card--accent'>
                    <div class='section-header'>How It Works</div>
                    <ul style='margin: 10px 0; padding-left: 20px;'>
                        <li>Enter or paste news content in the left panel</li>
                        <li>Click 'Analyze' to verify authenticity</li>
                        <li>Get instant Fake/Real classification</li>
                        <li>View confidence score and explanation</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
