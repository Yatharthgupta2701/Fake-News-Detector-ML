"""
=============================================================
  FILE: app.py
  PURPOSE: Streamlit Web App for Fake News Detection
=============================================================

WHAT IS STREAMLIT?
------------------
Streamlit is a Python library that lets you build beautiful
web apps with PURE PYTHON — no HTML, CSS, or JavaScript needed!

Just run: streamlit run app.py
And your app opens in the browser at http://localhost:8501

HOW TO RUN:
    1. First train the model: python train.py
    2. Then launch the app:  streamlit run app.py
"""

import streamlit as st   # streamlit = our web framework
import pickle            # pickle = load our saved model from disk
import os                # os = check if files exist
import time              # time = for adding dramatic pause 😄

# Import our preprocessing function so we clean input the same way as training
from utils.preprocess import clean_text

# ---------------------------------------------------------------
# PAGE CONFIGURATION — must be the FIRST streamlit call
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",    # Browser tab title
    page_icon="🔍",                      # Browser tab icon
    layout="centered",                  # Center the content
    initial_sidebar_state="expanded"    # Show sidebar by default
)

# ---------------------------------------------------------------
# CUSTOM CSS — make the app look better
# ---------------------------------------------------------------
st.markdown("""
<style>
    /* Main header */
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 15px;
        margin-bottom: 25px;
        color: white;
    }
    /* Result boxes */
    .result-fake {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(255,65,108,0.4);
    }
    .result-real {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(56,239,125,0.4);
    }
    /* Confidence bar */
    .confidence-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #0f3460;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# LOAD MODEL — cached so it only loads ONCE (not on every click)
# ---------------------------------------------------------------
@st.cache_resource  # This decorator caches the model in memory
def load_model():
    """
    Load the saved model and vectorizer from disk.

    @st.cache_resource = Streamlit caching decorator
    - First call: loads model from disk (slow, ~1 sec)
    - All future calls: returns the cached version (instant!)
    - This prevents the model from being reloaded on every user interaction

    Returns:
        model: The trained ML model
        vectorizer: The fitted TF-IDF vectorizer
        loaded: True if successfully loaded, False if not found
    """
    model_path = os.path.join('models', 'best_model.pkl')
    vec_path   = os.path.join('models', 'vectorizer.pkl')

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None, False

    # 'rb' = read binary mode
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer, True


def predict_news(text, model, vectorizer):
    """
    Predict whether a news article is FAKE or REAL.

    Parameters:
        text (str): Raw news article text entered by user
        model: Trained ML model
        vectorizer: Fitted TF-IDF vectorizer

    Returns:
        prediction (int): 0=FAKE, 1=REAL
        confidence (float): Probability of the prediction (0.0 to 1.0)
        cleaned (str): The cleaned version of the text
    """
    # Step 1: Clean the text using same preprocessing as training
    cleaned = clean_text(text)

    # Step 2: Convert to TF-IDF numbers
    # [cleaned] → wrap in list because vectorizer expects a list of strings
    features = vectorizer.transform([cleaned])

    # Step 3: Make prediction
    prediction = model.predict(features)[0]  # [0] gets first (only) result

    # Step 4: Get confidence score (probability)
    # predict_proba() returns [[prob_class_0, prob_class_1]]
    # [0] → first sample, then [prediction] → probability of predicted class
    probabilities = model.predict_proba(features)[0]
    confidence    = probabilities[prediction]

    return prediction, confidence, cleaned


# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📚 How It Works")
    st.markdown("""
    1. **Enter** a news article in the text box
    2. **Click** the Analyze button
    3. **See** whether it's FAKE or REAL

    ---
    ### 🧠 ML Pipeline
    ```
    Your Text
       ↓
    Preprocessing
    (clean, normalize)
       ↓
    TF-IDF Vectorizer
    (text → numbers)
       ↓
    ML Model
    (predict FAKE/REAL)
       ↓
    Result + Confidence
    ```
    ---
    ### 📊 About the Model
    - **Dataset**: ~44,000 articles
    - **Algorithm**: Logistic Regression
    - **Accuracy**: ~98-99%
    - **Features**: 50,000 TF-IDF features
    """)

    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.caption(
        "This tool is for educational purposes. "
        "Always verify news from trusted sources like Reuters, AP, BBC."
    )


# ---------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 Fake News Detector</h1>
    <p style="opacity:0.85; font-size:16px;">
        Powered by Machine Learning & NLP
    </p>
</div>
""", unsafe_allow_html=True)

# Load model
model, vectorizer, loaded = load_model()

if not loaded:
    # Show error if model hasn't been trained yet
    st.error("⚠️ Model not found! Please train the model first.")
    st.code("python train.py", language="bash")
    st.info("Make sure you have `Fake.csv` and `True.csv` in the `data/` folder before training.")
    st.stop()  # Stop rendering the rest of the app

# Model loaded successfully
st.success("✅ Model loaded successfully!")

# ---------------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------------
st.markdown("### 📝 Enter News Article")

# Text area for input
user_input = st.text_area(
    label="Paste the news article text here:",
    placeholder=(
        "Example: Scientists at NASA have announced the discovery of "
        "a new exoplanet that may support life. The team used the James Webb "
        "Telescope to detect atmospheric signatures..."
    ),
    height=200,
    help="Paste or type any news article — the longer and more detailed, the better!"
)

# Example buttons to quickly test the model
st.markdown("**Quick Examples:**")
col1, col2 = st.columns(2)

with col1:
    if st.button("📰 Try Real News Example"):
        user_input = (
            "The Federal Reserve announced on Wednesday that it will keep "
            "interest rates unchanged following its two-day policy meeting. "
            "Fed Chair Jerome Powell said the decision was unanimous among "
            "committee members, citing stable inflation and strong employment data. "
            "Markets reacted positively to the announcement, with major indices "
            "rising modestly in afternoon trading."
        )

with col2:
    if st.button("📰 Try Fake News Example"):
        user_input = (
            "SHOCKING: Government secretly putting microchips in COVID vaccines "
            "to track citizens, leaked documents reveal. A whistleblower has come "
            "forward with proof that Bill Gates and George Soros are behind a "
            "massive global control scheme. Mainstream media is hiding this truth "
            "from you! Share before they delete this!"
        )

# Character count helper
if user_input:
    word_count = len(user_input.split())
    st.caption(f"📊 Word count: {word_count} words")
    if word_count < 20:
        st.warning("⚠️ Short text may reduce accuracy. Try pasting a longer article!")

# ---------------------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------------------
st.markdown("---")

predict_clicked = st.button(
    "🔍 Analyze Article",
    type="primary",
    use_container_width=True,  # Full-width button
    disabled=(not user_input or len(user_input.strip()) < 10)
)

if predict_clicked and user_input:
    with st.spinner("🤔 Analyzing the article..."):
        time.sleep(0.8)  # Small pause for UX effect
        prediction, confidence, cleaned_text = predict_news(user_input, model, vectorizer)

    # ---------------------------------------------------------------
    # RESULTS SECTION
    # ---------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎯 Analysis Result")

    # Display result card
    if prediction == 0:  # FAKE
        st.markdown("""
        <div class="result-fake">
            ❌ FAKE NEWS DETECTED
        </div>
        """, unsafe_allow_html=True)
        result_emoji = "❌"
        result_text  = "FAKE"
        result_color = "#ff416c"
    else:  # REAL
        st.markdown("""
        <div class="result-real">
            ✅ REAL NEWS DETECTED
        </div>
        """, unsafe_allow_html=True)
        result_emoji = "✅"
        result_text  = "REAL"
        result_color = "#11998e"

    # Confidence Score
    st.markdown("#### 📊 Confidence Score")
    st.progress(float(confidence))  # Progress bar (0.0 to 1.0)
    st.markdown(f"**{confidence*100:.1f}%** confident this is **{result_text}** news")

    # Explanation of confidence
    if confidence >= 0.95:
        st.success("🎯 Very high confidence — the model is quite sure!")
    elif confidence >= 0.80:
        st.info("👍 Good confidence — result is likely correct")
    elif confidence >= 0.65:
        st.warning("⚠️ Moderate confidence — verify from other sources")
    else:
        st.warning("🤔 Low confidence — the article has mixed signals")

    # Expandable details section
    with st.expander("🔬 See Technical Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input Statistics:**")
            st.write(f"- Original length: {len(user_input.split())} words")
            st.write(f"- After cleaning: {len(cleaned_text.split())} words")
            st.write(f"- Words removed: {len(user_input.split()) - len(cleaned_text.split())}")

        with col2:
            st.markdown("**Model Info:**")
            model_type = type(model).__name__
            st.write(f"- Model: {model_type}")
            st.write(f"- Prediction: {'FAKE (class 0)' if prediction == 0 else 'REAL (class 1)'}")
            st.write(f"- Confidence: {confidence:.4f}")

        st.markdown("**Cleaned Text (what the model actually saw):**")
        st.code(cleaned_text[:500] + ("..." if len(cleaned_text) > 500 else ""), language=None)

    # Important reminder
    st.info(
        "💡 **Remember:** This AI tool is not perfect. "
        "Always verify important news from trusted, established news sources. "
        "No AI can replace critical thinking!"
    )

# ---------------------------------------------------------------
# CHARTS SECTION
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Dataset Visualizations")

chart_tab1, chart_tab2, chart_tab3 = st.tabs(
    ["📊 Distribution", "📏 Article Length", "🏆 Model Accuracy"]
)

with chart_tab1:
    chart_path = os.path.join('static', 'chart_distribution.png')
    if os.path.exists(chart_path):
        st.image(chart_path, use_column_width=True)
        st.caption("Distribution of fake vs real articles in the training dataset")
    else:
        st.info("Run `python train.py` to generate charts")

with chart_tab2:
    chart_path = os.path.join('static', 'chart_length.png')
    if os.path.exists(chart_path):
        st.image(chart_path, use_column_width=True)
        st.caption("Distribution of article lengths (word count) for fake vs real news")
    else:
        st.info("Run `python train.py` to generate charts")

with chart_tab3:
    chart_path = os.path.join('static', 'chart_accuracy.png')
    if os.path.exists(chart_path):
        st.image(chart_path, use_column_width=True)
        st.caption("Comparison of accuracy between Logistic Regression and Naive Bayes")
    else:
        st.info("Run `python train.py` to generate charts")

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 13px; padding: 10px;'>
    🔍 Fake News Detector | Built with Python, Scikit-learn & Streamlit<br>
    For educational purposes only • Always verify from trusted sources
</div>
""", unsafe_allow_html=True)
