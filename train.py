"""
=============================================================
  FILE: train.py
  PURPOSE: Load data → Preprocess → Train ML Models → Save
=============================================================

HOW DOES FAKE NEWS DETECTION WORK?
------------------------------------
1. We start with thousands of news articles labeled FAKE or REAL
2. We CLEAN the text (lowercase, remove noise)
3. We convert text to NUMBERS using TF-IDF (computers can't read words!)
4. We train a model to learn patterns from FAKE vs REAL articles
5. When a user types new text, the model predicts if it's FAKE or REAL

MACHINE LEARNING WORKFLOW:
    Data → Preprocess → Vectorize → Train → Evaluate → Save → Deploy
"""

# ---------------------------------------------------------------
# IMPORTS — loading all the libraries we need
# ---------------------------------------------------------------
import os           # os = interact with files and folders
import pickle       # pickle = save Python objects to disk (like our trained model)
import pandas as pd # pandas = work with spreadsheets/tables (DataFrames)
import numpy as np  # numpy = math operations on arrays
import matplotlib.pyplot as plt  # matplotlib = draw graphs/charts
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files

from sklearn.model_selection import train_test_split   # Split data into train/test sets
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to numbers
from sklearn.linear_model import LogisticRegression    # Model 1: Logistic Regression
from sklearn.naive_bayes import MultinomialNB          # Model 2: Naive Bayes
from sklearn.metrics import (
    accuracy_score,        # How often the model is correct (0 to 1)
    confusion_matrix,      # Table showing correct vs wrong predictions
    classification_report  # Detailed report with precision, recall, F1
)

# Import our custom preprocessing function from utils/preprocess.py
from utils.preprocess import preprocess_dataframe

# ---------------------------------------------------------------
# CONFIGURATION — easy-to-change settings at the top
# ---------------------------------------------------------------
DATA_DIR    = 'data'         # Folder where your downloaded CSV files live
MODELS_DIR  = 'models'       # Folder where we'll save trained models
STATIC_DIR  = 'static'       # Folder for saved graph images

# Create folders if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# ==============================================================
# STEP 1: LOAD DATA
# ==============================================================
def load_data():
    """
    Load the Fake and Real news CSV files from Kaggle dataset.

    Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

    The dataset has 2 files:
        - Fake.csv  → fake news articles  (label = 0)
        - True.csv  → real news articles  (label = 1)

    Each CSV has columns:
        - title   : headline of the article
        - text    : full body of the article
        - subject : topic category
        - date    : publication date

    Returns:
        df (DataFrame): Combined dataset with a 'label' column
    """
    print("\n📂 STEP 1: Loading Data...")

    fake_path = os.path.join(DATA_DIR, 'Fake.csv')
    true_path = os.path.join(DATA_DIR, 'True.csv')

    # Check if files exist — give helpful error if not
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print("\n❌ ERROR: Dataset files not found!")
        print("   Please download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("   Then place Fake.csv and True.csv inside the 'data/' folder")
        raise FileNotFoundError("Dataset not found. See instructions above.")

    # Read CSV files into DataFrames
    # pd.read_csv() = reads a CSV file into a table (DataFrame)
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    print(f"   📰 Fake news articles: {len(fake_df)}")
    print(f"   📰 Real news articles: {len(true_df)}")

    # Add LABELS — the answer key for our model
    # 0 = FAKE, 1 = REAL
    # Why numbers? ML models work with numbers, not text labels
    fake_df['label'] = 0   # 0 means FAKE
    true_df['label'] = 1   # 1 means REAL

    # Combine both DataFrames into one big table
    # pd.concat() = stack two DataFrames on top of each other
    # ignore_index=True = reset row numbers from 0
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine 'title' and 'text' into one column called 'text'
    # Why? The title often contains important clues about fake/real news
    # We add them together: "BREAKING NEWS: ... [full article text]"
    df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    # Shuffle the data so fake and real articles are mixed
    # frac=1 means "take 100% of rows" (i.e., all rows, shuffled)
    # random_state=42 makes shuffling reproducible (same result every time)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"   ✅ Total articles loaded: {len(df)}")
    return df


# ==============================================================
# STEP 2: VISUALIZE DATA
# ==============================================================
def visualize_data(df):
    """
    Create and save charts to understand our dataset better.

    Why visualize?
    - See if the dataset is BALANCED (equal fake/real) or unbalanced
    - Understand what words appear most in fake vs real news
    - Visualization helps catch problems before training
    """
    print("\n📊 STEP 2: Creating Visualizations...")

    # ---- Chart 1: Fake vs Real Count ----
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['FAKE News', 'REAL News']
    counts = [len(df[df['label'] == 0]), len(df[df['label'] == 1])]
    colors = ['#e74c3c', '#2ecc71']  # red for fake, green for real
    bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.8)

    # Add count numbers on top of each bar
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
                f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_title('Fake vs Real News Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Articles', fontsize=11)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'chart_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: static/chart_distribution.png")

    # ---- Chart 2: Article Length Distribution ----
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(8, 4))
    fake_lengths = df[df['label'] == 0]['text_length']
    real_lengths = df[df['label'] == 1]['text_length']
    ax.hist(fake_lengths, bins=50, alpha=0.6, color='#e74c3c', label='FAKE', edgecolor='none')
    ax.hist(real_lengths, bins=50, alpha=0.6, color='#2ecc71', label='REAL', edgecolor='none')
    ax.set_title('Article Length Distribution (Word Count)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Words', fontsize=11)
    ax.set_ylabel('Number of Articles', fontsize=11)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'chart_length.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: static/chart_length.png")

    print(f"   📈 Average FAKE article length: {fake_lengths.mean():.0f} words")
    print(f"   📈 Average REAL article length: {real_lengths.mean():.0f} words")


# ==============================================================
# STEP 3: FEATURE EXTRACTION (TF-IDF)
# ==============================================================
def extract_features(X_train, X_test):
    """
    Convert text into numbers using TF-IDF Vectorization.

    WHAT IS TF-IDF?
    ---------------
    TF-IDF = Term Frequency - Inverse Document Frequency

    ML models can't read words — they only understand numbers.
    TF-IDF converts each article into a row of numbers.

    TF (Term Frequency):
        How often does a word appear in THIS article?
        "president" appears 5 times → TF is high

    IDF (Inverse Document Frequency):
        Is this word RARE or COMMON across all articles?
        "the" appears in every article → IDF is low (not useful)
        "president" appears in only some articles → IDF is high (useful!)

    TF-IDF = TF × IDF
        Common words get LOW scores (useless)
        Rare but important words get HIGH scores (useful!)

    Parameters:
        X_train: Training text (list of cleaned article strings)
        X_test:  Testing text

    Returns:
        X_train_tfidf: Training data as numbers (matrix)
        X_test_tfidf:  Testing data as numbers (matrix)
        vectorizer:    The fitted TF-IDF object (save this for predictions!)
    """
    print("\n🔢 STEP 3: TF-IDF Feature Extraction...")

    # Create TF-IDF Vectorizer
    # max_features=50000 → only keep the 50,000 most important words
    # ngram_range=(1,2) → consider single words AND 2-word phrases
    #   e.g., "fake" alone + "fake news" as a combined feature
    # min_df=2 → ignore words that appear in fewer than 2 articles
    # sublinear_tf=True → apply log scaling to prevent very common words from dominating
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    # fit_transform() does TWO things at once:
    #   1. fit()      → learn the vocabulary from training data
    #   2. transform() → convert training data to TF-IDF numbers
    # We ONLY fit on training data — never on test data!
    # (This prevents "data leakage" — the model shouldn't see test data during training)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # transform() only — apply the SAME vocabulary to test data
    # We don't re-fit because we want to use the training vocabulary
    X_test_tfidf  = vectorizer.transform(X_test)

    print(f"   ✅ Vocabulary size: {len(vectorizer.vocabulary_):,} unique words/phrases")
    print(f"   ✅ Training matrix shape: {X_train_tfidf.shape}")
    print(f"      → {X_train_tfidf.shape[0]} articles × {X_train_tfidf.shape[1]} features")

    return X_train_tfidf, X_test_tfidf, vectorizer


# ==============================================================
# STEP 4: TRAIN MODELS
# ==============================================================
def train_models(X_train, y_train):
    """
    Train two ML classification models on our data.

    MODEL 1: Logistic Regression
    ----------------------------
    Despite the name, this is a CLASSIFICATION algorithm (not just regression).
    It learns a mathematical equation:
        P(FAKE) = sigmoid(w1×feature1 + w2×feature2 + ... + bias)
    If P(FAKE) > 0.5 → predict FAKE, else → predict REAL

    Think of it like a weighing scale: it weighs each word's importance
    and tips toward FAKE or REAL based on which side is heavier.

    MODEL 2: Naive Bayes
    --------------------
    Based on probability theory (Bayes' Theorem).
    "Given that this article contains the word 'election fraud',
     what is the probability it's FAKE?"
    It multiplies these probabilities for each word.
    Called "Naive" because it assumes all words are INDEPENDENT
    (which isn't true, but works surprisingly well in practice!)

    Parameters:
        X_train: Training features (TF-IDF matrix)
        y_train: Training labels (0=FAKE, 1=REAL)

    Returns:
        models (dict): Dictionary of trained model objects
    """
    print("\n🤖 STEP 4: Training Models...")

    models = {}

    # ---- Model 1: Logistic Regression ----
    print("   Training Logistic Regression...")
    # C=1.0 → regularization strength (prevents overfitting)
    # max_iter=1000 → max iterations to find optimal weights
    # solver='lbfgs' → optimization algorithm
    lr_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)

    # fit() = TRAIN the model
    # This is where the magic happens — the model learns from data!
    # It adjusts internal weights to minimize prediction errors
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    print("   ✅ Logistic Regression trained!")

    # ---- Model 2: Naive Bayes ----
    print("   Training Naive Bayes...")
    # alpha=0.1 → smoothing parameter (handles words not seen in training)
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)
    models['Naive Bayes'] = nb_model
    print("   ✅ Naive Bayes trained!")

    return models


# ==============================================================
# STEP 5: EVALUATE MODELS
# ==============================================================
def evaluate_models(models, X_test, y_test):
    """
    Measure how well each model performs on UNSEEN test data.

    KEY METRICS EXPLAINED:
    ----------------------
    Accuracy:  Out of 100 predictions, how many were correct?
               accuracy = correct predictions / total predictions

    Precision: When the model says FAKE, how often is it actually FAKE?
               High precision = fewer false alarms

    Recall:    Of all ACTUAL fake articles, what % did the model catch?
               High recall = fewer fake articles slipping through

    F1-Score:  The BALANCE between Precision and Recall
               F1 = 2 × (Precision × Recall) / (Precision + Recall)
               Best single metric to use when classes are balanced

    Confusion Matrix:
               A 2×2 table showing:
               - True Positives  (predicted FAKE, actually FAKE ✅)
               - True Negatives  (predicted REAL, actually REAL ✅)
               - False Positives (predicted FAKE, actually REAL ❌)
               - False Negatives (predicted REAL, actually FAKE ❌)

    Returns:
        best_model_name (str): Name of the best performing model
    """
    print("\n📏 STEP 5: Evaluating Models...\n")
    print("=" * 60)

    results = {}

    for name, model in models.items():
        print(f"\n🔍 Model: {name}")
        print("-" * 40)

        # predict() = use the trained model to make predictions on test data
        # Returns an array of 0s and 1s (0=FAKE, 1=REAL)
        y_pred = model.predict(X_test)

        # Calculate accuracy (0.0 to 1.0 → multiply by 100 for percentage)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"   🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"\n   📊 Classification Report:")

        # classification_report shows precision, recall, F1 for each class
        # target_names gives human-readable class names instead of 0/1
        report = classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'])
        # Indent the report for better readability
        for line in report.split('\n'):
            print(f"      {line}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"   🗂️  Confusion Matrix:")
        print(f"      [TN={cm[0,0]:5d}  FP={cm[0,1]:5d}]  ← Actual FAKE")
        print(f"      [FN={cm[1,0]:5d}  TP={cm[1,1]:5d}]  ← Actual REAL")
        print(f"       ↑              ↑")
        print(f"    Pred FAKE    Pred REAL")

    print("\n" + "=" * 60)

    # Find the best model (highest accuracy)
    best_model_name = max(results, key=results.get)
    print(f"\n🏆 Best Model: {best_model_name} with {results[best_model_name]*100:.2f}% accuracy")

    # Save comparison chart
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(results.keys())
    accs  = [v * 100 for v in results.values()]
    colors = ['#3498db', '#e67e22']
    bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=0.8, width=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(85, 101)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'chart_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: static/chart_accuracy.png")

    return best_model_name, models


# ==============================================================
# STEP 6: SAVE MODEL
# ==============================================================
def save_model(model, vectorizer, model_name="best_model"):
    """
    Save the trained model and vectorizer to disk using pickle.

    WHY SAVE THE MODEL?
    -------------------
    Training takes minutes. Predicting takes milliseconds.
    We save the model so we DON'T retrain every time someone
    visits our web app — we just load and predict instantly!

    pickle = Python's built-in way to serialize (save) any Python object
    joblib = Better alternative for large numpy arrays (sklearn uses it internally)

    We save TWO files:
    1. The MODEL  → knows how to predict FAKE/REAL
    2. The VECTORIZER → knows how to convert text to numbers
    Both are needed to make a prediction on new text!
    """
    print(f"\n💾 STEP 6: Saving Model...")

    model_path      = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')

    # 'wb' = write binary mode (pickle files are binary, not text)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"   ✅ Model saved to: {model_path}")
    print(f"   ✅ Vectorizer saved to: {vectorizer_path}")


# ==============================================================
# MAIN — runs everything in order
# ==============================================================
def main():
    print("=" * 60)
    print("  🚀 FAKE NEWS DETECTOR — TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    df = load_data()

    # Step 2: Visualize data
    visualize_data(df)

    # Step 3: Preprocess text
    print("\n🧹 STEP 3: Preprocessing Text...")
    df = preprocess_dataframe(df, text_column='text')

    # Step 4: Prepare features (X) and labels (y)
    X = df['cleaned_text']  # Features = the article text
    y = df['label']          # Labels   = 0 (FAKE) or 1 (REAL)

    # Step 5: Train-test split
    # We split data into:
    #   - TRAINING SET (80%) → the model LEARNS from this
    #   - TESTING SET  (20%) → we TEST the model on this (it hasn't seen it!)
    # This is like studying from textbooks (train) and then taking an exam (test)
    print("\n✂️  STEP 4: Splitting into Train/Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 20% goes to testing
        random_state=42,     # Fixed seed = reproducible split every time
        stratify=y           # Keep same fake/real ratio in both splits
    )
    print(f"   Training samples: {len(X_train):,}  |  Testing samples: {len(X_test):,}")

    # Step 6: TF-IDF feature extraction
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)

    # Step 7: Train models
    models = train_models(X_train_tfidf, y_train)

    # Step 8: Evaluate models
    best_name, models = evaluate_models(models, X_test_tfidf, y_test)

    # Step 9: Save the best model
    save_model(models[best_name], vectorizer, model_name='best_model')

    print("\n" + "=" * 60)
    print("  🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print("  Next step: Run 'streamlit run app.py' to launch the web app!")
    print("=" * 60)


# This block only runs when you execute this file directly:
#   python train.py
# It does NOT run when this file is imported by another file.
if __name__ == '__main__':
    main()
