"""
=============================================================
  FILE: utils/preprocess.py
  PURPOSE: Clean and prepare raw news text for ML model
=============================================================

What is preprocessing?
-----------------------
Raw news text is messy — it has UPPERCASE letters, punctuation,
"stop words" like "the", "is", "a" that carry no useful meaning.
Before feeding text to a machine learning model, we must CLEAN it.

Think of it like washing vegetables before cooking — same idea!
"""

import re                        # re = regular expressions — used to find/remove patterns in text
import string                    # string = gives us a list of punctuation marks like .,!?;
import nltk                      # nltk = Natural Language Toolkit — the #1 NLP library in Python
from nltk.corpus import stopwords       # stopwords = common words that don't carry meaning
from nltk.stem import WordNetLemmatizer # lemmatizer = reduces words to their base form

# ---------------------------------------------------------------
# STEP 1: Download required NLTK data (only runs once)
# ---------------------------------------------------------------
# nltk needs to download small helper files the first time you run this.
# "stopwords" = list of common English words like "the", "is", "and"
# "wordnet" = a large dictionary of English words and their meanings
# "punkt" = helps split text into sentences/words (tokenizer)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---------------------------------------------------------------
# STEP 2: Create a Lemmatizer object
# ---------------------------------------------------------------
# Lemmatization = converting a word to its ROOT form
# Examples:
#   "running" → "run"
#   "studies" → "study"
#   "better" → "good"
# This helps the model treat related words as THE SAME word!
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------
# STEP 3: Get the list of English stopwords
# ---------------------------------------------------------------
# Stopwords are words like: "the", "is", "a", "an", "and", "but"
# These words appear everywhere and don't help distinguish fake vs real news.
# Removing them reduces noise in the data.
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    """
    Main function: Takes a raw news article string and returns clean text.

    Parameters:
        text (str): The raw news article text

    Returns:
        str: Cleaned, processed text ready for ML model

    Example:
        Input:  "The President SAID he will NOT go to the White House!!"
        Output: "president say go white house"
    """

    # Guard: if text is not a string (e.g. NaN/missing), return empty string
    if not isinstance(text, str):
        return ""

    # ---- Step A: Convert to lowercase ----
    # Why? "Fake", "FAKE", "fake" should all be treated as the SAME word.
    # ML models are case-sensitive by default, so we standardize everything.
    text = text.lower()

    # ---- Step B: Remove URLs ----
    # URLs like "https://cnn.com/story" add noise — remove them.
    # re.sub(pattern, replacement, string) replaces all pattern matches with replacement
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # ---- Step C: Remove HTML tags ----
    # Some articles may have leftover HTML like <br>, <p>, etc.
    # The pattern <.*?> matches anything inside angle brackets
    text = re.sub(r'<.*?>', '', text)

    # ---- Step D: Remove punctuation ----
    # string.punctuation gives us: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # We translate each punctuation character to None (i.e., delete it)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # ---- Step E: Remove numbers ----
    # Numbers like "2024", "100" usually don't help the model distinguish fake/real.
    text = re.sub(r'\d+', '', text)

    # ---- Step F: Tokenization (split into words) ----
    # "i love machine learning" → ["i", "love", "machine", "learning"]
    # This is called "tokenization" — splitting text into individual tokens (words)
    words = text.split()

    # ---- Step G: Remove stopwords + Lemmatize ----
    # We do BOTH in one loop for efficiency:
    #   1. Skip the word if it's a stopword (e.g., "the", "is", "a")
    #   2. Lemmatize the remaining words (e.g., "running" → "run")
    words = [
        lemmatizer.lemmatize(word)   # Step 2: get root form
        for word in words            # for each word in our list
        if word not in STOP_WORDS    # Step 1: only keep non-stopwords
        and len(word) > 2            # Also remove very short words (e.g., "ok", "ab")
    ]

    # ---- Step H: Join words back into a string ----
    # ML vectorizers expect a string, not a list.
    # ["president", "say", "go", "white", "house"] → "president say go white house"
    return ' '.join(words)


def preprocess_dataframe(df, text_column='text'):
    """
    Apply clean_text() to every row in a Pandas DataFrame column.

    Parameters:
        df (DataFrame): Your dataset
        text_column (str): Which column has the news text

    Returns:
        DataFrame: Same dataframe with a new 'cleaned_text' column added
    """
    print(f"  🔄 Cleaning {len(df)} articles... (this may take 30-60 seconds)")

    # Apply our clean_text function to every row in the text column
    # df['column'].apply(function) = run the function on each value in that column
    df['cleaned_text'] = df[text_column].apply(clean_text)

    print(f"  ✅ Cleaning complete!")
    return df
