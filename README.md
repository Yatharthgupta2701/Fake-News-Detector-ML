# 🔍 Fake News Detector

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red?logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A beginner-friendly **Machine Learning + NLP** project that detects whether a news article is **FAKE or REAL** with ~98% accuracy.

---

## 📸 Screenshots

> Add screenshots after running the app locally.

| Home Screen | Fake Result | Real Result |
|:-----------:|:-----------:|:-----------:|
| <img width="1915" height="953" alt="image" src="https://github.com/user-attachments/assets/4c8b5d41-82d8-4ecb-a709-2aa94736c9cf" /> | <img width="1914" height="956" alt="image" src="https://github.com/user-attachments/assets/f3617c94-cc32-420b-8fda-fd26d0091185" /> | <img width="1918" height="955" alt="image" src="https://github.com/user-attachments/assets/c2591060-2acb-478c-bd25-7b521e750787" /> |

---

## ✨ Features

- 🧠 **ML-Powered** — Trained on 44,000+ real news articles
- 📊 **Confidence Score** — See HOW confident the model is
- 🧹 **Smart Preprocessing** — Removes stopwords, lemmatizes text
- 📈 **Data Visualizations** — Charts for dataset insights
- ⚡ **Instant Results** — Predicts in milliseconds
- 🌐 **Web Interface** — Clean Streamlit UI, no frontend skills needed

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| NLP | NLTK (stopwords, lemmatization) |
| Feature Extraction | TF-IDF Vectorizer |
| ML Models | Logistic Regression, Naive Bayes |
| Visualization | Matplotlib |
| Web App | Streamlit |
| Model Storage | Pickle |

---

## 📁 Project Structure

```
fake_news_detector/
│
├── app.py               # 🌐 Streamlit web app (run this!)
├── train.py             # 🤖 Model training pipeline
├── requirements.txt     # 📦 All dependencies
├── README.md            # 📖 You are here
│
├── data/                # 📂 Put your CSV files here
│   ├── Fake.csv
│   └── True.csv
│
├── models/              # 💾 Saved trained models
│   ├── best_model.pkl
│   └── vectorizer.pkl
│
├── utils/               # 🔧 Helper functions
│   ├── __init__.py
│   └── preprocess.py    # Text cleaning functions
│
└── static/              # 📊 Saved chart images
    ├── chart_distribution.png
    ├── chart_length.png
    └── chart_accuracy.png
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

### 2. Create Virtual Environment
```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
- Go to: [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Download `Fake.csv` and `True.csv`
- Place them in the `data/` folder

### 5. Train the Model
```bash
python train.py
```

### 6. Launch the Web App
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📊 Model Performance

| Metric | Logistic Regression | Naive Bayes |
|--------|:------------------:|:-----------:|
| Accuracy | ~98.7% | ~94.2% |
| Precision | ~99% | ~94% |
| Recall | ~98% | ~94% |
| F1-Score | ~98% | ~94% |

**Winner: Logistic Regression** 🏆

---

## 🧠 How It Works

```
User Input (raw text)
        ↓
  Text Preprocessing
  • lowercase
  • remove punctuation
  • remove stopwords
  • lemmatization
        ↓
  TF-IDF Vectorization
  (text → 50,000-dim number vector)
        ↓
  Logistic Regression Model
        ↓
  FAKE (0) or REAL (1) + Confidence %
```

---

## 🌐 Deployment

### Deploy on Streamlit Cloud (Free)
1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Click Deploy!

### Deploy on Render (Free)
1. Add `Procfile`:  `web: streamlit run app.py --server.port=$PORT`
2. Push to GitHub
3. Create Web Service on [render.com](https://render.com)

---

## 🔮 Future Improvements

- [ ] 🌐 Real-time news URL analysis (paste a link)
- [ ] 🌍 Hindi fake news detection
- [ ] 📱 Mobile-responsive design
- [ ] 🕒 User history tracking
- [ ] 📡 News API integration (check trending news)
- [ ] 🎯 Deep Learning model (BERT/LSTM) for higher accuracy
- [ ] 🗣️ Browser extension

---

## 📝 License

MIT License — feel free to use and modify!

---

## 🙋 Author

Built by **Yatharth Gupta** as a Machine Learning portfolio project.

⭐ **Star this repo** if you found it helpful!
