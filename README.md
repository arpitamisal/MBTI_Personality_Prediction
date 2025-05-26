# 🧠 Myers-Briggs Personality Prediction from Text

This project uses Natural Language Processing (NLP) and machine learning to predict a user's **MBTI personality type** based on their forum posts. The Myers-Briggs Type Indicator (MBTI) categorizes personalities into 16 types using 4 traits: Introversion/Extroversion, Sensing/Intuition, Thinking/Feeling, and Judging/Perceiving.

---

## 🚀 Project Overview

- **Goal:** Predict the MBTI type (e.g., INFP, ESTJ) from user-generated text
- **Dataset:** MBTI user forum posts from Kaggle + synthetic forum message data
- **Tech stack:** Python, scikit-learn, pandas, BeautifulSoup, Matplotlib, Seaborn, Plotly


---

## 🧬 Data Description

- `mbti_1.csv`: MBTI type (`type`) + concatenated forum posts (`posts`)
- `Users.csv`: User IDs and metadata (e.g., age, gender)
- `ForumMessages.csv`: Individual message-level data per user

---

## 🧹 Preprocessing Steps

- Cleaned HTML and URLs using `BeautifulSoup` and regex
- Removed special tokens and delimiters like `|||`
- Tokenized and vectorized text using:
  - **TF-IDF** for feature extraction
  - **Truncated SVD** for dimensionality reduction

---

## 🤖 Model Building

Tested multiple classifiers using `scikit-learn` pipelines:

- ✅ **Multinomial Naive Bayes**
- ✅ **Logistic Regression**
- ✅ **ExtraTreesClassifier** (ensemble)

### Evaluation Metrics (via cross-validation):
- **Accuracy**
- **F1-micro score**
- **Log loss**

Best result:  
🧠 **Logistic Regression — F1 Score: ~0.656, Log Loss: ~1.3**

---

## 📈 Learning Curve Analysis

Generated learning curves to assess:
- Model generalization
- Overfitting/underfitting patterns
- Impact of training set size on performance

---

## 📊 Visualization

- Distribution of MBTI types in the dataset
- Predicted MBTI types for new users (bar + pie charts)
- Trait mapping: e.g., `"INFJ"` → `"Introversion Intuition Feeling Judging"`

---

To install dependencies:

```bash
pip install -r requirements.txt
