# 📰 Fake vs Real News Classification and Word Cloud Visualization

This project is a **Natural Language Processing (NLP) pipeline** that classifies news articles as **Fake** or **Real** using Logistic Regression and TF-IDF features.  
It also generates **word clouds** to visualize the most common terms in each category.

---

## ✨ Features
- **ZIP file handling**: Reads CSV files directly from compressed ZIP archives.
- **Automatic labeling**: Assigns labels to fake (0) and real (1) news datasets.
- **Advanced text preprocessing**:
  - Lowercasing
  - Removing URLs, emails, and bracketed text
  - Removing punctuation and numbers
  - Stopword removal
  - Lemmatization
- **Model pipeline** with:
  - `TfidfVectorizer` (unigrams + bigrams)
  - `LogisticRegression` (balanced class weights)
- **Evaluation metrics**:
  - Accuracy
  - Precision, Recall, F1-score
- **Feature importance** listing for top fake and real indicators.
- **Word cloud visualizations** for both Fake and Real news.

---

## ⚙ Requirements

### 📌 Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud tqdm
```

### 📌 Download NLTK resources (first-time setup):
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 🚀 Usage

### 1️⃣ Prepare your dataset
- Place two ZIP files in your working directory:
  - `Fake.csv.zip` → Fake news dataset
  - `True.csv.zip` → Real news dataset  
  *(Each should contain one CSV file with a column named `text`)*

### 2️⃣ Set file paths in the script:
```python
FAKE_PATH = '/content/Fake.csv.zip'
TRUE_PATH = '/content/True.csv.zip'
```

### 3️⃣ Run the script:
```bash
python main.py
```

---

## 📊 Outputs

1. **Data loading summary** with sample counts.  
2. **Preprocessing progress bar** with `tqdm`.  
3. **Model evaluation**:
   - Accuracy score
   - Classification report
4. **Top word indicators** for Fake and Real news.
5. **Word cloud plots**:
   - Fake news common terms
   - Real news common terms

---

## 📝 Example Output

```
🔄 Loading datasets...
✅ Loaded 44898 samples (Fake: 23481, Real: 21417)
🧹 Preprocessing text...
🤖 Training model...

📊 === Final Evaluation ===
🔢 Accuracy: 98.25%
              precision    recall  f1-score   support
Fake              0.98      0.99      0.99     4696
Real              0.98      0.98      0.98     4285
```

**Top Fake Indicators:** `['allegedly', 'rumors', 'claims', ...]`  
**Top Real Indicators:** `['official', 'confirmed', 'report', ...]`  

---

## 💡 Notes
- Uses **balanced class weights** in Logistic Regression to handle class imbalance.
- TF-IDF vectorization is set to capture **unigrams and bigrams**.
- Stopwords are removed both in preprocessing and in `TfidfVectorizer`.

---

## 📄 License
This project is licensed under the MIT License.

