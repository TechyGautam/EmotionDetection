# 🧠 Emotion Detection using NLP & Machine Learning

A Natural Language Processing (NLP) project that detects the **emotion expressed in a text sentence** using **TF-IDF Vectorization** and **Logistic Regression**.

The project takes raw text as input, preprocesses it, converts it into numerical features using TF-IDF, and predicts the most likely emotion.

---

## 🚀 Project Overview

Understanding emotions from text is an important NLP task with applications in:

* 💬 Chatbots
* 📞 Customer feedback analysis
* 📱 Social media sentiment & emotion analysis
* 🎧 Customer support systems
* 🤖 AI assistants

This project focuses on building an end-to-end **text classification pipeline** for emotion detection.

### 🔄 Pipeline

```text
Raw Text
   ↓
Data Exploration
   ↓
Duplicate Removal
   ↓
Text Cleaning
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Lemmatization
   ↓
Train/Test Split
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Emotion Prediction
   ↓
Model Evaluation
```

---

## 📊 Dataset

The dataset contains two columns:

| Column    | Description          |
| --------- | -------------------- |
| `text`    | Input text/sentence  |
| `emotion` | Target emotion/label |

The dataset was examined for:

* Missing values
* Duplicate records
* Class distribution
* Text quality

Duplicate records were removed before training.

---

## 🧹 Text Preprocessing

Raw text cannot be directly given to a traditional machine learning model, so several NLP preprocessing steps were applied.

### Steps performed:

1. Convert text to lowercase
2. Remove unwanted characters using Regular Expressions
3. Tokenize text into individual words
4. Remove English stopwords
5. Apply lemmatization
6. Reconstruct the cleaned sentence

Example:

```text
"I am Extremely HAPPY today!!!"
                ↓
"extremely happy today"
```

---

## 🔢 Feature Engineering — TF-IDF

After preprocessing, text was converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF assigns importance to words based on:

* How frequently a word occurs in a document
* How common or rare the word is across the complete dataset

```python
TfidfVectorizer(max_features=50000)
```

The vectorizer was fitted only on the training data and then used to transform the test data.

```text
Training Text
     ↓
TF-IDF fit_transform()
     ↓
Numerical Features
     ↓
Logistic Regression
```

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression was used as the classification model.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
```

Although its name contains "Regression", Logistic Regression is widely used for **classification problems**, especially text classification.

---

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

Example:

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Why F1-score matters?

Accuracy alone may not tell the complete story, especially when different emotion classes have different numbers of samples.

F1-score provides a balance between:

* Precision
* Recall

---

## 🧪 Example Prediction

Input:

```text
"I am extremely happy today"
```

The trained pipeline processes the sentence and predicts its corresponding emotion.

```text
Input Text
    ↓
Text Cleaning
    ↓
TF-IDF Transformation
    ↓
Logistic Regression
    ↓
Predicted Emotion
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data manipulation
* **NumPy** — Numerical operations
* **NLTK** — NLP preprocessing
* **Regular Expressions** — Text cleaning
* **Scikit-learn** — TF-IDF & Machine Learning
* **Matplotlib / Seaborn** — Data visualization

---

## 📁 Project Structure

```text
emotion-detection/
│
├── emotion.txt
├── emotion_detection.ipynb
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/TechyGautam/emotion-detection.git
cd emotion-detection
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Download the required NLTK resources:

```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
```

---

## ▶️ How to Run

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Then run the notebook cells sequentially.

---

## 💡 Key Learning Outcomes

Through this project, I worked with a complete NLP machine learning workflow:

* Data exploration and preprocessing
* Handling duplicate data
* Text normalization
* Tokenization
* Stopword removal
* Lemmatization
* Train/test splitting
* TF-IDF feature extraction
* Text classification
* Model evaluation
* Understanding classification metrics

---

## 🔮 Future Improvements

The current version uses TF-IDF + Logistic Regression. Possible improvements include:

* Compare Logistic Regression with **Naive Bayes and Linear SVM**
* Add a **Confusion Matrix**
* Perform hyperparameter tuning
* Build a Streamlit web interface
* Save and load the trained model
* Create an API using FastAPI
* Experiment with **Word Embeddings**
* Upgrade the system using **Deep Learning**
* Explore Transformer-based emotion classification

---

## 👨‍💻 Author

**Tanishk Gautam**

B.Tech CSE — AI/ML

Interested in **Artificial Intelligence, Machine Learning, NLP and Deep Learning**.

---

⭐ If you found this project useful, consider giving the repository a star!
