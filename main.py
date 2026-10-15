import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

st.title("Emotion Detection App")

pd.set_option('display.max_rows', None)
df = pd.read_csv("emotion.txt",sep=";",names=["text","emotion"])

# df.head(10)

df["emotion"].value_counts()
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
# df.head()
# Start Text cleaning 


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def text_cleaning(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)
df["text"] = df["text"].apply(text_cleaning)
# df.head()
### Split data for tain and test
from sklearn.model_selection import train_test_split

x = df["text"]
y = df["emotion"]

x_train,x_test,y_train,y_test =train_test_split(
    x,y, 
    test_size = 0.2 , 
    random_state = 42, 
    stratify=y
)
## Vectorization 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=50000)

X_train_vec = vectorizer.fit_transform(x_train)
X_test_vec = vectorizer.transform(x_test)
print(X_train_vec.shape)
print(X_test_vec.shape)

print(y_train.shape)
print(y_test.shape)
# Model Training
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)
model.score(X_test_vec, y_test)
y_pred  = model.predict(X_test_vec)
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("\nClassification Report:\n")
# print(classification_report(y_test, y_pred))
text = ["I will beat you today because i am frustrated ","I am feel sad today"]

txt = st.text_input("Enter your text here")

cleaned_text = [text_cleaning(txt)]

text_vec = vectorizer.transform(cleaned_text)

prediction = model.predict(text_vec)

if st.button("Predict"):
    st.write("Predicted Emotion: " + prediction[0])

print(prediction)