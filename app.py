import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return ' '.join(tokens)
    else:
        return ""

# Load the model
@st.cache_resource
def load_model():
 model_path = 'C:/Users/PC/Downloads/Sentiment model/sentiment_model.pkl'
 if not os.path.exists(model_path):
        raise FileNotFoundError(f"The file {model_path} does not exist.")
 return joblib.load(model_path)

# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    # Upload CSV data
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Dataset")
        st.write(df.head())

        # Allow user to select text and label columns
        text_column = st.selectbox("Select the text column", df.columns)
        label_column = st.selectbox("Select the label column", df.columns)
        
        # Preprocess and vectorize text
        st.write("Original Text Sample: ", df[text_column].head())
        df['Processed_Text'] = df[text_column].apply(preprocess_text)
        st.write("Processed Text Sample: ", df['Processed_Text'].head())
        df = df[df['Processed_Text'].str.strip() != ""]
        
        if df['Processed_Text'].empty:
            st.error("All processed text entries are empty or invalid. Please check your data.")
            return
        
        tfidf = TfidfVectorizer(max_features=5000)
        X = tfidf.fit_transform(df['Processed_Text']).toarray()
        y = df[label_column]
        
        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Load or train the model
        model = load_model()
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        st.write(classification_report(y_test, y_pred))
        
        # User input for sentiment prediction
        st.subheader("Predict Sentiment")
        user_input = st.text_area("Enter text for sentiment analysis:")
        
        if st.button("Analyze"):
            if user_input:
                processed_input = preprocess_text(user_input)
                if processed_input.strip() == "":
                    st.write("The input text only contains stop words or invalid characters.")
                else:
                    input_vector = tfidf.transform([processed_input])
                    prediction = model.predict(input_vector)
                    st.write(f"Predicted Sentiment: {prediction[0]}")
            else:
                st.write("Please enter text for analysis.")

if __name__ == "__main__":
    main()
