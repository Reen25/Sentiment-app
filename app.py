import os
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer

# Set environment variable for local NLTK data
os.environ['NLTK_DATA'] = 'nltk_data'

# Load the trained model and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to predict sentiment
def predict_sentiment(text):
    cleaned_text = clean(text)  # Use the same cleaning function as in your training script
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

# Streamlit UI
st.title("Sentiment Analysis App")

# Input text
user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Predict Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter some text.")
