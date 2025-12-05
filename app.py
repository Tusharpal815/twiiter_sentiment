import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from dotenv import load_dotenv
import os
import nltk
import requests


load_dotenv()
# ---------------------------
# Load stopwords (cached)
# ---------------------------
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# ---------------------------
# Load model and vectorizer (cached)
# ---------------------------
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# ---------------------------
# Predict sentiment
# ---------------------------
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text_vector = vectorizer.transform([text])
    sentiment = model.predict(text_vector)
    return "Positive" if sentiment == 1 else "Negative"

# ---------------------------
# Twitter API setup
# ---------------------------
bearer_token = os.getenv("BEARER_TOKEN")
# ---------------------------
# Fetch tweets with caching (no retry/sleep)
# ---------------------------
@st.cache_data(ttl=300)  # Cache results for 5 minutes
def get_tweets(username, count=10):
    # Ensure count is within Twitter API limits
    if count < 10:
        count = 10
    elif count > 100:
        count = 100

    headers = {"Authorization": f"Bearer {bearer_token}"}
    url = f"https://api.twitter.com/2/tweets/search/recent?query=from:{username}&tweet.fields=text&max_results={count}"

    tweets = []
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for tweet in data.get("data", []):
                tweets.append(tweet["text"])
        elif response.status_code == 429:
            st.warning("⚠️ Rate limit exceeded. Try again later.")
            # No sleep/retry for faster performance
        else:
            st.error(f"Error fetching tweets: {response.status_code} — {response.text}")
    except Exception as e:
        st.error(f"Error connecting to Twitter API: {e}")

    return tweets

# ---------------------------
# Create colored sentiment cards
# ---------------------------
def create_card(tweet_text, sentiment):
    color = "#2ecc71" if sentiment == "Positive" else "#e74c3c"
    card_html = f"""
    <div style="background-color: {color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

# ---------------------------
# Main Streamlit app
# ---------------------------
def main():
    st.title("Twitter Sentiment Analysis")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])

    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            if text_input.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                st.success(f"Sentiment: {sentiment}")

    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username (without @)")
        if st.button("Fetch Tweets"):
            if username.strip() == "":
                st.warning("Please enter a Twitter username.")
            else:
                st.info(f"Fetching latest tweets from @{username}...")
                with st.spinner('Fetching tweets, please wait...'):
                    tweets_data = get_tweets(username, count=10)
                if not tweets_data:
                    st.error("⚠️ No tweets found or unable to fetch data. Try another username.")
                else:
                    st.success(f"Showing latest {len(tweets_data)} tweets from @{username}:")
                    for tweet_text in tweets_data:
                        sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                        card_html = create_card(tweet_text, sentiment)
                        st.markdown(card_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
