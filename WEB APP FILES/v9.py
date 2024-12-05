
import streamlit as st
import joblib
import pandas as pd
import praw
from PIL import Image
from deep_translator import GoogleTranslator
import requests
from io import BytesIO
from collections import Counter
import google.generativeai as genai
import pickle
import cv2
import numpy as np
import whisper
import tempfile
import os
from pydub import AudioSegment
import subprocess

import tweepy

import re
import librosa
import librosa.display
import tensorflow as tf

# Added
from scipy.stats import gaussian_kde
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import plotly.express as px

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier

from deepface import DeepFace

import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import pipeline

import pytesseract

# Configure Tesseract and FFMPEG
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"

# Load Whisper model for audio transcription
# whisper_model = whisper.load_model("base")
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# ------------- ENSEMBLE LEARNING REQUIREMENTS -----------------

@st.cache_resource
def load_lr_model():
    return joblib.load('LRmodel.pkl')

@st.cache_resource
def load_lr_vectorizer():
    return joblib.load('LRvectorizer.pkl')

@st.cache_resource
def load_xgb_model():
    return joblib.load('xgb_model.pkl')

@st.cache_resource
def load_tfidf_vectorizer():
    return joblib.load('tfidf_vectorizer.pkl')

@st.cache_resource
def load_meta_learner():
    return joblib.load('meta_learner.pkl')

@st.cache_resource
def load_label_encoder():
    return joblib.load('label_encoder.pkl')

# Load models and resources
lr_model = load_lr_model()
lr_vectorizer = load_lr_vectorizer()
xgb_model = load_xgb_model()
tfidf_vectorizer = load_tfidf_vectorizer()
meta_learner = load_meta_learner()
label_encoder = load_label_encoder()

# ------------- ENSEMBLE LEARNING REQUIREMENTS -----------------

# Initialize Reddit API
reddit = praw.Reddit(client_id='DAOso5_7CHzXzdtd-070fg',
                     client_secret='JtdGFRDM10avSQFYthzYUQNfLeI8rQ',
                     user_agent='Mental Health')

# Initialize Twitter API
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAFIXxAEAAAAALoweYwOStJHwIm0wZDWcaOEWh0M%3DmqvsIXkLqEU9CBUd5nD5HtQI3GDQu1w29EQe8vp5MufrYQRY4i"
client = tweepy.Client(bearer_token=BEARER_TOKEN)


# Configure Gemini API for wellbeing insights
genai.configure(api_key="AIzaSyD-pu0AuG2dbzzspRfgS8DjO10Ffh08JiU")
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)


# Twitter
def fetch_image_content(image_url):
    """Fetch and process an image from a URL."""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Ensure the request was successful
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.write(f"Error fetching image: {e}")
        return None

def get_latest_tweets_with_images(username, max_items=10):
    """Fetch latest tweets with text and associated images."""
    # Fetch user details to get user ID
    user = client.get_user(username=username)
    if not user.data:
        return [], []

    user_id = user.data.id

    # Fetch the latest tweets (exclude retweets and replies)
    response = client.get_users_tweets(
        id=user_id,
        tweet_fields=["attachments"],
        expansions=["attachments.media_keys"],
        media_fields=["url"],
        exclude=["retweets", "replies"],
        max_results=max_items
    )

    tweet_data = []

    if response.data:
        for tweet in response.data:
            # Extract text
            text = tweet.text

            # Extract images if available
            images = []
            if hasattr(tweet, "attachments") and tweet.attachments is not None:
                if "media_keys" in tweet.attachments:
                    for media_key in tweet.attachments["media_keys"]:
                        media = next(
                            (media for media in response.includes.get("media", []) if media["media_key"] == media_key), None
                        )
                        if media and media.type == "photo":
                            images.append(media.url)

            # Append tweet data
            tweet_data.append({"text": text, "images": images})

    return tweet_data



# Function to fetch text-based posts from Reddit
def fetch_user_text_posts(username):
    try:
        user = reddit.redditor(username)
        posts = [post.title + " " + post.selftext for post in user.submissions.new(limit=20)]
        return posts
    except Exception as e:
        st.write(f"Error fetching text posts: {e}")
        return []

# Function to fetch image-based posts from Reddit and perform OCR
def fetch_user_images_and_extract_text(username):
    try:
        user = reddit.redditor(username)
        images = [post.url for post in user.submissions.new(limit=20) if post.url.endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'))]

        extracted_texts = []
        all_emotions = {'happy': 0, 'sad': 0, 'angry': 0, 'disgust': 0, 'fear': 0, 'surprise': 0, 'neutral': 0}

        combined_caption = ""

        for image_url in images:
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Fetched Image", use_container_width=True)  # Updated to use_container_width
                # Generate and display the caption
                caption = generate_caption(image)
                st.success(caption)
                combined_caption += caption + " "

                # Extract text from the image and handle cases where extracted text is a list
                extracted_text = extract_text_from_image(image)
                if isinstance(extracted_text, list):
                    extracted_text = "\n".join(extracted_text)  # Join the list into a single string
                if extracted_text.strip():  # Strip leading/trailing spaces
                    translated_text = GoogleTranslator(source='auto', target='en').translate(extracted_text)
                    extracted_texts.append(translated_text)
                    st.write("Extracted and Translated Text from Image:")
                    st.text(translated_text)

                # Analyze facial emotions in the image
                dominant_emotion = detect_emotions_from_image(image)
                if dominant_emotion:
                    st.success(f"Dominant Emotion Detected: {dominant_emotion}")
                    all_emotions[dominant_emotion] += 1
                else:
                    st.error("No faces detected or error in emotion analysis.")
            except Exception as e:
                st.error(f"Error processing image {image_url}: {e}")

        # After processing all images, analyze the emotion counts and provide a suggestion
        if all_emotions:
            # Add graph for all_emotions
            emotion_df = pd.DataFrame(list(all_emotions.items()), columns=['Emotion', 'Count'])

            # Create and display a bar chart for all emotions
            fig = px.bar(emotion_df, x='Emotion', y='Count',
                        color='Emotion',
                        title="Aggregated Emotion Counts Across All Images",
                        labels={'Emotion': 'Detected Emotions', 'Count': 'Frequency'},
                        text='Count')  # Display count on bars for better clarity

            # Show the graph in Streamlit
            st.plotly_chart(fig)

            dominant_emotion = max(all_emotions, key=all_emotions.get)
            st.success(f"Most Frequent Emotion Across All Images or no Images(Default): {dominant_emotion}")
            emotion_summary = ", ".join([f"{emotion}: {count}" for emotion, count in all_emotions.items()])
            analyze_with_gemini(dominant_emotion, all_emotions)
        else:
            st.error("No images processed or error in emotion analysis.")

        return extracted_texts, combined_caption
    except Exception as e:
        st.error(f"Error fetching images: {e}")
        return []


# ---------------- CHANGED AS PER ENSEMBLE MODEL -----------------

# Function to classify text and display result
def classify_text(text):
    # Preprocess the input for both models
    lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
    xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost

    # Get probabilities from base models
    lr_proba = lr_model.predict_proba(lr_features)
    xgb_proba = xgb_model.predict_proba(xgb_features)

    # Combine probabilities as input for the meta-learner
    stacked_features = np.hstack((lr_proba, xgb_proba))

    # Predict using the meta-learner
    final_prediction_proba = meta_learner.predict_proba(stacked_features)
    final_prediction = meta_learner.predict(stacked_features)

    # Decode the predicted label
    top_issue = label_encoder.inverse_transform(final_prediction)[0]
    top_probability = final_prediction_proba[0].max()

    response = get_actual_issue(text,top_issue)
    if response != "" and len(response.split()) == 1 and response != top_issue:
        top_issue = response

    # Display the results
    st.success(f"The most likely mental health concern from all the text obtained is: {top_issue} with a probability of {top_probability:.2%}")

    # Pass to a custom insight function if needed
    get_wellbeing_insight(text, top_issue)

# ---------------- CHANGED AS PER ENSEMBLE MODEL -----------------

# Function to get wellbeing insights from Gemini model
def get_wellbeing_insight(text, top_issue):
    try:
        chat_session = gemini_model.start_chat(history=[])
        prompt = f""" The Ryff Scale is based on six factors: autonomy, environmental mastery, personal growth, positive relations with others, purpose in life, and self-acceptance. Higher total scores indicate higher psychological well-being. Following are explanations of each criterion, and an example statement from the Ryff Inventory to measure each criterion: Autonomy: High scores indicate that the respondent is independent and regulates his or her behavior independent of social pressures. An example statement for this criterion is "I have confidence in my opinions, even if they are contrary to the general consensus." Environmental Mastery: High scores indicate that the respondent makes effective use of opportunities and has a sense of mastery in managing environmental factors and activities, including managing everyday affairs and creating situations to benefit personal needs. An example statement for this criterion is "In general, I feel I am in charge of the situation in which I live."Personal Growth: High scores indicate that the respondent continues to develop, is welcoming to new experiences, and recognizes improvement in behavior and self over time. An example statement for this criterion is "I think it is important to have new experiences that challenge how you think about yourself and the world."Positive Relations with Others: High scores reflect the respondent's engagement in meaningful relationships with others that include reciprocal empathy, intimacy, and affection. An example statement for this criterion is "People would describe me as a giving person, willing to share my time with others." Purpose in Life: High scores reflect the respondent's strong goal orientation and conviction that life holds meaning. An example statement for this criterion is "Some people wander aimlessly through life, but I am not one of them."Self-Acceptance: High scores reflect the respondent's positive attitude about his or her self. An example statement for this criterion is "I like most aspects of my personality." Now, please use the above information and {text}, along with image captions (if added) that have been added and the mental health issue: {top_issue}, to generate a short paragraph for each of the following subtopics, discussing how the {top_issue} and {text} may relate to these factors of mental well-being: 1. **Autonomy**: How might {top_issue} impact a person's ability to be independent and self-regulate behavior? 2. **Environmental Mastery**: Discuss how {top_issue} may affect a person's ability to manage their environment and activities. 3. **Personal Growth**: What impact might {top_issue} have on an individual's development, openness to new experiences, and recognition of self-improvement? 4. **Positive Relations with Others**: How does {top_issue} influence the ability to maintain meaningful and empathetic relationships? 5. **Purpose in Life**: How might {top_issue} shape an individual's sense of purpose or goal orientation in life? 6. **Self-Acceptance**: What role does {top_issue} play in a person's self-image and acceptance of themselves? Based on the {text}, provide practical advice to improve or reduce the impact of {top_issue}."""

        response = chat_session.send_message(prompt)

        st.write("### Wellbeing Insight:")
        st.write(response.text)
    except Exception as e:
        st.write(f"Error retrieving wellbeing insights: {e}")

# Function to extract text from image using Tesseract
def extract_text_from_image(image):
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text.splitlines()

# Function to extract text from an image using Tesseract
def extract_text_from_image_video(image):
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text if extracted_text else ""  # Return empty string if no text is found


# Function to extract audio from a video file and classify it
# Function to extract 20 frames from a video file
def extract_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_interval = total_frames // num_frames  # Calculate frame interval

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)

        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames

def transcribe_audio_from_video(video_file):
    try:
        # Save the uploaded video file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(video_file.read())
            temp_video_path = temp_video_file.name

        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        # Extract audio from video using subprocess
        subprocess.run(["ffmpeg", "-i", temp_video_path, "-q:a", "0", "-map", "a", audio_path, "-y"])
        audio = AudioSegment.from_file(audio_path)

        # Use Whisper to transcribe the audio
        result = whisper_model.transcribe(audio_path)

        # Get the transcribed text and translate if necessary
        transcribed_text = result["text"]
        translated_text = GoogleTranslator(source="auto", target="en").translate(transcribed_text)

        # Clean up temporary files
        os.remove(temp_video_path)
        os.remove(audio_path)

        return translated_text

    except Exception as e:
        # Display a user-friendly message if the video is too long or another error occurs
        if "duration" in str(e).lower() or "length" in str(e).lower():
            return "The video is too long to process. Please upload a shorter video."
        else:
            return f"An error occurred: {e}"

# Function to translate text using DeepL
def translate_text(text, target_lang="en"):
    try:
        if text:
            translated_text = GoogleTranslator(source="auto", target=target_lang).translate(text)
            return translated_text
        return ""  # Return empty string if text is empty or None
    except Exception as e:
        return f"Error translating text: {str(e)}"

# Function to extract audio from a video file
def extract_audio_from_video(video_path):
    try:
        # Generate a temporary audio file path
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        # Use FFmpeg to extract audio from video
        subprocess.run(["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"])

        # Return the path of the extracted audio
        return audio_path

    except Exception as e:
        return f"Error extracting audio: {str(e)}"

# Function to analyze audio mood based on extracted audio
def analyze_audio_mood(video_path):
    try:
        # Extract audio from the video (assuming extract_audio_from_video is implemented)
        audio_path = extract_audio_from_video(video_path)

        # Load the audio file using librosa
        y, sr = librosa.load(audio_path)

        # Extract MFCCs (Mel-frequency cepstral coefficients) from the audio signal
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Divide the MFCC array into 4 frequency bands and calculate scalar mean for each band

        # Low Frequencies: MFCC 0, 1, 2
        low_freq_mfcc = np.mean(mfcc[0:3], axis=1)
        mean_low = np.mean(low_freq_mfcc)  # Scalar mean for low frequencies

        # Mid-Low Frequencies: MFCC 3, 4
        mid_low_freq_mfcc = np.mean(mfcc[3:5], axis=1)
        mean_mid_low = np.mean(mid_low_freq_mfcc)  # Scalar mean for mid-low frequencies

        # Mid-High Frequencies: MFCC 5, 6, 7
        mid_high_freq_mfcc = np.mean(mfcc[5:8], axis=1)
        mean_mid_high = np.mean(mid_high_freq_mfcc)  # Scalar mean for mid-high frequencies

        # High Frequencies: MFCC 8, 9, 10, 11, 12
        high_freq_mfcc = np.mean(mfcc[8:13], axis=1)
        mean_high = np.mean(high_freq_mfcc)  # Scalar mean for high frequencies

        # Now use these scalar means for classification

        if mean_high <= mean_low and mean_high <= mean_mid_low and mean_high <= mean_mid_high:
            return "Audio sounds normal, with no dominant emotion detected"

        elif mean_mid_high <= mean_low and mean_mid_high <= mean_mid_low and mean_mid_high <= mean_high:
            return "Audio sounds neutral, calm, or peaceful"

        elif mean_mid_low <= mean_low and mean_mid_low <= mean_mid_high and mean_mid_low <= mean_high:
            return "Audio sounds slightly melancholic or neutral"

        elif mean_low <= mean_mid_low and mean_low <= mean_mid_high and mean_low <= mean_high:
            return "Audio sounds calm or melancholic, with less intensity"

        elif mean_high > mean_low and mean_high > mean_mid_low and mean_high <= mean_mid_high:
            return "Audio sounds depressive or anxious in nature"

        else :
            return "Audio sounds upbeat and energetic (Happy)"

    except Exception as e:
        return f"Error analyzing audio mood: {str(e)}"


# ----------------- Adding Retrain Model functionality

# File path for dataset
dataset_path = 'preprocessed_mental_health.csv'
# Download stopwords (if you haven't already)
nltk.download('stopwords')
nltk.download('punkt')
# Download the 'punkt_tab' resource
nltk.download('punkt_tab')  # This line is added to download the necessary data
# Define a list of negative words to retain
negative_words = {"not", "no", "nor", "never", "nothing", "nowhere", "neither", "cannot", "n't", "without", "barely", "hardly", "scarcely"}

# Define a function to clean the text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove special characters, numbers, and punctuations
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords, but keep negative words
    tokens = [word for word in tokens if word not in stopwords.words('english') or word in negative_words]

    # Join the tokens back into a single string
    clean_text = ' '.join(tokens)

    return clean_text

def update_dataset(new_text, mental_health_issue):
    """
    Append new data to the dataset and save it.

    Args:
        new_text (str): The new text to be added.
        mental_health_issue (str): The mental health issue associated with the text.
    """
    if os.path.exists(dataset_path):
        dataset = pd.read_csv(dataset_path)
    else:
        # If the dataset doesn't exist, create a new one
        dataset = pd.DataFrame(columns=['text', 'mental_health_issue', 'cleaned_text'])

    # Clean the text (adjust cleaning based on your preprocessing)
    # cleaned_text = new_text.lower()
    cleaned_text = clean_text(new_text)

    # Create a new DataFrame for the new row
    new_row = pd.DataFrame({
        'text': [new_text],
        'mental_health_issue': [mental_health_issue],
        'cleaned_text': [cleaned_text]
    })

    # Use pd.concat to append the new row
    dataset = pd.concat([dataset, new_row], ignore_index=True)

    # Save the updated dataset
    dataset.to_csv(dataset_path, index=False)

# ---------------- CHANGED AS PER ENSEMBLE MODEL -----------------

def retrain_model():
    lr_model = None
    lr_vectorizer = None
    xgb_model = None
    label_encoder = None
    tfidf_vectorizer = None

    # Initialize Streamlit progress bar
    progress = st.progress(0)
    progress_step = 0

    with st.expander("Detailed Status", expanded=False):
      try:
          st.info("Loading models and vectorizers...")

          # Load the Logistic Regression model and vectorizer
          with open('LRmodel.pkl', 'rb') as file:
              lr_model = pickle.load(file)

          with open('LRvectorizer.pkl', 'rb') as file:
              lr_vectorizer = pickle.load(file)

          st.success("Logistic Regression model loaded successfully!")
          progress_step += 10
          progress.progress(progress_step)

          # Load the XGBoost model, label encoder, and TF-IDF vectorizer
          with open('xgb_model.pkl', 'rb') as file:
              xgb_model = pickle.load(file)

          with open('label_encoder.pkl', 'rb') as file:
              label_encoder = pickle.load(file)

          with open('tfidf_vectorizer.pkl', 'rb') as file:
              tfidf_vectorizer = pickle.load(file)

          st.success("XGBoost model loaded successfully!")
          progress_step += 10
          progress.progress(progress_step)

          # Load the updated dataset
          data = pd.read_csv(dataset_path)

          # Ensure required columns exist
          if 'cleaned_text' not in data.columns or 'mental_health_issue' not in data.columns:
              st.error("Dataset must have 'cleaned_text' and 'mental_health_issue' columns.")
              return None, None

          # Preprocessing
          data.dropna(subset=['cleaned_text'], inplace=True)

          # Split features and target
          X_test = data['cleaned_text']
          y_test = data['mental_health_issue']

          # Encode target labels
          y_test = label_encoder.transform(y_test)

          st.info("Dataset loaded and processed successfully!")
          progress_step += 10
          progress.progress(progress_step)

          # Transform the text using the respective vectorizers
          X_test_lr = lr_vectorizer.transform(X_test)  # Logistic Regression vectorizer
          X_test_xgb = tfidf_vectorizer.transform(X_test)  # XGBoost vectorizer

          st.success("Text processed successfully!")
          progress_step += 10
          progress.progress(progress_step)

          st.info("Predicting using base models...")

          # Get predictions from the base models
          lr_predictions_proba = lr_model.predict_proba(X_test_lr)
          xgb_predictions_proba = xgb_model.predict_proba(X_test_xgb)

          # Combine predictions as new features
          stacked_features = np.hstack((lr_predictions_proba, xgb_predictions_proba))

          st.success("Predictions generated successfully!")
          progress_step += 10
          progress.progress(progress_step)

          # Train meta-learner using combined features (optional step if not pre-trained)
          X_train_meta, y_train_meta = stacked_features, y_test  # Example using test data as meta-training data

          meta_learner = LogisticRegression(max_iter=5000)
          meta_learner.fit(X_train_meta, y_train_meta)

          st.success("Meta-learner trained successfully!")
          progress_step += 10
          progress.progress(progress_step)

          # Save the trained meta-learner
          with open('meta_learner.pkl', 'wb') as file:
              pickle.dump(meta_learner, file)

          # Load the pre-trained meta-learner
          with open('meta_learner.pkl', 'rb') as file:
              meta_learner = pickle.load(file)

          # Predict using the meta-learner
          final_predictions = meta_learner.predict(stacked_features)

          # Evaluate the ensemble model
          accuracy = accuracy_score(y_test, final_predictions)

          # Celebrate successful execution
          progress.progress(100)
          st.success("Execution completed successfully!")
          st.balloons()

          return meta_learner, accuracy
      except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    


# ---------------- CHANGED AS PER ENSEMBLE MODEL -----------------

def update_and_retrain(example_text, example_issue):
    """
    Updates the dataset with new data, displays the last three rows,
    and retrains the Logistic Regression model.

    Args:
        example_text (str): The new input text to be added.
        example_issue (str): The predicted mental health issue to be added.
    """
    try:
        # Update dataset
        update_dataset(example_text, example_issue)

        # Display the last three rows of the updated dataset
        dataset = pd.read_csv(dataset_path)
        st.write("Updated Dataset (Last 3 Rows):")
        st.write(dataset.tail(3))

        # Notify user about retraining process
        st.info("Model is being retrained...")

        # Retrain model
        model, accuracy = retrain_model()

        # Display retraining success message
        if model:
            # st.success(f"Model retrained successfully! Accuracy: {accuracy * 100:.2f}%")
            st.metric(label="Retrained Model Accuracy", value=f"{accuracy * 100:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# ---------------- CHANGED AS PER ENSEMBLE MODEL -----------------

# Function to classify text, display result and retrain model
def classify_text_retrain_model(text):
    # Preprocess the input for both models
    lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
    xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost

    # Get probabilities from base models
    lr_proba = lr_model.predict_proba(lr_features)
    xgb_proba = xgb_model.predict_proba(xgb_features)

    # Combine probabilities as input for the meta-learner
    stacked_features = np.hstack((lr_proba, xgb_proba))

    # Predict using the meta-learner
    final_prediction_proba = meta_learner.predict_proba(stacked_features)
    final_prediction = meta_learner.predict(stacked_features)

    # Decode the predicted label
    top_issue = label_encoder.inverse_transform(final_prediction)[0]
    top_probability = final_prediction_proba[0].max()

    response = get_actual_issue(text,top_issue)
    if response != "" and len(response.split()) == 1 and response != top_issue:
        top_issue = response

    # Display the results
    st.success(f"The most likely mental health concern from all the text obtained is: {top_issue} with a probability of {top_probability:.2%}")

    # Pass to a custom insight function if needed
    get_wellbeing_insight(text, top_issue)

    # Adding Model Retraining Functionality
    update_and_retrain(text, top_issue)

# ---------------- CHANGED AS PER ENSEMBLE MODEL -----------------

# ----------------- Adding Retrain Model functionality


# ---------------------- Adding Facial Recognition for video

def detect_emotions_from_frame(frame):
    try:
        # Use DeepFace to analyze emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception as e:
        print(f"No expression or error detecting emotion: {e}")
        return None

def analyze_emotions_from_frames(frames):
    emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0, 'disgust': 0, 'fear': 0, 'surprise': 0, 'neutral': 0}
    frame_emotions = []

    for idx, frame in enumerate(frames):
        emotion = detect_emotions_from_frame(frame)
        if emotion:
            frame_emotions.append(emotion)
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1

    return emotion_counts, frame_emotions

def display_emotion_summary(emotion_counts):
    # Convert the emotion counts to a DataFrame for display
    # Convert the emotion counts to a DataFrame for display and plotting
    emotion_df = pd.DataFrame(list(emotion_counts.items()), columns=['Emotion', 'Count'])
    st.write("Emotion Analysis Summary:")

    # Add a bar chart for emotion counts
    fig = px.bar(emotion_df, x='Emotion', y='Count',
                 color='Emotion',
                 title="Emotion Counts",
                 labels={'Emotion': 'Detected Emotions', 'Count': 'Frequency'})
    st.plotly_chart(fig)
    return max(emotion_counts, key=emotion_counts.get)

def analyze_with_gemini(dominant_emotion, emotion_counts):
    """
    Analyze the detected emotions using Gemini API with exception handling and chat session.

    Parameters:
        dominant_emotion (str): The most frequent emotion detected across frames.
        emotion_counts (dict): A dictionary with emotion types as keys and their counts as values.

    Returns:
        str: The response text from the Gemini API or an error message if the API call fails.
    """
    try:
        # Start a chat session with the Gemini API
        chat_session = gemini_model.start_chat(history=[])

        # Create a detailed summary of emotion counts for the prompt
        emotion_summary = ", ".join([f"{emotion}: {count}" for emotion, count in emotion_counts.items()])

        # Craft the prompt with the dominant emotion and emotion summary
        # prompt = (
           # f"The detected dominant emotion is '{dominant_emotion}', and the counts for each emotion are as follows: {emotion_summary}. "
          #  f"Analyze this data in the context of possible mental health issues (e.g., depression, anxiety, PTSD, or bipolar) and provide a suggestion."
        #)
        prompt = f"The detected dominant emotion is '{dominant_emotion}'. {emotion_summary}. Based on this information, analyze the potential implications for mental health conditions such as depression, anxiety, PTSD, or bipolar disorder. Provide insights into how these emotions might relate to these mental health issues and suggest actionable advice or strategies to improve mental well-being. Give only three lines."


        # Send the prompt via the chat session
        response = chat_session.send_message(prompt)
        st.write(response.text)
    except Exception as e:
        # Log the error (optional, for debugging purposes)
        print(f"Error in Gemini API call: {e}")

        # Return a user-friendly error message
        return "An error occurred while communicating with the Gemini API. Please try again later."



# ---------------------- Adding Facial Recognition for video

# ---------------------- Adding Facial Recognition for image

def detect_emotions_from_image(image):
    """
    Detect the dominant emotion from a single image using DeepFace.

    Parameters:
        image (PIL.Image.Image or np.ndarray): The input image to analyze.

    Returns:
        str: The dominant emotion detected, or None if no emotion is detected.
    """
    try:
        # Convert PIL image to NumPy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Use DeepFace to analyze emotions
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception as e:
        print(f"No expression or error detecting emotion: {e}")
        return None

def analyze_emotions_from_image(image):
    """
    Analyze emotions from an image and count occurrences of each emotion.

    Parameters:
        image (PIL.Image.Image or np.ndarray): The input image to analyze.

    Returns:
        Tuple[Dict[str, int], List[str]]:
            - A dictionary with emotion counts.
            - A list of detected emotions for each face.
    """
    try:
        # Convert PIL image to NumPy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Initialize emotion counters
        emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0, 'disgust': 0, 'fear': 0, 'surprise': 0, 'neutral': 0}
        detected_emotions = []

        # Use DeepFace to detect and analyze faces
        results = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)

        # Initialize a DataFrame to store emotion probabilities
        emotion_data = []

        # If results contain multiple faces, process each one
        for result in results:
            if 'emotion' in result:
                emotion_data.append(result['emotion'])

        # Convert emotion data to a DataFrame
        emotion_df = pd.DataFrame(emotion_data)

        # Calculate mean probabilities for each emotion (if multiple faces are detected)
        mean_emotions = emotion_df.mean().reset_index()
        mean_emotions.columns = ['Emotion', 'Average Probability']

        # Create and display a bar chart
        fig = px.bar(mean_emotions, x='Emotion', y='Average Probability',
                    color='Emotion',
                    title="Average Emotion Probabilities from Analyzed Faces",
                    labels={'Emotion': 'Detected Emotions', 'Average Probability': 'Probability'})
        st.plotly_chart(fig)

        # If results contain multiple faces, process each one
        for result in results:
            emotion = result.get('dominant_emotion')
            if emotion:
                detected_emotions.append(emotion)
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1

        return emotion_counts, detected_emotions
    except Exception as e:
        print(f"Error analyzing emotions from image: {e}")
        return {}, []


# ---------------------- Adding Facial Recognition for image


# ---------------------- Get Image Description

# Function to load the model (cached for efficiency)
@st.cache_resource
def load_model():
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, feature_extractor, tokenizer

# Load the model
IDmodel, IDfeature_extractor, IDtokenizer = load_model()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDmodel.to(device)

# Function to generate caption
def generate_caption(image):
    # Preprocess the image
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = IDfeature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate caption (you can adjust max_length and num_beams as needed)
    with torch.no_grad():
        output_ids = IDmodel.generate(pixel_values, max_length=16, num_beams=4)
    caption = IDtokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# ---------------- CHANGED AS PER ENSEMBLE MODEL -----------------

def classify_text_with_desc(text,text2):
    # Preprocess the input for both models
    lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
    xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost

    # Get probabilities from base models
    lr_proba = lr_model.predict_proba(lr_features)
    xgb_proba = xgb_model.predict_proba(xgb_features)

    # Combine probabilities as input for the meta-learner
    stacked_features = np.hstack((lr_proba, xgb_proba))

    # Predict using the meta-learner
    final_prediction_proba = meta_learner.predict_proba(stacked_features)
    final_prediction = meta_learner.predict(stacked_features)

    # Decode the predicted label
    top_issue = label_encoder.inverse_transform(final_prediction)[0]
    top_probability = final_prediction_proba[0].max()

    response = get_actual_issue(text+" "+text2,top_issue)
    if response != "" and len(response.split()) == 1 and response != top_issue:
        top_issue = response

    # Display the results
    st.success(f"The most likely mental health concern from all the text obtained is: {top_issue} with a probability of {top_probability:.2%}")

    get_wellbeing_insight(text+" "+text2, top_issue)

def classify_text_retrain_model_desc(text,text2):
    # Preprocess the input for both models
    lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
    xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost

    # Get probabilities from base models
    lr_proba = lr_model.predict_proba(lr_features)
    xgb_proba = xgb_model.predict_proba(xgb_features)

    # Combine probabilities as input for the meta-learner
    stacked_features = np.hstack((lr_proba, xgb_proba))

    # Predict using the meta-learner
    final_prediction_proba = meta_learner.predict_proba(stacked_features)
    final_prediction = meta_learner.predict(stacked_features)

    # Decode the predicted label
    top_issue = label_encoder.inverse_transform(final_prediction)[0]
    top_probability = final_prediction_proba[0].max()

    response = get_actual_issue(text+" "+text2,top_issue)
    if response != "" and len(response.split()) == 1 and response != top_issue:
        top_issue = response

    # Display the results
    st.success(f"The most likely mental health concern from all the text obtained is: {top_issue} with a probability of {top_probability:.2%}")

    get_wellbeing_insight(text+" "+text2, top_issue)

    # Adding Model Retraining Functionality
    update_and_retrain(text, top_issue)

# ---------------- CHANGED AS PER ENSEMBLE MODEL -----------------


# ---------------------- Get Image Description


# ---------------------- Get Video Description

# Function to load the model (cached for efficiency)
@st.cache_resource
def load_image_captioning_model():
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, feature_extractor, tokenizer

@st.cache_resource
def load_summary_pipeline():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Load models
caption_model, Vfeature_extractor, Vtokenizer = load_image_captioning_model()
summary_pipeline = load_summary_pipeline()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)

# Function to generate captions for an image
def generate_caption_video(image):
    # Convert the numpy array (video frame) to a PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    # Preprocess the image
    pixel_values = Vfeature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate caption
    with torch.no_grad():
        output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=4)
    caption = Vtokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Function to generate an overall description of the video
def describe_video(frames):
    # Generate captions for each frame
    captions = [generate_caption_video(frame) for frame in frames]
    combined_captions = " ".join(captions)

    # Summarize the captions to get an overall description
    summary = summary_pipeline(combined_captions, max_length=50, min_length=25, do_sample=False)[0]["summary_text"]
    return captions, summary

# ---------------------- Get Video Description
def get_actual_issue(text,top_issue):
    try:
        chat_session = gemini_model.start_chat(history=[])
        prompt = f"The given text is : {text}. Is this statement normal or depression or ptsd or anxiety or bipolar? I need only one word answer. I need no explanation. My model gave {top_issue}. I want to confirm. Please give the exactly correct one word answer about what you think."
        response = chat_session.send_message(prompt)
        return response.text.strip().lower()
    except Exception as e:
        print(f"Error: {e}")
        return ""

# Define the Streamlit app
def run_app():
    st.title("Mental Health Disorder Detection")

    option = st.sidebar.selectbox(
        "Choose an option",
        ["Text Input", "Image Upload", "Video Upload", "Reddit Username Analysis", "Twitter Username Analysis"]
    )

    # Text Input
    if option == "Text Input":
        st.subheader("Enter Text to Classify Mental Health Issue")
        input_text = st.text_area("Enter your text here:")

        if st.button("Classify Text"):
            if input_text.strip() == "":
                st.write("Please enter some text to classify.")
            else:
                translated_text = GoogleTranslator(source='auto', target='en').translate(input_text)
                st.write("Translated Text (to English):")
                st.write(translated_text)
                classify_text(translated_text)
        # Adding model retraining
        elif st.button("Classify Text and Retrain Model"):
            if input_text.strip() == "":
                st.write("Please enter some text to classify.")
            else:
                translated_text = GoogleTranslator(source='auto', target='en').translate(input_text)
                st.subheader("Translated Text (to English):")
                st.write(translated_text)
                classify_text_retrain_model(translated_text)


    # Image Upload
    elif option == "Image Upload":
        st.subheader("Upload an Image to Extract and Classify Text")
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Generate and display the caption
            caption = generate_caption(image)
            st.success(caption)

            # Step 1: Extract text from the image
            extracted_text = extract_text_from_image(image)
            translated_text = GoogleTranslator(source='auto', target='en').translate("\n".join(extracted_text))

            st.subheader("Translated Text (to English)")
            st.text(translated_text)

            # Step 2: Detect faces and analyze emotions
            # st.write("Detecting faces and analyzing emotions...")
            emotion_counts, detected_emotions = analyze_emotions_from_image(image)

            if emotion_counts:
                # Display emotion counts in a table
                # st.write("Detected Emotions:")
                # st.table(pd.DataFrame.from_dict(emotion_counts, orient="index", columns=["Count"]))

                # Determine the dominant emotion
                dominant_emotion = max(emotion_counts, key=emotion_counts.get)
                st.success(f"Dominant Emotion: **{dominant_emotion}**")

                # Step 2: Use Gemini API for mental health analysis
                # st.write("### Facial Recognition Insight")
                analyze_with_gemini(dominant_emotion, emotion_counts)

            else:
                st.error("No faces detected in the uploaded image.")

            # Step 3: Classify extracted text
            if st.button("Classify Extracted Text"):
                if not translated_text or translated_text.strip() == "":
                    st.write("It is normal with probability 100%")
                else:
                    classify_text_with_desc(translated_text,caption)

            # Adding model retraining option
            elif st.button("Classify Extracted Text and Retrain Model"):
                if not translated_text or translated_text.strip() == "":
                    st.write("It is normal with probability 100%")
                else:
                    classify_text_retrain_model_desc(translated_text,caption)


    # Video Upload
    elif option == "Video Upload":
        st.subheader("Upload a Video to Extract and Classify Text")
        # File upload widget
        video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

        if video_file:
            # Save the uploaded video file temporarily
            video_path = "/tmp/uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())

            st.video(video_file)  # Display the uploaded video

            # Extract frames from the uploaded video
            frames = extract_frames(video_path)
            combined_text = ""

            st.subheader("Extracting frames from video...")

            # Emotion recognition
            emotion_counts, frame_emotions = analyze_emotions_from_frames(frames)

            # Display results
            for idx, emotion in enumerate(frame_emotions):
                st.image(frames[idx], caption=f"Frame {idx + 1} - Emotion: {emotion}")

            # Display summary table and most frequent emotion
            dominant_emotion = display_emotion_summary(emotion_counts)
            st.success(f"Dominant Emotion: **{dominant_emotion}**")

            # Use the dominant emotion and emotion counts to craft a Gemini API prompt
            # st.write("### Facial Recognition Insight")
            analyze_with_gemini(dominant_emotion, emotion_counts)

            # st.write("Gemini API Response:")
            #st.text(gemini_response)

            for idx, frame in enumerate(frames):
                # st.image(frame, caption=f"Frame {idx + 1}", use_column_width=True)
                text_from_frame = extract_text_from_image_video(frame)

                if text_from_frame and text_from_frame not in combined_text:
                    combined_text += text_from_frame + " "

            # Generate and display descriptions
            frame_captions, overall_description = describe_video(frames)
            st.subheader("Overall Description")
            st.success(overall_description)


            st.subheader("Text Extracted from Video Frames:")
            st.text(combined_text)

            # Translate the extracted text from frames
            translated_frame_text = translate_text(combined_text)
            # st.write("Translated Text from Video Frames:")
            # st.text(translated_frame_text)

            # Extract audio and transcribe it
            # st.write("Transcribing Audio from Video...")
            transcribed_audio_text = transcribe_audio_from_video(video_file)

            st.subheader("Transcribed Audio Text:")
            st.text(transcribed_audio_text)

            translated_audio_text = translate_text(transcribed_audio_text)
            # st.write("Translated Audio Text:")
            # st.text(translated_audio_text)

            # Combine the text extracted from both images and audio
            full_combined_text = combined_text + " " + transcribed_audio_text
            st.subheader("Combined Extracted Text (from both video frames and audio):")
            st.text(full_combined_text)

            translated_combined_text = translate_text(full_combined_text)
            st.subheader("Translated Combined Text (Frames + Audio):")
            st.text(translated_combined_text)

            # Analyze audio mood
            st.subheader("Analyzing Audio Mood...")
            mood_result = analyze_audio_mood(video_path)
            st.write(mood_result)

            cleaned_text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", translated_combined_text)

            if st.button("Classify Extracted Text"):
                if not cleaned_text or cleaned_text.strip() == "":
                    # If audio_text is empty or contains only whitespace
                    st.write("It is normal with probability 100%")
                else:
                    classify_text_with_desc(cleaned_text,overall_description)
            # Adding model retraining
            elif st.button("Classify Extracted Text and Retrain Model"):
                if not cleaned_text or cleaned_text.strip() == "":
                    # If audio_text is empty or contains only whitespace
                    st.write("It is normal with probability 100%")
                else:
                    classify_text_retrain_model_desc(cleaned_text,overall_description)


    # Reddit Username Analysis
    elif option == "Reddit Username Analysis":
        st.subheader("Enter Reddit Username for Analysis")
        username = st.text_input("Enter Reddit username:")

        if st.button("Analyze"):
            if username.strip() == "":
                st.write("Please enter a Reddit username.")
            else:
                # Fetch and display text posts
                text_posts = fetch_user_text_posts(username)
                if text_posts:
                    st.write("Recent Text Posts:")
                    st.write(text_posts[:3])  # Display a few posts for review

                # Fetch and display image-based posts with extracted text
                image_texts, image_caption = fetch_user_images_and_extract_text(username)

                # Combine text from both text posts and image text
                all_text = text_posts + image_texts
                if all_text:
                    predictions = []
                    probabilities = []
                    issue_probabilities = defaultdict(float)  # To store the sum of probabilities for each issue

                    for text in all_text:
                        # Preprocess the text for both models
                        lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
                        xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost

                        # Get probabilities from base models
                        lr_proba = lr_model.predict_proba(lr_features)
                        xgb_proba = xgb_model.predict_proba(xgb_features)

                        # Combine probabilities as input for the meta-learner
                        stacked_features = np.hstack((lr_proba, xgb_proba))

                        # Predict using the meta-learner
                        final_prediction_proba = meta_learner.predict_proba(stacked_features)
                        final_prediction = meta_learner.predict(stacked_features)
                        decoded_prediction = label_encoder.inverse_transform(final_prediction)[0]

                        # Append the prediction
                        predictions.append(decoded_prediction)
                        probabilities.append(final_prediction_proba.max())  # Highest probability

                        # Add the probability to the sum for the respective issue
                        issue_probabilities[decoded_prediction] += final_prediction_proba.max()

                    # Count the most common mental health issue
                    issue_counts = Counter(predictions)
                    top_issue, top_count = issue_counts.most_common(1)[0]
                    top_percentage = (top_count / len(predictions)) * 100

                    response = get_actual_issue(" ".join(all_text)+" Image captions are as follows : "+image_caption,top_issue)
                    if response != "" and len(response.split()) == 1 and response != top_issue:
                        top_issue = response

                    # Display results
                    st.success(f"The most frequently detected mental health concern from all the text obtained is: {top_issue} with a probability of{top_percentage:.2f}% from the analyzed text.")
                    issue_distribution = pd.DataFrame(issue_counts.items(), columns=['Mental Health Issue', 'Count'])
                    st.write("Mental health issue distribution across posts:")
                    st.write(issue_distribution)

                    # Add a bar chart
                    st.bar_chart(issue_distribution.set_index('Mental Health Issue')['Count'])

                    # Call the Gemini model to get well-being insights
                    get_wellbeing_insight(" ".join(all_text)+" Image captions are as follows : "+image_caption, top_issue)

                    # Adding Model Retraining Functionality
                    # update_and_retrain(" ".join(all_text), top_issue)

                else:
                    st.write("No valid text found for analysis.")
        # Adding Model Retraining
        elif st.button("Analyze and Retrain Model"):
            if username.strip() == "":
                st.write("Please enter a Reddit username.")
            else:
                # Fetch and display text posts
                text_posts = fetch_user_text_posts(username)
                if text_posts:
                    st.write("Recent Text Posts:")
                    st.write(text_posts[:3])  # Display a few posts for review

                # Fetch and display image-based posts with extracted text
                image_texts, image_caption = fetch_user_images_and_extract_text(username)

                # Combine text from both text posts and image text
                all_text = text_posts + image_texts
                if all_text:
                    predictions = []
                    probabilities = []
                    issue_probabilities = defaultdict(float)  # To store the sum of probabilities for each issue

                    for text in all_text:
                        # Preprocess the text for both models
                        lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
                        xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost

                        # Get probabilities from base models
                        lr_proba = lr_model.predict_proba(lr_features)
                        xgb_proba = xgb_model.predict_proba(xgb_features)

                        # Combine probabilities as input for the meta-learner
                        stacked_features = np.hstack((lr_proba, xgb_proba))

                        # Predict using the meta-learner
                        final_prediction_proba = meta_learner.predict_proba(stacked_features)
                        final_prediction = meta_learner.predict(stacked_features)
                        decoded_prediction = label_encoder.inverse_transform(final_prediction)[0]

                        # Append the prediction
                        predictions.append(decoded_prediction)
                        probabilities.append(final_prediction_proba.max())  # Highest probability

                        # Add the probability to the sum for the respective issue
                        issue_probabilities[decoded_prediction] += final_prediction_proba.max()

                    # Count the most common mental health issue
                    issue_counts = Counter(predictions)
                    top_issue, top_count = issue_counts.most_common(1)[0]
                    top_percentage = (top_count / len(predictions)) * 100

                    response = get_actual_issue(" ".join(all_text)+" Image captions are as follows : "+image_caption,top_issue)
                    if response != "" and len(response.split()) == 1 and response != top_issue:
                        top_issue = response

                    # Display results
                    st.success(f"The most frequently detected mental health concern from all the text obtained is: {top_issue} with a probability of{top_percentage:.2f}% from the analyzed text.")
                    issue_distribution = pd.DataFrame(issue_counts.items(), columns=['Mental Health Issue', 'Count'])
                    st.write("Mental health issue distribution across posts:")
                    st.write(issue_distribution)

                    # Add a bar chart
                    st.bar_chart(issue_distribution.set_index('Mental Health Issue')['Count'])

                    # Call the Gemini model to get well-being insights
                    get_wellbeing_insight(" ".join(all_text)+" Image captions are as follows : "+image_caption, top_issue)

                    # Adding Model Retraining Functionality
                    update_and_retrain(" ".join(all_text), top_issue)

                else:
                    st.write("No valid text found for analysis.")


    # Twitter Username Analysis
    elif option == "Twitter Username Analysis":
        st.subheader("Enter Twitter Username for Analysis")
        username = st.text_input("Enter Twitter username:")

        if st.button("Analyze"):
            if username.strip() == "":
                st.write("Please enter a Twitter username.")
            else:
                # Fetch the latest tweets with associated images
                tweets_with_images = get_latest_tweets_with_images(username)

                # Extract text content from tweets
                text_posts = [tweet['text'] for tweet in tweets_with_images if tweet['text']]
                st.write("Recent Text Posts from Tweets:")
                st.write(text_posts[:3])  # Display a few posts for review

                # Extract and process text from associated images
                image_texts = []
                all_emotions = {'happy': 0, 'sad': 0, 'angry': 0, 'disgust': 0, 'fear': 0, 'surprise': 0, 'neutral': 0}
                combined_caption = ""
                for tweet in tweets_with_images:
                    for image_url in tweet['images']:
                        image = fetch_image_content(image_url)
                        if image:
                            st.image(image, caption=f"Image from Tweet", use_container_width=True)
                            # Generate and display the caption
                            caption = generate_caption(image)
                            st.success(caption)
                            combined_caption += caption + " "
                        if image:
                            extracted_text = extract_text_from_image(image)  # Assuming a text extraction function is defined
                            if extracted_text:
                                image_texts.append(extracted_text)
                            # Analyze facial emotions in the image
                            dominant_emotion = detect_emotions_from_image(image)
                            if dominant_emotion:
                                st.success(f"Dominant Emotion Detected: {dominant_emotion}")
                                all_emotions[dominant_emotion] += 1
                            else:
                                st.error("No faces detected or error in emotion analysis.")

                # After processing all images, analyze the emotion counts and provide a suggestion
                if all_emotions:
                    # Convert `all_emotions` to a DataFrame
                    emotion_df = pd.DataFrame(list(all_emotions.items()), columns=['Emotion', 'Count'])

                    # Create and display a bar chart for all emotions
                    fig = px.bar(emotion_df, x='Emotion', y='Count',
                                color='Emotion',
                                title="Aggregated Emotion Counts Across All Images",
                                labels={'Emotion': 'Detected Emotions', 'Count': 'Frequency'},
                                text='Count')  # Display count on bars for better clarity

                    # Show the graph in Streamlit
                    st.plotly_chart(fig)

                    dominant_emotion = max(all_emotions, key=all_emotions.get)
                    st.success(f"Most Frequent Emotion Across All Images or no Images(Default): {dominant_emotion}")
                    emotion_summary = ", ".join([f"{emotion}: {count}" for emotion, count in all_emotions.items()])
                    analyze_with_gemini(dominant_emotion, all_emotions)
                else:
                    st.error("No images processed or error in emotion analysis.")

                # Combine text from both tweet text and extracted image text
                all_text = text_posts + image_texts

                # Ensure all entries in all_text are strings
                all_text = [str(text) for text in all_text if text]

                if all_text:
                    predictions = []
                    probabilities = []
                    issue_probabilities = defaultdict(float)  # To store the sum of probabilities for each issue
                    for text in all_text:
                        try:
                            # Preprocess the text for both models
                            lr_features = lr_vectorizer.transform([text])  # Logistic Regression
                            xgb_features = tfidf_vectorizer.transform([text])  # XGBoost

                            # Get probabilities from base models
                            lr_proba = lr_model.predict_proba(lr_features)
                            xgb_proba = xgb_model.predict_proba(xgb_features)

                            # Combine probabilities as input for the meta-learner
                            stacked_features = np.hstack((lr_proba, xgb_proba))

                            # Predict using the meta-learner
                            final_prediction_proba = meta_learner.predict_proba(stacked_features)
                            final_prediction = meta_learner.predict(stacked_features)
                            decoded_prediction = label_encoder.inverse_transform(final_prediction)[0]

                            # Append the prediction
                            predictions.append(decoded_prediction)
                            probabilities.append(final_prediction_proba.max())  # Highest probability

                            # Add the probability to the sum for the respective issue
                            issue_probabilities[decoded_prediction] += final_prediction_proba.max()
                        except Exception as e:
                            st.write(f"Error processing text: {text[:50]}... - {e}")
                            continue

                    # Count the most common mental health issue
                    issue_counts = Counter(predictions)
                    top_issue, top_count = issue_counts.most_common(1)[0]
                    top_percentage = (top_count / len(predictions)) * 100

                    response = get_actual_issue(" ".join(all_text)+" The image captions are as follows :  "+combined_caption,top_issue)
                    if response != "" and len(response.split()) == 1 and response != top_issue:
                        top_issue = response

                    st.success(f"The most frequently detected mental health concern obtained from all the text obtained is: {top_issue}, with a probability of {top_percentage:.2f}% from the analyzed text.")
                    issue_distribution = pd.DataFrame(issue_counts.items(), columns=['Mental Health Issue', 'Count'])
                    st.write("Mental health issue distribution across posts:")
                    st.write(issue_distribution)

                    # Add a bar chart
                    st.bar_chart(issue_distribution.set_index('Mental Health Issue')['Count'])

                    # Call the Gemini model to get well-being insights
                    get_wellbeing_insight(" ".join(all_text)+" The image captions are as follows :  "+combined_caption, top_issue)

                    # Adding Model Retraining Functionality
                    # update_and_retrain(" ".join(all_text), top_issue)

                else:
                    st.write("No valid text found for analysis.")

        # Adding Retrain Model
        elif st.button("Analyze and Retrain Model"):
            if username.strip() == "":
                st.write("Please enter a Twitter username.")
            else:
                # Fetch the latest tweets with associated images
                tweets_with_images = get_latest_tweets_with_images(username)

                # Extract text content from tweets
                text_posts = [tweet['text'] for tweet in tweets_with_images if tweet['text']]
                st.write("Recent Text Posts from Tweets:")
                st.write(text_posts[:3])  # Display a few posts for review

                # Extract and process text from associated images
                image_texts = []
                all_emotions = {'happy': 0, 'sad': 0, 'angry': 0, 'disgust': 0, 'fear': 0, 'surprise': 0, 'neutral': 0}
                combined_caption = ""
                for tweet in tweets_with_images:
                    for image_url in tweet['images']:
                        image = fetch_image_content(image_url)
                        if image:
                            st.image(image, caption=f"Image from Tweet", use_container_width=True)
                            # Generate and display the caption
                            caption = generate_caption(image)
                            st.success(caption)
                            combined_caption += caption + " "
                        if image:
                            extracted_text = extract_text_from_image(image)  # Assuming a text extraction function is defined
                            if extracted_text:
                                image_texts.append(extracted_text)
                            # Analyze facial emotions in the image
                            dominant_emotion = detect_emotions_from_image(image)
                            if dominant_emotion:
                                st.success(f"Dominant Emotion Detected: {dominant_emotion}")
                                all_emotions[dominant_emotion] += 1
                            else:
                                st.error("No faces detected or error in emotion analysis.")

                # After processing all images, analyze the emotion counts and provide a suggestion
                if all_emotions:
                    # Convert `all_emotions` to a DataFrame
                    emotion_df = pd.DataFrame(list(all_emotions.items()), columns=['Emotion', 'Count'])

                    # Create and display a bar chart for all emotions
                    fig = px.bar(emotion_df, x='Emotion', y='Count',
                                color='Emotion',
                                title="Aggregated Emotion Counts Across All Images",
                                labels={'Emotion': 'Detected Emotions', 'Count': 'Frequency'},
                                text='Count')  # Display count on bars for better clarity

                    # Show the graph in Streamlit
                    st.plotly_chart(fig)

                    dominant_emotion = max(all_emotions, key=all_emotions.get)
                    st.success(f"Most Frequent Emotion Across All Images or no Images(Default): {dominant_emotion}")
                    emotion_summary = ", ".join([f"{emotion}: {count}" for emotion, count in all_emotions.items()])
                    analyze_with_gemini(dominant_emotion, all_emotions)
                else:
                    st.error("No images processed or error in emotion analysis.")

                # Combine text from both tweet text and extracted image text
                all_text = text_posts + image_texts

                # Ensure all entries in all_text are strings
                all_text = [str(text) for text in all_text if text]

                if all_text:
                    predictions = []
                    probabilities = []
                    issue_probabilities = defaultdict(float)  # To store the sum of probabilities for each issue
                    for text in all_text:
                        try:
                            # Preprocess the text for both models
                            lr_features = lr_vectorizer.transform([text])  # Logistic Regression
                            xgb_features = tfidf_vectorizer.transform([text])  # XGBoost

                            # Get probabilities from base models
                            lr_proba = lr_model.predict_proba(lr_features)
                            xgb_proba = xgb_model.predict_proba(xgb_features)

                            # Combine probabilities as input for the meta-learner
                            stacked_features = np.hstack((lr_proba, xgb_proba))

                            # Predict using the meta-learner
                            final_prediction_proba = meta_learner.predict_proba(stacked_features)
                            final_prediction = meta_learner.predict(stacked_features)
                            decoded_prediction = label_encoder.inverse_transform(final_prediction)[0]

                            # Append the prediction
                            predictions.append(decoded_prediction)
                            probabilities.append(final_prediction_proba.max())  # Highest probability

                            # Add the probability to the sum for the respective issue
                            issue_probabilities[decoded_prediction] += final_prediction_proba.max()
                        except Exception as e:
                            st.write(f"Error processing text: {text[:50]}... - {e}")
                            continue

                    # Count the most common mental health issue
                    issue_counts = Counter(predictions)
                    top_issue, top_count = issue_counts.most_common(1)[0]
                    top_percentage = (top_count / len(predictions)) * 100

                    response = get_actual_issue(" ".join(all_text)+" The image captions are as follows :  "+combined_caption,top_issue)
                    if response != "" and len(response.split()) == 1 and response != top_issue:
                        top_issue = response

                    st.success(f"The most frequently detected mental health concern obtained from all the text obtained is: {top_issue}, with a probability of {top_percentage:.2f}% from the analyzed text.")
                    issue_distribution = pd.DataFrame(issue_counts.items(), columns=['Mental Health Issue', 'Count'])
                    st.write("Mental health issue distribution across posts:")
                    st.write(issue_distribution)

                    # Add a bar chart
                    st.bar_chart(issue_distribution.set_index('Mental Health Issue')['Count'])

                    # Call the Gemini model to get well-being insights
                    get_wellbeing_insight(" ".join(all_text)+" The image options are as follows :  "+combined_caption, top_issue)

                    # Adding Model Retraining Functionality
                    update_and_retrain(" ".join(all_text), top_issue)

                else:
                    st.write("No valid text found for analysis.")


# Run the app
if __name__ == '__main__':
    run_app()
