
import pickle
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

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.utils import pad_sequences, custom_object_scope, to_categorical
from tensorflow.keras.layers import MultiHeadAttention, Input, Dense, Embedding, GlobalAveragePooling1D, LayerNormalization, Layer
from tensorflow.keras.optimizers import Adam

from xgboost import XGBClassifier

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

# Define functions to load each model and resource with caching
@st.cache_resource
def load_lr_model():
    return joblib.load('LRmodel.pkl')

@st.cache_resource
def load_lr_vectorizer():
    return joblib.load('LRvectorizer.pkl')

@st.cache_resource
def load_nb_model():
    return joblib.load('NBmodel.pkl')

@st.cache_resource
def load_nb_vectorizer():
    return joblib.load('NBvectorizer.pkl')

@st.cache_resource
def load_svm_model():
    return joblib.load('SVMmodel.pkl')

@st.cache_resource
def load_svm_vectorizer():
    return joblib.load('SVMvectorizer.pkl')

@st.cache_resource
def load_label_encoder():
    return joblib.load('label_encoder.pkl')

@st.cache_resource
def load_xgb_model():
    return joblib.load('xgb_model.pkl')

@st.cache_resource
def load_tfidf_vectorizer():
    return joblib.load('tfidf_vectorizer.pkl')

@st.cache_resource
def load_lstm_label_encoder():
    return joblib.load('LSTM_label_encoder.pkl')

@st.cache_resource
def load_lstm_tokenizer():
    return joblib.load('LSTM_tokenizer.pkl')

@st.cache_resource
def load_lstm_model():
    return load_model('lstm_model.h5')

@st.cache_resource
def load_meta_learner_rf():
    return joblib.load('meta_learner_rf.pkl')

@st.cache_resource
def load_transformer_label_encoder():
    return joblib.load('Tlabel_encoder.pkl')

@st.cache_resource
def load_transformer_vectorizer():
    return joblib.load('Tvectorizer_layer.pkl')




# Load all models and resources by calling the above functions
lr_model = load_lr_model()
lr_vectorizer = load_lr_vectorizer()
nb_model = load_nb_model()
nb_vectorizer = load_nb_vectorizer()
svm_model = load_svm_model()
svm_vectorizer = load_svm_vectorizer()
label_encoder = load_label_encoder()
xgb_model = load_xgb_model()
tfidf_vectorizer = load_tfidf_vectorizer()
lstm_label_encoder = load_lstm_label_encoder()  # Optional: Only if separate from main label encoder
lstm_tokenizer = load_lstm_tokenizer()
lstm_model = load_lstm_model()
meta_learner_rf = load_meta_learner_rf()

t_label_encoder = load_transformer_label_encoder()
t_vectorizer = load_transformer_vectorizer()

# Define the custom layers
class EmbeddingLayer(Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.word_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embedding = Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, tokens):
        sequence_length = tf.shape(tokens)[-1]
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        positions_encoding = self.position_embedding(positions)
        words_encoding = self.word_embedding(tokens)
        return positions_encoding + words_encoding


class EncoderLayer(Layer):
    def __init__(self, total_heads, total_dense_units, embed_dim, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.multihead = MultiHeadAttention(num_heads=total_heads, key_dim=embed_dim)
        self.nnw = Sequential([Dense(total_dense_units, activation="relu"), Dense(embed_dim)])
        self.normalize_layer = LayerNormalization()

    def call(self, inputs):
        attn_output = self.multihead(inputs, inputs)
        normalize_attn = self.normalize_layer(inputs + attn_output)
        nnw_output = self.nnw(normalize_attn)
        final_output = self.normalize_layer(normalize_attn + nnw_output)
        return final_output

# Load the saved transformer model with custom objects
custom_objects = {
    "EmbeddingLayer": EmbeddingLayer,
    "EncoderLayer": EncoderLayer
}

@st.cache_resource
def load_transformer_model():
    return load_model('Ttransformer_model.h5', custom_objects=custom_objects)

transformer_model = load_transformer_model()


# ------------- ENSEMBLE LEARNING REQUIREMENTS -----------------

# Initialize Reddit API
reddit = praw.Reddit(client_id='DAOso5_7CHzXzdtd-070fg',
                     client_secret='JtdGFRDM10avSQFYthzYUQNfLeI8rQ',
                     user_agent='Mental Health')

# Initialize Twitter API
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAADhOxAEAAAAA5L4saD5wsvoQlB2qrUtYkmJ0DiY%3Dmjn2X0dux3zvrsY7j0HYwN95UGDEHPWQrm7EWX63nhW77NXSNE"
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
    # Preprocess the input for each base model
    lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
    svm_features = svm_vectorizer.transform([text])  # For SVM
    nb_features = nb_vectorizer.transform([text])  # For Naive Bayes
    xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost
    lstm_features = lstm_tokenizer.texts_to_sequences([text])  # For LSTM
    transformer_features = t_vectorizer([text])  # For Transformer

    # Pad sequences for LSTM
    lstm_features = pad_sequences(lstm_features, maxlen=100, padding='post', truncating='post')

    # Get probabilities from all base models
    lr_proba = lr_model.predict_proba(lr_features)
    svm_proba = svm_model.predict_proba(svm_features)
    nb_proba = nb_model.predict_proba(nb_features)
    xgb_proba = xgb_model.predict_proba(xgb_features)
    lstm_proba = lstm_model.predict(lstm_features)
    transformer_proba = transformer_model.predict(transformer_features)

    # Combine probabilities as input for the meta-learner
    stacked_features = np.hstack((lr_proba, svm_proba, nb_proba, xgb_proba, lstm_proba, transformer_proba))

    # Predict using the meta-learner
    final_prediction_proba = meta_learner_rf.predict_proba(stacked_features)
    final_prediction = meta_learner_rf.predict(stacked_features)

    # Decode the predicted label
    top_issue = label_encoder.inverse_transform(final_prediction)[0]
    top_probability = final_prediction_proba[0].max()

    response = get_actual_issue(text,top_issue)
    if response != "" and len(response.split()) == 1 and response != top_issue:
        top_issue = response

    # Display the results
    st.success(f"The most likely mental health concern from the text provided is: {top_issue} with a probability of {top_probability:.2%}")

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
    # Load Logistic Regression model and vectorizer
    with open('LRmodel.pkl', 'rb') as file:
        lr_model = pickle.load(file)

    with open('LRvectorizer.pkl', 'rb') as file:
        lr_vectorizer = pickle.load(file)

    # Load SVM model and vectorizer
    with open('SVMmodel.pkl', 'rb') as file:
        svm_model = pickle.load(file)

    with open('SVMvectorizer.pkl', 'rb') as file:
        svm_vectorizer = pickle.load(file)

    # Load XGBoost model, vectorizer, and label encoder
    with open('xgb_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)

    with open('tfidf_vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)

    # Load LSTM model, tokenizer, and label encoder
    lstm_model = load_model('lstm_model.h5')

    with open('LSTM_tokenizer.pkl', 'rb') as file:
        lstm_tokenizer = pickle.load(file)

    # Load Naive Bayes model and vectorizer
    with open('NBmodel.pkl', 'rb') as file:
        nb_model = pickle.load(file)

    with open('NBvectorizer.pkl', 'rb') as file:
        nb_vectorizer = pickle.load(file)

    # Load the transformer model and associated files
    with open('Tlabel_encoder.pkl', 'rb') as file:
        t_label_encoder = pickle.load(file)

    with open('Tvectorizer_layer.pkl', 'rb') as file:
        t_vectorize_layer = pickle.load(file)

    transformer_model = load_model('Ttransformer_model.h5', custom_objects=custom_objects)

    # Load the test dataset
    data = pd.read_csv('preprocessed_mental_health.csv')

    # Check if 'cleaned_text' column exists
    if 'cleaned_text' not in data.columns:
        raise ValueError("The dataset must have a 'cleaned_text' column.")

    # Remove rows with missing values in 'cleaned_text'
    data.dropna(subset=['cleaned_text'], inplace=True)

    # Split features and target
    X_test = data['cleaned_text']
    y_test = data['mental_health_issue']

    # Encode target labels
    y_test = label_encoder.transform(y_test)

    # Process the text for each model
    X_test_lr = lr_vectorizer.transform(X_test)  # Logistic Regression vectorizer
    X_test_svm = svm_vectorizer.transform(X_test)  # SVM vectorizer
    X_test_xgb = tfidf_vectorizer.transform(X_test)  # XGBoost vectorizer
    X_test_nb = nb_vectorizer.transform(X_test)  # Naive Bayes vectorizer
    X_test_lstm = lstm_tokenizer.texts_to_sequences(X_test)  # LSTM tokenizer
    # Preprocess the text for the transformer model
    X_test_transformer = t_vectorize_layer(X_test)

    # Pad sequences for LSTM
    X_test_lstm = pad_sequences(X_test_lstm, maxlen=100, padding='post', truncating='post')

    # Get predictions from the base models
    lr_predictions_proba = lr_model.predict_proba(X_test_lr)  # Logistic Regression probabilities
    svm_predictions_proba = svm_model.predict_proba(X_test_svm)  # SVM probabilities
    xgb_predictions_proba = xgb_model.predict_proba(X_test_xgb)  # XGBoost probabilities
    nb_predictions_proba = nb_model.predict_proba(X_test_nb)  # Naive Bayes probabilities
    lstm_predictions_proba = lstm_model.predict(X_test_lstm)  # LSTM probabilities
    # Get probabilities from the transformer model
    transformer_predictions_proba = transformer_model.predict(X_test_transformer)

    # Stack the predictions of all models to create the feature matrix for the meta-learner
    stacked_features = np.hstack((
        lr_predictions_proba,
        svm_predictions_proba,
        xgb_predictions_proba,
        nb_predictions_proba,
        lstm_predictions_proba,
        transformer_predictions_proba
    ))

    # Split data into training and test sets
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
    stacked_features, y_test, test_size=0.2, random_state=42, stratify=y_test
    )

    # Train Random Forest as the meta-learner
    meta_learner_rf = RandomForestClassifier(
      max_depth=None,            # Maximum depth of each tree
      min_samples_split=20,      # Minimum number of samples to split a node

      min_samples_leaf=1,        # Minimum number of samples in a leaf node
      max_features='sqrt',       # Number of features to consider at each split
      bootstrap=False,            # Whether to use bootstrapping

      random_state=42            # For reproducibility
    )
    meta_learner_rf.fit(X_train1, y_train1)

    # Save the trained XGBoost meta-learner
    with open('meta_learner_rf.pkl', 'wb') as file:
        pickle.dump(meta_learner_rf, file)

    # Predict using the XGBoost meta-learner
    final_predictions_lr = meta_learner_rf.predict(X_test1)

    # Evaluate the XGBoost ensemble model
    accuracy_rf = accuracy_score(y_test1, final_predictions_lr)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform stratified cross-validation and compute accuracies
    cross_val_accuracies = []
    for train_index, test_index in skf.split(stacked_features, y_test):
        X_train_fold, X_test_fold = stacked_features[train_index], stacked_features[test_index]
        y_train_fold, y_test_fold = y_test[train_index], y_test[test_index]

        meta_learner_rf.fit(X_train_fold, y_train_fold)
        fold_accuracy = meta_learner_rf.score(X_test_fold, y_test_fold)
        cross_val_accuracies.append(fold_accuracy)

    # Convert to NumPy array for consistency
    cross_val_accuracies = np.array(cross_val_accuracies)

    # Calculate mean and standard deviation
    mean_val_accuracy = np.mean(cross_val_accuracies)
    std_val_accuracy = np.std(cross_val_accuracies)

    return meta_learner_rf, mean_val_accuracy

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
            st.success(f"Model retrained successfully! Accuracy: {accuracy * 100:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# ---------------- CHANGED AS PER ENSEMBLE MODEL -----------------

# Function to classify text, display result and retrain model
def classify_text_retrain_model(text):
    # Preprocess the input for each base model
    lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
    svm_features = svm_vectorizer.transform([text])  # For SVM
    nb_features = nb_vectorizer.transform([text])  # For Naive Bayes
    xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost
    lstm_features = lstm_tokenizer.texts_to_sequences([text])  # For LSTM
    transformer_features = t_vectorizer([text])  # For Transformer

    # Pad sequences for LSTM
    lstm_features = pad_sequences(lstm_features, maxlen=100, padding='post', truncating='post')

    # Get probabilities from all base models
    lr_proba = lr_model.predict_proba(lr_features)
    svm_proba = svm_model.predict_proba(svm_features)
    nb_proba = nb_model.predict_proba(nb_features)
    xgb_proba = xgb_model.predict_proba(xgb_features)
    lstm_proba = lstm_model.predict(lstm_features)
    transformer_proba = transformer_model.predict(transformer_features)

    # Combine probabilities as input for the meta-learner
    stacked_features = np.hstack((lr_proba, svm_proba, nb_proba, xgb_proba, lstm_proba, transformer_proba))

    # Predict using the meta-learner
    final_prediction_proba = meta_learner_rf.predict_proba(stacked_features)
    final_prediction = meta_learner_rf.predict(stacked_features)

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
    emotion_df = pd.DataFrame(list(emotion_counts.items()), columns=['Emotion', 'Count'])
    st.write("Emotion Analysis Summary:")
    st.table(emotion_df)
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
def load_model_img():
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, feature_extractor, tokenizer

# Load the model
IDmodel, IDfeature_extractor, IDtokenizer = load_model_img()

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
    # Preprocess the input for each base model
    lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
    svm_features = svm_vectorizer.transform([text])  # For SVM
    nb_features = nb_vectorizer.transform([text])  # For Naive Bayes
    xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost
    lstm_features = lstm_tokenizer.texts_to_sequences([text])  # For LSTM
    transformer_features = t_vectorizer([text])  # For Transformer

    # Pad sequences for LSTM
    lstm_features = pad_sequences(lstm_features, maxlen=100, padding='post', truncating='post')

    # Get probabilities from all base models
    lr_proba = lr_model.predict_proba(lr_features)
    svm_proba = svm_model.predict_proba(svm_features)
    nb_proba = nb_model.predict_proba(nb_features)
    xgb_proba = xgb_model.predict_proba(xgb_features)
    lstm_proba = lstm_model.predict(lstm_features)
    transformer_proba = transformer_model.predict(transformer_features)

    # Combine probabilities as input for the meta-learner
    stacked_features = np.hstack((lr_proba, svm_proba, nb_proba, xgb_proba, lstm_proba, transformer_proba))

    # Predict using the meta-learner
    final_prediction_proba = meta_learner_rf.predict_proba(stacked_features)
    final_prediction = meta_learner_rf.predict(stacked_features)

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
    # Preprocess the input for each base model
    lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
    svm_features = svm_vectorizer.transform([text])  # For SVM
    nb_features = nb_vectorizer.transform([text])  # For Naive Bayes
    xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost
    lstm_features = lstm_tokenizer.texts_to_sequences([text])  # For LSTM
    transformer_features = t_vectorizer([text])  # For Transformer

    # Pad sequences for LSTM
    lstm_features = pad_sequences(lstm_features, maxlen=100, padding='post', truncating='post')

    # Get probabilities from all base models
    lr_proba = lr_model.predict_proba(lr_features)
    svm_proba = svm_model.predict_proba(svm_features)
    nb_proba = nb_model.predict_proba(nb_features)
    xgb_proba = xgb_model.predict_proba(xgb_features)
    lstm_proba = lstm_model.predict(lstm_features)
    transformer_proba = transformer_model.predict(transformer_features)

    # Combine probabilities as input for the meta-learner
    stacked_features = np.hstack((lr_proba, svm_proba, nb_proba, xgb_proba, lstm_proba, transformer_proba))

    # Predict using the meta-learner
    final_prediction_proba = meta_learner_rf.predict_proba(stacked_features)
    final_prediction = meta_learner_rf.predict(stacked_features)

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
                st.table(pd.DataFrame.from_dict(emotion_counts, orient="index", columns=["Count"]))

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

                    for text in all_text:
                        # Preprocess the input for each base model
                        lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
                        svm_features = svm_vectorizer.transform([text])  # For SVM
                        nb_features = nb_vectorizer.transform([text])  # For Naive Bayes
                        xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost
                        lstm_features = lstm_tokenizer.texts_to_sequences([text])  # For LSTM
                        transformer_features = t_vectorizer([text])  # For Transformer

                        # Pad sequences for LSTM
                        lstm_features = pad_sequences(lstm_features, maxlen=100, padding='post', truncating='post')

                        # Get probabilities from all base models
                        lr_proba = lr_model.predict_proba(lr_features)
                        svm_proba = svm_model.predict_proba(svm_features)
                        nb_proba = nb_model.predict_proba(nb_features)
                        xgb_proba = xgb_model.predict_proba(xgb_features)
                        lstm_proba = lstm_model.predict(lstm_features)
                        transformer_proba = transformer_model.predict(transformer_features)

                        # Combine probabilities as input for the meta-learner
                        stacked_features = np.hstack((lr_proba, svm_proba, nb_proba, xgb_proba, lstm_proba, transformer_proba))

                        # Predict using the meta-learner
                        final_prediction_proba = meta_learner_rf.predict_proba(stacked_features)
                        final_prediction = meta_learner_rf.predict(stacked_features)
                        decoded_prediction = label_encoder.inverse_transform(final_prediction)[0]

                        # Append the prediction
                        predictions.append(decoded_prediction)

                    # Count the most common mental health issue
                    issue_counts = Counter(predictions)
                    top_issue, top_count = issue_counts.most_common(1)[0]
                    top_percentage = (top_count / len(predictions)) * 100

                    response = get_actual_issue(" ".join(all_text)+" Image captions are as follows : "+image_caption,top_issue)
                    if response != "" and len(response.split()) == 1 and response != top_issue:
                        top_issue = response

                    # Display results
                    st.success(f"The most frequently detected mental health concern from all the text obtained is: {top_issue} with a probability of {top_percentage:.2f}% from the analyzed text.")
                    issue_distribution = pd.DataFrame(issue_counts.items(), columns=['Mental Health Issue', 'Count'])
                    st.write("Mental health issue distribution across posts:")
                    st.write(issue_distribution)

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
                    for text in all_text:
                        # Preprocess the input for each base model
                        lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
                        svm_features = svm_vectorizer.transform([text])  # For SVM
                        nb_features = nb_vectorizer.transform([text])  # For Naive Bayes
                        xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost
                        lstm_features = lstm_tokenizer.texts_to_sequences([text])  # For LSTM
                        transformer_features = t_vectorizer([text])  # For Transformer

                        # Pad sequences for LSTM
                        lstm_features = pad_sequences(lstm_features, maxlen=100, padding='post', truncating='post')

                        # Get probabilities from all base models
                        lr_proba = lr_model.predict_proba(lr_features)
                        svm_proba = svm_model.predict_proba(svm_features)
                        nb_proba = nb_model.predict_proba(nb_features)
                        xgb_proba = xgb_model.predict_proba(xgb_features)
                        lstm_proba = lstm_model.predict(lstm_features)
                        transformer_proba = transformer_model.predict(transformer_features)

                        # Combine probabilities as input for the meta-learner
                        stacked_features = np.hstack((lr_proba, svm_proba, nb_proba, xgb_proba, lstm_proba, transformer_proba))

                        # Predict using the meta-learner
                        final_prediction_proba = meta_learner_rf.predict_proba(stacked_features)
                        final_prediction = meta_learner_rf.predict(stacked_features)
                        decoded_prediction = label_encoder.inverse_transform(final_prediction)[0]

                        # Append the prediction
                        predictions.append(decoded_prediction)

                    # Count the most common mental health issue
                    issue_counts = Counter(predictions)
                    top_issue, top_count = issue_counts.most_common(1)[0]
                    top_percentage = (top_count / len(predictions)) * 100

                    response = get_actual_issue(" ".join(all_text)+" Image captions are as follows : "+image_caption,top_issue)
                    if response != "" and len(response.split()) == 1 and response != top_issue:
                        top_issue = response

                    # Display results
                    st.success(f"The most frequently detected mental health concern from all the text obtained is: {top_issue} with a probability of {top_percentage:.2f}% from the analyzed text.")
                    issue_distribution = pd.DataFrame(issue_counts.items(), columns=['Mental Health Issue', 'Count'])
                    st.write("Mental health issue distribution across posts:")
                    st.write(issue_distribution)

                    # Call the Gemini model to get well-being insights
                    get_wellbeing_insight(" ".join(all_text)+" The image captions are as follows : "+image_caption, top_issue)

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
                    for text in all_text:
                        try:
                            # Preprocess the input for each base model
                            lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
                            svm_features = svm_vectorizer.transform([text])  # For SVM
                            nb_features = nb_vectorizer.transform([text])  # For Naive Bayes
                            xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost
                            lstm_features = lstm_tokenizer.texts_to_sequences([text])  # For LSTM
                            transformer_features = t_vectorizer([text])  # For Transformer

                            # Pad sequences for LSTM
                            lstm_features = pad_sequences(lstm_features, maxlen=100, padding='post', truncating='post')

                            # Get probabilities from all base models
                            lr_proba = lr_model.predict_proba(lr_features)
                            svm_proba = svm_model.predict_proba(svm_features)
                            nb_proba = nb_model.predict_proba(nb_features)
                            xgb_proba = xgb_model.predict_proba(xgb_features)
                            lstm_proba = lstm_model.predict(lstm_features)
                            transformer_proba = transformer_model.predict(transformer_features)

                            # Combine probabilities as input for the meta-learner
                            stacked_features = np.hstack((lr_proba, svm_proba, nb_proba, xgb_proba, lstm_proba, transformer_proba))

                            # Predict using the meta-learner
                            final_prediction_proba = meta_learner_rf.predict_proba(stacked_features)
                            final_prediction = meta_learner_rf.predict(stacked_features)
                            decoded_prediction = label_encoder.inverse_transform(final_prediction)[0]

                            # Append the prediction
                            predictions.append(decoded_prediction)
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
                    for text in all_text:
                        try:
                            # Preprocess the input for each base model
                            lr_features = lr_vectorizer.transform([text])  # For Logistic Regression
                            svm_features = svm_vectorizer.transform([text])  # For SVM
                            nb_features = nb_vectorizer.transform([text])  # For Naive Bayes
                            xgb_features = tfidf_vectorizer.transform([text])  # For XGBoost
                            lstm_features = lstm_tokenizer.texts_to_sequences([text])  # For LSTM
                            transformer_features = t_vectorizer([text])  # For Transformer

                            # Pad sequences for LSTM
                            lstm_features = pad_sequences(lstm_features, maxlen=100, padding='post', truncating='post')

                            # Get probabilities from all base models
                            lr_proba = lr_model.predict_proba(lr_features)
                            svm_proba = svm_model.predict_proba(svm_features)
                            nb_proba = nb_model.predict_proba(nb_features)
                            xgb_proba = xgb_model.predict_proba(xgb_features)
                            lstm_proba = lstm_model.predict(lstm_features)
                            transformer_proba = transformer_model.predict(transformer_features)

                            # Combine probabilities as input for the meta-learner
                            stacked_features = np.hstack((lr_proba, svm_proba, nb_proba, xgb_proba, lstm_proba, transformer_proba))

                            # Predict using the meta-learner
                            final_prediction_proba = meta_learner_rf.predict_proba(stacked_features)
                            final_prediction = meta_learner_rf.predict(stacked_features)
                            decoded_prediction = label_encoder.inverse_transform(final_prediction)[0]

                            # Append the prediction
                            predictions.append(decoded_prediction)
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

                    st.success(f"The most frequently detected mental health concern from all the text obtained is: {top_issue}, with a probability of {top_percentage:.2f}% from the analyzed text.")
                    issue_distribution = pd.DataFrame(issue_counts.items(), columns=['Mental Health Issue', 'Count'])
                    st.write("Mental health issue distribution across posts:")
                    st.write(issue_distribution)

                    # Call the Gemini model to get well-being insights
                    get_wellbeing_insight(" ".join(all_text)+" The image options are as follows :  "+combined_caption, top_issue)

                    # Adding Model Retraining Functionality
                    update_and_retrain(" ".join(all_text), top_issue)

                else:
                    st.write("No valid text found for analysis.")


# Run the app
if __name__ == '__main__':
    run_app()
