# Analyzing Social Media Posts for Mental Health Disorder Detection

This repository contains a comprehensive project aimed at detecting mental health disorders through the analysis of social media posts. It leverages machine learning and deep learning techniques alongside Natural Language Processing (NLP) to classify text data into categories such as Normal, Depression, Anxiety, Bipolar Disorder, and PTSD.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Features](#project-features)
3. [Technologies Used](#technologies-used)
4. [Setup and Installation](#setup-and-installation)
5. [Usage Guide](#usage-guide)
6. [Model Architecture](#model-architecture)
7. [Dataset Information](#dataset-information)

---

## Introduction

Mental health disorders affect millions of people globally. With the rise of social media, individuals often unknowingly share insights into their mental health through their posts. This project uses AI techniques to analyze such posts and provide early detection for mental health concerns, enabling better intervention strategies.

---

## Project Features

- **Text Analysis:** Detect mental health issues based on textual inputs.
- **Image Upload Analysis:** Extract text from images, analyze it, detect any expression or gestures and classify mental health concerns.
- **Video Upload Analysis:** Convert video content to audio, process it into text, analyze video frames for text extraction, and perform facial expression recognition to aid classification.
- **Image Caption:** Added feature to generate image descriptions using transformers to aid in mental wellbeing mapping.
- **Reddit and Twitter User Analysis:** Fetch and classify posts made by Reddit or Twitter users for mental health analysis.
- **Audio Input Support:** Convert audio extracted from video to text, analyze the tone or mood using MFCC, and classify mental health concerns.
- **Multilingual Support:** Analyze posts in non-English languages by translating them to English before processing.
- **Mental Wellbeing:** Map the classified mental issues with wellbeing insights based on Ryff's Scale of Psychological Wellbeing.
- **Visualization:** Generate confusion matrices and performance metrics for each model.
<<<<<<< HEAD
- **Ensemble Model:** Leveraged high accuracy of Logistic regression and XGBoost to create an ensemble model with accuracy of 99.59% (Used this model in the web application).
||||||| 84a5052
- **Ensemble Model:** Leveraged high accuracy of Logistic regression and XGBoost to create an ensemble model with accuracy of 99.92% (Used this model in the web application).
=======
- **Ensemble Model:** Leveraged high accuracy of Logistic regression, Naive Bayes, SVM, LSTM and XGBoost to create an ensemble model with accuracy of 99.92% (Used this model in the web application).
>>>>>>> 844a25831082c230f60ca96a5513828c5983c93a
- **Model Retraining:** Allow users with the option to retrain the existing model using the input data in the application along with the result obtained.

---

## Technologies Used

### Programming Languages:
- **Python 3.x**

### Frameworks and Libraries:
- **NLP and Machine Learning**:
  - `scikit-learn`, `TensorFlow`, `transformers`, `torch`
  - `pandas`, `numpy`, `joblib`
  - `nltk`, `spaCy`, `deep-translator`
- **OCR and Translation**:
  - `pytesseract`, `Pillow`, `google-generativeai`
- **Web Deployment**:
  - `Streamlit`, `pyngrok`
- **Audio and Speech Processing**:
  - `pydub`, `sounddevice`, `SpeechRecognition`, `PyAudio`, `wavio`
  - `openai-whisper`, `librosa`, `deepface`
- **Social Media Integration**:
  - `tweepy`, `praw`
- **Video and Image Processing**:
  - `opencv-python`, `Pillow`

### Tools:
- **Development**: Google Colab, Jupyter Notebook
- **Version Control**: Git
- **Deployment**: Streamlit Cloud, ngrok

---

## Setup and Installation

### Prerequisites:
1. Python 3.8 or higher installed on your system.
2. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
3. `pip install -r requirements.txt`
4. Download Tesseract-OCR for OCR functionality: </br>
Refer to the official Tesseract installation guide for your operating system. (Follow packages.txt inside Prototype folder for the packages to be installed)

## Usage Guide

### Run the Web App:
```bash
streamlit run app.py
```

## Model Architecture

### Machine Learning Models:
- Logistic Regression (with hyperparameter tuning via RandomizedSearchCV)
- k-Nearest Neighbors (k-NN)
- Support Vector Machines (SVM)
- Naive Bayes
- Random Forest
- XGBooost

### Deep Learning Models:
- LSTM for text classification.

## Dataset Information

- **Source**: Social media posts from Reddit (25396 records).
- **Dataset File**: `mental_health.csv` (preprocessed as `preprocessed_mental_health.csv`).

### Structure:
- `text`: The post content.
- `image`: The post content.
- `video`: The uploaded file.
- `audio`: Extracted from video.
- `mental_health_issue`: Mental health category (Normal, Depression, Anxiety, Bipolar Disorder, PTSD).

![Web App](https://github.com/arkapg211002/FYP/blob/main/App%20Images/01%20Interface.png)

