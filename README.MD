# Multimodal AI Framework for Social Media Based Mental Disorder Detection and Personalized Wellbeing Insights

---

## Abstract

This project leverages artificial intelligence to detect early signs of mental health issues by analyzing multimodal data from social media platforms. By processing text, images, videos, audio, and PDFs, the system uses advanced machine learning and deep learning models to classify mental disorders such as depression, anxiety, bipolar disorder, and PTSD. The framework also generates personalized wellbeing insights by mapping mental health cues to Ryff’s Wellbeing Scale. With an ensemble of algorithms—including Logistic Regression, SVM, Naïve Bayes, LSTM, Transformer, and XGBoost—the system achieves high accuracy and robustness while enabling real-time web-based analysis via Streamlit.


## Introduction

Social media platforms have become a mirror reflecting the mental state of their users. This project addresses the challenge of detecting mental disorders early by analyzing posts and multimedia content on platforms such as Reddit and Twitter. The multimodal approach integrates:

- **Text Analysis:** Natural Language Processing (NLP) for sentiment and context analysis.
- **Image Analysis:** Optical Character Recognition (OCR) and facial emotion detection.
- **Video & Audio Analysis:** Extraction of frames and mood analysis via audio features.
- **PDF Analysis:** Text extraction from documents for mental health cue detection.

By combining traditional machine learning with modern deep learning techniques, the system provides accurate classification and actionable wellbeing insights.

---

## Features

- 🔍 **Multimodal Analysis:** Processes text, images, videos, audio, and PDFs.
- 📊 **Data-Driven Insights:** Uses ensemble learning for robust classification.
- ⚙️ **Advanced Preprocessing:** Cleans and tokenizes raw social media data.
- 🧠 **Diverse ML Models:** Implements Logistic Regression, Naïve Bayes, SVM, Random Forest, and XGBoost.
- 🤖 **Deep Learning Models:** LSTM-based and Transformer-based models for sequential and contextual data.
- 🌐 **Web Application:** Deployable via Streamlit for real-time analysis.
- 📈 **Performance Evaluation:** Extensive testing with cross-validation, ROC curves, and confusion matrices.
- 📋 **Wellbeing Mapping:** Integrates Ryff’s Wellbeing Scale to translate predictions into personalized insights.

### Feature Comparison

| **Feature**         | **Description**                                                                               |
|---------------------|-----------------------------------------------------------------------------------------------|
| **Text Analysis**   | Classifies mental health issues from social media posts using NLP techniques.                 |
| **Image Analysis**  | Extracts text via OCR and detects emotions using DeepFace.                                    |
| **Video Analysis**  | Extracts frames, analyzes emotions, and transcribes audio for a complete multimedia assessment.|
| **Audio Analysis**  | Uses MFCC-based features to determine the mood from audio clips.                              |
| **PDF Analysis**    | Processes documents to extract and analyze text for mental health cues.                       |

---

## Tech Stack and Libraries

- **Programming Language:** Python 3.x
- **Frameworks & Environments:**  
  - TensorFlow & Keras  
  - Streamlit  
  - Google Colab (for GPU-based training)
- **Key Libraries:**  
  - Data Processing: `pandas`, `numpy`  
  - Machine Learning: `scikit-learn`, `xgboost`  
  - Deep Learning: `tensorflow`, `keras`, `transformers` (Hugging Face)  
  - NLP: `nltk`, `tiktoken`  
  - Image & Video Processing: `opencv-python`, `pytesseract`, `deepface`, `ffmpeg`  
  - Audio Processing: `librosa`, `pydub`, `SpeechRecognition`  
  - Social Media Integration: `PRAW` (Reddit), `tweepy` (Twitter)  
  - Visualization: `plotly`, `matplotlib`  
  - Other Tools: `joblib`, `protobuf`, `deep-translator`, `google-generativeai`

---
