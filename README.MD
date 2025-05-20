# Multimodal AI Framework for Social Media Based Mental Disorder Detection and Personalized Wellbeing Insights

![Demo](https://github.com/arkapg211002/MAFSMBMDDPWI-FYP-2025/blob/main/WEB%20APP%20LANDING%20PAGE/demo.gif)

---

## Abstract

This project leverages artificial intelligence to detect early signs of mental health issues by analyzing multimodal data from social media platforms. By processing text, images, videos, audio, and PDFs, the system uses advanced machine learning and deep learning models to classify mental disorders such as depression, anxiety, bipolar disorder, and PTSD. The framework also generates personalized wellbeing insights by mapping mental health cues to Ryff’s Wellbeing Scale. With an ensemble of algorithms—including Logistic Regression, SVM, Naïve Bayes, LSTM, Transformer, and XGBoost—the system achieves high accuracy and robustness while enabling real-time web-based analysis via Streamlit.

---

## Features

- 🔍 **Multimodal Analysis:** Processes text, images, videos, audio, and PDFs.
- 📊 **Data-Driven Insights:** Uses ensemble learning for robust classification.
- ⚙️ **Advanced Preprocessing:** Cleans and tokenizes raw social media data.
- 🧠 **Diverse ML Models:** Implements Logistic Regression, Naïve Bayes, SVM, Random Forest, and XGBoost.
- 🤖 **Deep Learning Models:** LSTM-based and Transformer-based models for sequential and contextual data.
- 🌐 **Web Application:** Deployable via Streamlit for real-time analysis.
- 📈 **Performance Evaluation:** Extensive testing with cross-validation, ROC curves, and confusion matrices.
- 📋 **Wellbeing Mapping:** Uses an association matrix and a brief wellbeing survey to map predictions to Ryff’s Psychological Wellbeing dimensions. It highlights areas (e.g., purpose, autonomy) users should focus on.
- 🧠 **AI-Powered Wellbeing Insights:** Uses Gemini API to generate personalized wellbeing insights based on predictions.
- 🔁 **RAG-Style Insights:** Employs Retrieval-Augmented Generation using dynamic datasets and similarity-based input matching—with Gemini—for scalable and explainable wellbeing analysis.

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

![Streamlit app](https://github.com/arkapg211002/MAFSMBMDDPWI-FYP-2025/blob/main/WEB%20APP%20IMAGES/01%20Interface.png)

---

## How to run ?

> ⚠️ **Note:** To create a Reddit API Client ID and Secret Key, visit  
> [https://old.reddit.com/prefs/apps/](https://old.reddit.com/prefs/apps/)  
> and follow the steps below.
<!-- Step 1 -->
<p><strong>Step 1:</strong></p>
<img src="https://github.com/arkapg211002/MAFSMBMDDPWI-FYP-2025/blob/main/WEB%20APP%20LANDING%20PAGE/REDDIT1.png" width="300" />

<!-- Step 2 -->
<p><strong>Step 2:</strong></p>
<img src="https://github.com/arkapg211002/MAFSMBMDDPWI-FYP-2025/blob/main/WEB%20APP%20LANDING%20PAGE/REDDIT2.png" width="300" />

<!-- Step 3 -->
<p><strong>Step 3:</strong></p>
<img src="https://github.com/arkapg211002/MAFSMBMDDPWI-FYP-2025/blob/main/WEB%20APP%20LANDING%20PAGE/REDDIT3.png" width="300" />

---

> 📌 **Note:** To use Google AI Studio features, create a **Google API Key**  
> 👉 Visit [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey) and follow the instructions on the page.

---

> 🚨 **Note:** To access Twitter (X) API features, you must create a **Bearer Token**
> 
> 👉 First, create a profile on X <br>
> 👉 Then open [https://developer.x.com/en/portal/dashboard](https://developer.x.com/en/portal/dashboard) in a new tab and follow the steps below.

<p><strong>Step 1:</strong></p>
<img src="https://github.com/arkapg211002/MAFSMBMDDPWI-FYP-2025/blob/main/WEB%20APP%20LANDING%20PAGE/X1.png" width="300" />

<p><strong>Step 2: GENERATE BEARER TOKEN AND SAVE IT</strong></p>
<img src="https://github.com/arkapg211002/MAFSMBMDDPWI-FYP-2025/blob/main/WEB%20APP%20LANDING%20PAGE/X2.png" width="300" />

---

<div style="display: flex; gap: 10px; align-items: center;">
  <a href="https://drive.google.com/uc?export=download&id=1LguxWEB64xMTUtJ46tKPhQzZ37jGZilV" download>
    <img src="https://img.shields.io/badge/Download-Pickles_and_Required_Files-green?style=for-the-badge" />
  </a>
  <a href="https://drive.google.com/uc?export=download&id=1rCKbEAwJvjPJJICKXNYOGxI6GVJQ7kPC" download>
    <img src="https://img.shields.io/badge/Download-Notebook-blue?style=for-the-badge" />
  </a>
  <!-- Download Samples -->
  <a href="https://drive.google.com/uc?export=download&id=1VH3QKcllymxj3CSeRkWBe9XbN0PTrvs8" download>
    <img src="https://img.shields.io/badge/Download-Sample_Data-purple?style=for-the-badge" />
  </a>
</div>

> Unzip the `Pickles and Required Files` and `Samples`
> Upload `WebAppV13-6.ipynb` in drive and open it in Google Colab

