# 🧠 Mental Health Self-Check App

A modern, intelligent web app that helps users evaluate their mental well-being through a combination of structured questionnaires and sentiment analysis using NLP.

---

## 💡 Project Overview

The **Mental Health Self-Check App** is a personal mental health companion that combines:
- ✅ A structured 9-question mental health assessment (Machine Learning classification)
- 💬 Journal-based sentiment analysis (using Transformers for emotional insight)
- 📊 A weighted scoring system combining both inputs
- 🧘 Dynamic mental wellness tips (meditation, breathing, motivational quotes)
- 🔁 A beautiful, modern UI built with **Streamlit**, featuring animations and pastel-glass aesthetics

This app is aimed at supporting users in performing a quick, reflective check on their current mental state.

---

## 🎯 Features

- 📝 **Personalized Journal Input** with NLP-based emotional analysis
- ❓ **9 Psychological Questions** designed to assess stress, anxiety, focus, and motivation
- 📈 **Weighted Scoring Logic** that combines both sources into an overall mental health score
- 💡 **Real-time Explainable Feedback** with visual and text-based insights
- 🌈 **Modern Pastel-Themed UI** with breathing animations and feel-good design
- 🧘‍♂️ **Well-being Tools** like breathing exercise GIFs, motivational quotes, and meditation links
- 🔄 **Retest Option** and smooth page transitions for great user experience

---

## 🔍 Technologies Used

| Area              | Stack Used                                 |
|-------------------|---------------------------------------------|
| UI / Frontend     | Streamlit, HTML/CSS                        |
| ML / NLP          | scikit-learn, Hugging Face Transformers    |
| Sentiment Model   | `cardiffnlp/twitter-roberta-base-sentiment` |
| Visualization     | Streamlit charts, markdown, emoji support  |
| Dataset           | Custom generated for stress/self-check     |

---

## 🧠 Mental Health Disclaimer

This app is built for **self-awareness** and is **not a substitute for professional diagnosis or therapy**. If you're struggling, please seek help from licensed mental health professionals.

---

## 🚀 Run Locally

```bash
git clone https://github.com/vasal666/mental-health-selfcheck.git
cd mental-health-selfcheck
pip install -r requirements.txt
streamlit run app.py

here is the public link for the website for testing
https://mental-health-selfcheck.streamlit.app/
