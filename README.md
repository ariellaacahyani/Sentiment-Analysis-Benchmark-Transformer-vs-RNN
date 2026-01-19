# Sentiment Analysis Benchmark: Transformer (IndoBERT) vs RNN (LSTM/GRU) 🚀

![Language](https://img.shields.io/badge/Language-Python-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20HuggingFace-orange)
![Rating](https://img.shields.io/badge/Rating-⭐⭐⭐⭐⭐-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Project Achievement:** This project was developed as part of the **Dicoding Deep Learning** course and achieved a perfect **5-star rating (Distinction)** from the reviewer.

## 📌 Project Overview
This project focuses on **Multi-class Sentiment Analysis** (Positive, Neutral, Negative) on Indonesian travel app reviews. The primary goal is to benchmark the performance of traditional Deep Learning architectures (**LSTM, GRU**) against State-of-the-Art Transformer models (**IndoBERT**).

The dataset consists of **64,000+ scraped reviews**, making it a robust case study for handling real-world, noisy text data (slang, abbreviations, and mixed languages).

## ✨ Key Features
* **Comprehensive Benchmark:** Comparison of three distinct architectures:
    * **GRU** (Gated Recurrent Unit)
    * **LSTM** (Long Short-Term Memory)
    * **IndoBERT** (Fine-tuned BERT for Indonesian Language)
* **Large-Scale Dataset:** Utilized 64k data points scraped independently.
* **Advanced Preprocessing:** Implementation of text cleaning, stopwords removal (Sastrawi), and tokenization.
* **Robust Training:** Implemented `EarlyStopping` and `ModelCheckpoint` to prevent overfitting.
* **High Performance:** Achieved **>92% accuracy** on the Test Set using the Transformer architecture.

## 📊 Experiment Results

| Model Architecture | Feature Extraction | Test Accuracy | Observations |
| :--- | :--- | :--- | :--- |
| **GRU** | Word Embedding | ~95% | Fast training, good baseline. |
| **LSTM** | Word Embedding | ~95% | Slightly better context handling than GRU. |
| **IndoBERT** 🏆 | **BERT Tokenizer** | **96%** | **Best Performance.** Superior in understanding context and slang. |

> *Note: While RNNs (GRU/LSTM) are lighter and faster, IndoBERT proved to be significantly more robust in capturing semantic meaning in Indonesian text.*

## 🚀 Roadmap & Future Improvements
Based on expert feedback, the following advanced techniques are planned for implementation in upcoming projects:

* **Pipeline Automation:** Implementing `tf.data` or Scikit-Learn Pipelines to automate the workflow from preprocessing to evaluation efficiently.
* **Advanced Data Handling:** Restructuring the workflow to perform augmentation and scaling *after* splitting the dataset to strictly prevent **Data Leakage**.
* **Explainable AI (XAI):** Integrating tools like **SHAP** or **LIME** to improve model interpretability (visualizing which features influence predictions the most).
* **Experiment Tracking:** Utilizing **MLflow** or **Weights & Biases** to track hyperparameters and model versions systematically.

## 🛠️ Tech Stack
* **Core:** Python, TensorFlow, Keras
* **NLP & Transformer:** Hugging Face (`transformers`), Sastrawi
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn, WordCloud

## 📥 How to Run
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ariellaacahyani/Sentiment-Analysis-Benchmark-Transformer-vs-RNN
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Notebook:**
    Open `notebook.ipynb` in Jupyter Notebook or Google Colab.

---
**Created by [Ariella Cahyani]**
