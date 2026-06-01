# 🎬 Movie Review Sentiment Analysis

Deep Learning sentiment classifier comparing **SimpleRNN**, **LSTM**, and **GRU** on the IMDB dataset, deployed with a Streamlit dashboard.

---

## 📁 Project Structure

```
MovieReview-LSTM_RNN_GRU/
├── notebooks/
│   └── Movie_Review(RNN_vs_LSTM_vs_GRU).ipynb
├── models/                  # created after training
├── app.py                   # Streamlit dashboard
├── train_models.py          # train & save all 3 models
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (run once)
python train_models.py

# 3. Launch app
streamlit run app.py
```

---

## �️ Dashboard

- **Predict** — enter a review, get sentiment + confidence from any model
- **Compare** — run all three models side-by-side on the same review
- **Metrics** — accuracy, loss curves, training time, parameter count

---

## 🛠️ Tech Stack

`TensorFlow/Keras` · `Streamlit` · `Plotly` · `NLTK` · `Scikit-learn`

---

## 👤 Author

**Your Name** — Tekworks
