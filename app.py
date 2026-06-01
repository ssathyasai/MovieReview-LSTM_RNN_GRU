"""
app.py  –  Movie Review Sentiment Analysis  |  Streamlit App
-------------------------------------------------------------
Models are trained automatically on first run if not found.
Launch:
    streamlit run app.py
"""

import os
import json
import re
import string
import time

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_WORDS  = 10000
MAXLEN     = 200
EMBED_DIM  = 32
UNITS      = 32
EPOCHS     = 5
BATCH_SIZE = 64
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_FILES = {
    "SimpleRNN": os.path.join(MODELS_DIR, "simplernn_model.keras"),
    "LSTM":      os.path.join(MODELS_DIR, "lstm_model.keras"),
    "GRU":       os.path.join(MODELS_DIR, "gru_model.keras"),
}
METRICS_FILES = {
    "SimpleRNN": os.path.join(MODELS_DIR, "simplernn_metrics.json"),
    "LSTM":      os.path.join(MODELS_DIR, "lstm_metrics.json"),
    "GRU":       os.path.join(MODELS_DIR, "gru_metrics.json"),
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 700;
        color: #E50914; text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem; color: #888;
        text-align: center; margin-bottom: 2rem;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        color: #fff; padding: 1.2rem 1.8rem; border-radius: 12px;
        font-size: 1.4rem; font-weight: 700; text-align: center;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #7b1d1d, #c0392b);
        color: #fff; padding: 1.2rem 1.8rem; border-radius: 12px;
        font-size: 1.4rem; font-weight: 700; text-align: center;
    }
    .section-title {
        font-size: 1.2rem; font-weight: 600; color: #ccc;
        border-bottom: 2px solid #E50914;
        padding-bottom: 0.3rem; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Auto-train on first run ───────────────────────────────────────────────────
def models_exist() -> bool:
    return all(os.path.exists(p) for p in MODEL_FILES.values())


def train_all_models():
    """Train SimpleRNN, LSTM, GRU and save to models/ directory."""
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    os.makedirs(MODELS_DIR, exist_ok=True)

    st.info("⏳ First run detected — training models on IMDB dataset. This takes ~5–10 minutes.")
    progress = st.progress(0, text="Loading dataset…")

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
    X_train = pad_sequences(X_train, maxlen=MAXLEN)
    X_test  = pad_sequences(X_test,  maxlen=MAXLEN)

    def build(layer):
        m = Sequential([
            Embedding(NUM_WORDS, EMBED_DIM, input_length=MAXLEN),
            layer,
            Dense(16, activation="relu"),
            Dense(1,  activation="sigmoid"),
        ])
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return m

    configs = [
        ("SimpleRNN", lambda: SimpleRNN(UNITS)),
        ("LSTM",      lambda: LSTM(UNITS)),
        ("GRU",       lambda: GRU(UNITS)),
    ]

    for i, (name, layer_fn) in enumerate(configs):
        progress.progress((i * 33), text=f"Training {name} ({i+1}/3)…")
        model = build(layer_fn())
        t0 = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_split=0.2, verbose=0,
        )
        elapsed = round(time.time() - t0, 2)

        y_prob = model.predict(X_test, verbose=0)
        y_pred = (y_prob > 0.5).astype(int)

        metrics = {
            "accuracy":           float(accuracy_score(y_test, y_pred)),
            "precision":          float(precision_score(y_test, y_pred)),
            "recall":             float(recall_score(y_test, y_pred)),
            "f1":                 float(f1_score(y_test, y_pred)),
            "val_accuracy":       float(history.history["val_accuracy"][-1]),
            "val_loss":           float(history.history["val_loss"][-1]),
            "training_time":      elapsed,
            "num_params":         int(model.count_params()),
            "train_acc_history":  [float(v) for v in history.history["accuracy"]],
            "val_acc_history":    [float(v) for v in history.history["val_accuracy"]],
            "train_loss_history": [float(v) for v in history.history["loss"]],
            "val_loss_history":   [float(v) for v in history.history["val_loss"]],
        }

        model.save(MODEL_FILES[name])
        with open(METRICS_FILES[name], "w") as f:
            json.dump(metrics, f, indent=2)

    progress.progress(100, text="✅ All models trained and saved!")
    st.success("Models trained successfully! Reloading…")
    st.rerun()


if not models_exist():
    train_all_models()
    st.stop()


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading word index…")
def get_word_index():
    from tensorflow.keras.datasets import imdb
    return imdb.get_word_index()


@st.cache_resource(show_spinner="Loading model…")
def load_keras_model(name: str):
    from tensorflow.keras.models import load_model
    path = MODEL_FILES[name]
    if not os.path.exists(path):
        return None
    return load_model(path)


@st.cache_data(show_spinner=False)
def load_metrics(name: str):
    path = METRICS_FILES[name]
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text: str, word_index: dict) -> np.ndarray:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    seq = [word_index[w] + 3 for w in words
           if w in word_index and word_index[w] + 3 < NUM_WORDS]
    return pad_sequences([seq], maxlen=MAXLEN)


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(model, padded):
    prob = float(model.predict(padded, verbose=0)[0][0])
    sentiment  = "Positive" if prob > 0.5 else "Negative"
    confidence = prob if prob > 0.5 else 1 - prob
    return sentiment, confidence, prob


# ── Charts ────────────────────────────────────────────────────────────────────
def gauge_chart(confidence, sentiment, title):
    color = "#2ecc71" if sentiment == "Positive" else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  50], "color": "#fadbd8"},
                {"range": [50, 75], "color": "#fdebd0"},
                {"range": [75, 100],"color": "#d5f5e3"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#eee")
    return fig


def prob_bar(pos_prob, model_name):
    neg_prob = 1 - pos_prob
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Positive", x=[round(pos_prob * 100, 1)], y=[model_name],
        orientation="h", marker_color="#2ecc71",
        text=[f"{pos_prob*100:.1f}%"], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Negative", x=[round(neg_prob * 100, 1)], y=[model_name],
        orientation="h", marker_color="#e74c3c",
        text=[f"{neg_prob*100:.1f}%"], textposition="inside",
    ))
    fig.update_layout(
        barmode="stack", height=90,
        margin=dict(t=5, b=5, l=0, r=0),
        xaxis=dict(range=[0, 100], showticklabels=False),
        yaxis=dict(showticklabels=False),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/IMDB_Logo_2016.svg/320px-IMDB_Logo_2016.svg.png",
        width=120,
    )
    st.markdown("## ⚙️ Settings")

    mode = st.radio(
        "Analysis Mode",
        ["Single Model", "Compare All Models"],
        index=0,
    )

    selected_model = "LSTM"
    if mode == "Single Model":
        selected_model = st.radio("Select Model", ["SimpleRNN", "LSTM", "GRU"], index=1)

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    perf_data = {n: load_metrics(n) for n in ["SimpleRNN", "LSTM", "GRU"]}
    if all(v is not None for v in perf_data.values()):
        best_model = max(perf_data, key=lambda n: perf_data[n]["accuracy"])
        for name in ["SimpleRNN", "LSTM", "GRU"]:
            m = perf_data[name]
            badge = " 🏆" if name == best_model else ""
            st.markdown(f"**{name}{badge}**")
            st.caption(f"Acc: {m['accuracy']*100:.1f}%  |  F1: {m['f1']*100:.1f}%")

    st.markdown("---")
    st.caption("Deep Learning · IMDB Dataset · TensorFlow/Keras")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎬 Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning Based Sentiment Classification</div>', unsafe_allow_html=True)

word_index = get_word_index()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_compare, tab_metrics = st.tabs(
    ["🔍 Predict", "⚖️ Compare Models", "📈 Model Metrics"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown('<div class="section-title">Enter Your Movie Review</div>', unsafe_allow_html=True)

    review_text = st.text_area(
        label="Review", placeholder="Enter your movie review here…",
        height=160, label_visibility="collapsed",
    )

    analyze_clicked = st.button("🎯 Analyze Review", type="primary")

    if analyze_clicked:
        if not review_text.strip():
            st.warning("Please enter a review before clicking Analyze.")
        else:
            padded = preprocess(review_text, word_index)

            if mode == "Single Model":
                model = load_keras_model(selected_model)
                sentiment, confidence, pos_prob = predict(model, padded)

                st.markdown("---")
                st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
                res_col, gauge_col = st.columns(2)

                with res_col:
                    css = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"
                    icon = "😊" if sentiment == "Positive" else "😞"
                    st.markdown(
                        f'<div class="{css}">{icon} Sentiment: {sentiment}<br>'
                        f'<span style="font-size:1rem;font-weight:400;">Confidence: {confidence*100:.1f}%</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"**Model:** `{selected_model}`")
                    st.markdown(f"**Positive probability:** `{pos_prob*100:.2f}%`")
                    st.markdown(f"**Negative probability:** `{(1-pos_prob)*100:.2f}%`")

                with gauge_col:
                    st.plotly_chart(gauge_chart(confidence, sentiment, f"{selected_model} Confidence"),
                                    use_container_width=True)

                st.markdown('<div class="section-title">Probability Breakdown</div>', unsafe_allow_html=True)
                st.plotly_chart(prob_bar(pos_prob, selected_model), use_container_width=True)

            else:
                st.markdown("---")
                st.markdown('<div class="section-title">All Models – Side-by-Side</div>', unsafe_allow_html=True)
                cols = st.columns(3)
                all_probs = {}

                for idx, name in enumerate(["SimpleRNN", "LSTM", "GRU"]):
                    model = load_keras_model(name)
                    sentiment, confidence, pos_prob = predict(model, padded)
                    all_probs[name] = pos_prob
                    with cols[idx]:
                        css = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"
                        icon = "😊" if sentiment == "Positive" else "😞"
                        st.markdown(
                            f'<div class="{css}"><b>{name}</b><br>{icon} {sentiment}<br>'
                            f'<span style="font-size:0.9rem;">{confidence*100:.1f}% confident</span></div>',
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(gauge_chart(confidence, sentiment, name),
                                        use_container_width=True)

                st.markdown('<div class="section-title">Confidence Chart</div>', unsafe_allow_html=True)
                fig = go.Figure()
                for name, pp in all_probs.items():
                    fig.add_trace(go.Bar(name="Positive", x=[name], y=[pp * 100],
                                         marker_color="#2ecc71", showlegend=(name == "SimpleRNN")))
                    fig.add_trace(go.Bar(name="Negative", x=[name], y=[(1-pp) * 100],
                                         marker_color="#e74c3c", showlegend=(name == "SimpleRNN")))
                fig.update_layout(
                    barmode="group", yaxis=dict(range=[0, 100], title="Probability (%)"),
                    height=350, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font_color="#eee",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – COMPARE
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown('<div class="section-title">Compare All Three Models on a Review</div>', unsafe_allow_html=True)

    compare_text = st.text_area(
        "Review", placeholder="Enter your movie review here…",
        height=140, key="compare_input", label_visibility="collapsed",
    )

    if st.button("⚖️ Compare Models", type="primary"):
        if not compare_text.strip():
            st.warning("Please enter a review.")
        else:
            import pandas as pd
            padded = preprocess(compare_text, word_index)
            rows, all_probs = [], {}

            for name in ["SimpleRNN", "LSTM", "GRU"]:
                model = load_keras_model(name)
                sentiment, confidence, pos_prob = predict(model, padded)
                all_probs[name] = pos_prob
                rows.append({
                    "Model": name, "Sentiment": sentiment,
                    "Confidence": f"{confidence*100:.1f}%",
                    "Pos Prob": f"{pos_prob*100:.2f}%",
                    "Neg Prob": f"{(1-pos_prob)*100:.2f}%",
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">Probability Comparison</div>', unsafe_allow_html=True)
            names    = list(all_probs.keys())
            pos_vals = [all_probs[n] * 100 for n in names]
            neg_vals = [(1 - all_probs[n]) * 100 for n in names]

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Positive", x=names, y=pos_vals,
                                  marker_color="#2ecc71",
                                  text=[f"{v:.1f}%" for v in pos_vals], textposition="outside"))
            fig.add_trace(go.Bar(name="Negative", x=names, y=neg_vals,
                                  marker_color="#e74c3c",
                                  text=[f"{v:.1f}%" for v in neg_vals], textposition="outside"))
            fig.update_layout(
                barmode="group", yaxis=dict(range=[0, 115], title="Probability (%)"),
                height=380, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font_color="#eee",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-title">Stacked Confidence Bars</div>', unsafe_allow_html=True)
            for name in names:
                st.markdown(f"**{name}**")
                st.plotly_chart(prob_bar(all_probs[name], name), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – METRICS
# ─────────────────────────────────────────────────────────────────────────────
with tab_metrics:
    import pandas as pd
    st.markdown('<div class="section-title">Training & Evaluation Metrics</div>', unsafe_allow_html=True)

    all_metrics = {n: load_metrics(n) for n in ["SimpleRNN", "LSTM", "GRU"]}

    summary_rows = []
    for name, m in all_metrics.items():
        summary_rows.append({
            "Model":        name,
            "Accuracy":     f"{m['accuracy']*100:.2f}%",
            "Precision":    f"{m['precision']*100:.2f}%",
            "Recall":       f"{m['recall']*100:.2f}%",
            "F1 Score":     f"{m['f1']*100:.2f}%",
            "Val Accuracy": f"{m['val_accuracy']*100:.2f}%",
            "Val Loss":     f"{m['val_loss']:.4f}",
            "Train Time":   f"{m['training_time']}s",
            "Parameters":   f"{m['num_params']:,}",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    best = max(all_metrics, key=lambda n: all_metrics[n]["accuracy"])
    st.success(f"🏆 Best Model: **{best}** ({all_metrics[best]['accuracy']*100:.2f}%)")

    st.markdown("---")

    # Accuracy bar
    st.markdown('<div class="section-title">Accuracy Comparison</div>', unsafe_allow_html=True)
    names  = list(all_metrics.keys())
    colors = ["#E50914" if n == best else "#555" for n in names]
    fig_acc = go.Figure(go.Bar(
        x=names, y=[all_metrics[n]["accuracy"] * 100 for n in names],
        marker_color=colors,
        text=[f"{all_metrics[n]['accuracy']*100:.2f}%" for n in names],
        textposition="outside",
    ))
    fig_acc.update_layout(yaxis=dict(range=[0, 105], title="Accuracy (%)"), height=320,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#eee")
    st.plotly_chart(fig_acc, use_container_width=True)

    # Training curves
    st.markdown('<div class="section-title">Training Curves</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    def line_chart(col, title, key):
        fig = go.Figure()
        for name, m in all_metrics.items():
            fig.add_trace(go.Scatter(y=m[key], mode="lines+markers", name=name))
        fig.update_layout(title=title, xaxis_title="Epoch", height=300,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#eee")
        col.plotly_chart(fig, use_container_width=True)

    line_chart(c1, "Training Accuracy",   "train_acc_history")
    line_chart(c2, "Validation Accuracy", "val_acc_history")
    c3, c4 = st.columns(2)
    line_chart(c3, "Training Loss",   "train_loss_history")
    line_chart(c4, "Validation Loss", "val_loss_history")

    # Time & params
    st.markdown('<div class="section-title">Training Time & Parameters</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)

    fig_time = go.Figure(go.Bar(
        x=names, y=[all_metrics[n]["training_time"] for n in names],
        marker_color=colors,
        text=[f"{all_metrics[n]['training_time']}s" for n in names], textposition="outside",
    ))
    fig_time.update_layout(title="Training Time (s)", height=300,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#eee")
    t1.plotly_chart(fig_time, use_container_width=True)

    fig_par = go.Figure(go.Bar(
        x=names, y=[all_metrics[n]["num_params"] for n in names],
        marker_color=colors,
        text=[f"{all_metrics[n]['num_params']:,}" for n in names], textposition="outside",
    ))
    fig_par.update_layout(title="Number of Parameters", height=300,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#eee")
    t2.plotly_chart(fig_par, use_container_width=True)
