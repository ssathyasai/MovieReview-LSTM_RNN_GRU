"""
app.py  –  Movie Review Sentiment Analysis  |  Streamlit App
-------------------------------------------------------------
Requires models to be trained first:
    python train_models.py

Then launch:
    streamlit run app.py
"""

import os
import json
import re
import string

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_WORDS  = 10000
MAXLEN     = 200
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
        font-size: 2.4rem;
        font-weight: 700;
        color: #E50914;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        color: #fff;
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #7b1d1d, #c0392b);
        color: #fff;
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ccc;
        border-bottom: 2px solid #E50914;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Cached helpers ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading word index …")
def get_word_index():
    return imdb.get_word_index()


@st.cache_resource(show_spinner="Loading model …")
def load_keras_model(name: str):
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


def models_exist() -> bool:
    return all(os.path.exists(p) for p in MODEL_FILES.values())


# ── Text → padded sequence ────────────────────────────────────────────────────
def preprocess(text: str, word_index: dict) -> np.ndarray:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    seq = [word_index[w] + 3 for w in words if w in word_index and word_index[w] + 3 < NUM_WORDS]
    return pad_sequences([seq], maxlen=MAXLEN)


# ── Predict with one model ────────────────────────────────────────────────────
def predict(model, padded: np.ndarray):
    prob = float(model.predict(padded, verbose=0)[0][0])
    sentiment = "Positive" if prob > 0.5 else "Negative"
    confidence = prob if prob > 0.5 else 1 - prob
    return sentiment, confidence, prob


# ── Confidence gauge chart ────────────────────────────────────────────────────
def gauge_chart(confidence: float, sentiment: str, title: str):
    color = "#2ecc71" if sentiment == "Positive" else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  50], "color": "#fadbd8"},
                {"range": [50, 75], "color": "#fdebd0"},
                {"range": [75, 100],"color": "#d5f5e3"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": confidence * 100,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#eee")
    return fig


# ── Pos/Neg probability bar ───────────────────────────────────────────────────
def prob_bar(pos_prob: float, model_name: str):
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
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/IMDB_Logo_2016.svg/320px-IMDB_Logo_2016.svg.png", width=120)
    st.markdown("## ⚙️ Settings")

    mode = st.radio(
        "Analysis Mode",
        ["Single Model", "Compare All Models"],
        index=0,
        help="Single Model: pick one architecture. Compare: run all three side-by-side.",
    )

    selected_model = "LSTM"
    if mode == "Single Model":
        selected_model = st.radio(
            "Select Model",
            ["SimpleRNN", "LSTM", "GRU"],
            index=1,
        )

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    all_metrics_loaded = all(load_metrics(n) is not None for n in ["SimpleRNN", "LSTM", "GRU"])
    if all_metrics_loaded:
        perf_data = {n: load_metrics(n) for n in ["SimpleRNN", "LSTM", "GRU"]}
        best_model = max(perf_data, key=lambda n: perf_data[n]["accuracy"])
        for name in ["SimpleRNN", "LSTM", "GRU"]:
            m = perf_data[name]
            badge = " 🏆" if name == best_model else ""
            st.markdown(f"**{name}{badge}**")
            st.caption(f"Acc: {m['accuracy']*100:.1f}%  |  F1: {m['f1']*100:.1f}%")
    else:
        st.warning("Run `python train_models.py` first to see metrics.")

    st.markdown("---")
    st.caption("Deep Learning · IMDB Dataset · TensorFlow/Keras")


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎬 Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning Based Sentiment Classification</div>', unsafe_allow_html=True)

# ── Guard: models not trained yet ─────────────────────────────────────────────
if not models_exist():
    st.error(
        "**Models not found.**  \n"
        "Please train the models first by running:  \n"
        "```\npython train_models.py\n```"
    )
    st.stop()

# ── Load word index ───────────────────────────────────────────────────────────
word_index = get_word_index()

# ═════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
tab_predict, tab_compare, tab_metrics = st.tabs(
    ["🔍 Predict", "⚖️ Compare Models", "📈 Model Metrics"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown('<div class="section-title">Enter Your Movie Review</div>', unsafe_allow_html=True)

    review_text = st.text_area(
        label="Review",
        placeholder="Enter your movie review here…",
        height=160,
        label_visibility="collapsed",
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze_clicked = st.button("🎯 Analyze Review", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=False):
            review_text = ""

    if analyze_clicked:
        if not review_text.strip():
            st.warning("Please enter a review before clicking Analyze.")
        else:
            padded = preprocess(review_text, word_index)

            if mode == "Single Model":
                model = load_keras_model(selected_model)
                if model is None:
                    st.error(f"Could not load {selected_model} model.")
                else:
                    sentiment, confidence, pos_prob = predict(model, padded)

                    st.markdown("---")
                    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

                    res_col, gauge_col = st.columns([1, 1])

                    with res_col:
                        css_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"
                        icon = "😊" if sentiment == "Positive" else "😞"
                        st.markdown(
                            f'<div class="{css_class}">'
                            f'{icon} Sentiment: {sentiment}<br>'
                            f'<span style="font-size:1rem;font-weight:400;">Confidence: {confidence*100:.1f}%</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"**Model used:** `{selected_model}`")
                        st.markdown(f"**Positive probability:** `{pos_prob*100:.2f}%`")
                        st.markdown(f"**Negative probability:** `{(1-pos_prob)*100:.2f}%`")

                    with gauge_col:
                        st.plotly_chart(
                            gauge_chart(confidence, sentiment, f"{selected_model} Confidence"),
                            use_container_width=True,
                        )

                    st.markdown('<div class="section-title">Probability Breakdown</div>', unsafe_allow_html=True)
                    st.plotly_chart(prob_bar(pos_prob, selected_model), use_container_width=True)

            else:  # Compare All Models
                st.markdown("---")
                st.markdown('<div class="section-title">All Models – Side-by-Side Results</div>', unsafe_allow_html=True)

                cols = st.columns(3)
                all_probs = {}
                for idx, name in enumerate(["SimpleRNN", "LSTM", "GRU"]):
                    model = load_keras_model(name)
                    if model is None:
                        cols[idx].error(f"{name} not loaded")
                        continue
                    sentiment, confidence, pos_prob = predict(model, padded)
                    all_probs[name] = pos_prob

                    with cols[idx]:
                        css_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"
                        icon = "😊" if sentiment == "Positive" else "😞"
                        st.markdown(
                            f'<div class="{css_class}">'
                            f'<b>{name}</b><br>{icon} {sentiment}<br>'
                            f'<span style="font-size:0.9rem;">{confidence*100:.1f}% confident</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(
                            gauge_chart(confidence, sentiment, name),
                            use_container_width=True,
                        )

                if all_probs:
                    st.markdown('<div class="section-title">Confidence Chart – All Models</div>', unsafe_allow_html=True)
                    fig = go.Figure()
                    for name, pos_prob in all_probs.items():
                        neg_prob = 1 - pos_prob
                        fig.add_trace(go.Bar(name="Positive", x=[name], y=[pos_prob * 100],
                                             marker_color="#2ecc71", showlegend=(name == "SimpleRNN")))
                        fig.add_trace(go.Bar(name="Negative", x=[name], y=[neg_prob * 100],
                                             marker_color="#e74c3c", showlegend=(name == "SimpleRNN")))
                    fig.update_layout(
                        barmode="group",
                        yaxis_title="Probability (%)",
                        yaxis=dict(range=[0, 100]),
                        height=350,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#eee",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – COMPARE MODELS (on a custom review)
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown('<div class="section-title">Compare All Three Models on a Review</div>', unsafe_allow_html=True)

    compare_text = st.text_area(
        "Review for comparison",
        placeholder="Enter your movie review here…",
        height=140,
        key="compare_input",
        label_visibility="collapsed",
    )

    if st.button("⚖️ Compare Models", type="primary"):
        if not compare_text.strip():
            st.warning("Please enter a review.")
        else:
            padded = preprocess(compare_text, word_index)
            rows = []
            all_probs = {}

            for name in ["SimpleRNN", "LSTM", "GRU"]:
                model = load_keras_model(name)
                if model is None:
                    continue
                sentiment, confidence, pos_prob = predict(model, padded)
                all_probs[name] = pos_prob
                rows.append({
                    "Model":       name,
                    "Sentiment":   sentiment,
                    "Confidence":  f"{confidence*100:.1f}%",
                    "Pos Prob":    f"{pos_prob*100:.2f}%",
                    "Neg Prob":    f"{(1-pos_prob)*100:.2f}%",
                })

            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Grouped bar
                st.markdown('<div class="section-title">Probability Comparison Chart</div>', unsafe_allow_html=True)
                fig = go.Figure()
                names = list(all_probs.keys())
                pos_vals = [all_probs[n] * 100 for n in names]
                neg_vals = [(1 - all_probs[n]) * 100 for n in names]

                fig.add_trace(go.Bar(name="Positive", x=names, y=pos_vals,
                                     marker_color="#2ecc71",
                                     text=[f"{v:.1f}%" for v in pos_vals],
                                     textposition="outside"))
                fig.add_trace(go.Bar(name="Negative", x=names, y=neg_vals,
                                     marker_color="#e74c3c",
                                     text=[f"{v:.1f}%" for v in neg_vals],
                                     textposition="outside"))
                fig.update_layout(
                    barmode="group",
                    yaxis=dict(range=[0, 115], title="Probability (%)"),
                    height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#eee",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Stacked horizontal bars
                st.markdown('<div class="section-title">Stacked Confidence Bars</div>', unsafe_allow_html=True)
                for name in names:
                    st.markdown(f"**{name}**")
                    st.plotly_chart(prob_bar(all_probs[name], name), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – MODEL METRICS
# ─────────────────────────────────────────────────────────────────────────────
with tab_metrics:
    st.markdown('<div class="section-title">Training & Evaluation Metrics</div>', unsafe_allow_html=True)

    all_metrics = {n: load_metrics(n) for n in ["SimpleRNN", "LSTM", "GRU"]}
    if any(v is None for v in all_metrics.values()):
        st.warning("Metrics not found. Run `python train_models.py` first.")
    else:
        import pandas as pd

        # Summary table
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

        # Best model highlight
        best = max(all_metrics, key=lambda n: all_metrics[n]["accuracy"])
        st.success(f"🏆 Best Model by Accuracy: **{best}** ({all_metrics[best]['accuracy']*100:.2f}%)")

        st.markdown("---")

        # Accuracy comparison bar
        st.markdown('<div class="section-title">Accuracy Comparison</div>', unsafe_allow_html=True)
        names = list(all_metrics.keys())
        acc_vals = [all_metrics[n]["accuracy"] * 100 for n in names]
        colors = ["#E50914" if n == best else "#555" for n in names]

        fig_acc = go.Figure(go.Bar(
            x=names, y=acc_vals,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in acc_vals],
            textposition="outside",
        ))
        fig_acc.update_layout(
            yaxis=dict(range=[0, 105], title="Accuracy (%)"),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eee",
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        # Training curves
        st.markdown('<div class="section-title">Training Curves</div>', unsafe_allow_html=True)
        curve_col1, curve_col2 = st.columns(2)

        with curve_col1:
            fig_tacc = go.Figure()
            for name, m in all_metrics.items():
                fig_tacc.add_trace(go.Scatter(
                    y=m["train_acc_history"], mode="lines+markers", name=name,
                ))
            fig_tacc.update_layout(
                title="Training Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy",
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eee",
            )
            st.plotly_chart(fig_tacc, use_container_width=True)

        with curve_col2:
            fig_vacc = go.Figure()
            for name, m in all_metrics.items():
                fig_vacc.add_trace(go.Scatter(
                    y=m["val_acc_history"], mode="lines+markers", name=name,
                ))
            fig_vacc.update_layout(
                title="Validation Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy",
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eee",
            )
            st.plotly_chart(fig_vacc, use_container_width=True)

        loss_col1, loss_col2 = st.columns(2)

        with loss_col1:
            fig_tloss = go.Figure()
            for name, m in all_metrics.items():
                fig_tloss.add_trace(go.Scatter(
                    y=m["train_loss_history"], mode="lines+markers", name=name,
                ))
            fig_tloss.update_layout(
                title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss",
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eee",
            )
            st.plotly_chart(fig_tloss, use_container_width=True)

        with loss_col2:
            fig_vloss = go.Figure()
            for name, m in all_metrics.items():
                fig_vloss.add_trace(go.Scatter(
                    y=m["val_loss_history"], mode="lines+markers", name=name,
                ))
            fig_vloss.update_layout(
                title="Validation Loss", xaxis_title="Epoch", yaxis_title="Loss",
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eee",
            )
            st.plotly_chart(fig_vloss, use_container_width=True)

        # Training time & params
        st.markdown('<div class="section-title">Training Time & Parameter Count</div>', unsafe_allow_html=True)
        time_col, param_col = st.columns(2)

        with time_col:
            fig_time = go.Figure(go.Bar(
                x=names,
                y=[all_metrics[n]["training_time"] for n in names],
                marker_color=["#E50914" if n == best else "#555" for n in names],
                text=[f"{all_metrics[n]['training_time']}s" for n in names],
                textposition="outside",
            ))
            fig_time.update_layout(
                title="Training Time (seconds)", yaxis_title="Seconds",
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eee",
            )
            st.plotly_chart(fig_time, use_container_width=True)

        with param_col:
            fig_params = go.Figure(go.Bar(
                x=names,
                y=[all_metrics[n]["num_params"] for n in names],
                marker_color=["#E50914" if n == best else "#555" for n in names],
                text=[f"{all_metrics[n]['num_params']:,}" for n in names],
                textposition="outside",
            ))
            fig_params.update_layout(
                title="Number of Parameters", yaxis_title="Parameters",
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eee",
            )
            st.plotly_chart(fig_params, use_container_width=True)
