import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.llm_explanation import generate_explanation
from src.models import CLASS_LABELS, MODEL_CONFIGS, load_model_by_name, predict, preprocess_image
from src.saliency_map_generation import generate_saliency_map

NO_TUMOR_COLOR = "#2a9d8f"
TUMOR_COLOR = "#e76f51"
NEUTRAL_COLOR = "#8d99ae"


@st.cache_resource(show_spinner="Loading model (first time only)...")
def get_model(model_name):
    return load_model_by_name(model_name)


@st.cache_data(show_spinner=False)
def analyze_scan(model_name, image_bytes):
    model = get_model(model_name)
    img_size = MODEL_CONFIGS[model_name]["img_size"]
    _, img_array = preprocess_image(image_bytes, img_size)
    label, confidence, probabilities = predict(model, img_array)
    class_index = CLASS_LABELS.index(label)
    # saliency = generate_saliency_map(model, img_array, class_index)
    saliency = generate_saliency_map(model, img_array, class_index, img_size, image_bytes)
    return label, confidence, probabilities, saliency


@st.cache_data(show_spinner=False)
def explain_scan(saliency, label, confidence):
    return generate_explanation(Image.fromarray(saliency), label, confidence)


def get_current_scan():
    if "scan_bytes" not in st.session_state:
        return None
    return st.session_state["scan_name"], st.session_state["scan_bytes"]


def result_color(label):
    return NO_TUMOR_COLOR if label == "No tumor" else TUMOR_COLOR


def result_card(label, confidence):
    color = result_color(label)
    st.markdown(
        f"""
        <div style="border: 2px solid {color}; border-radius: 12px; padding: 16px; text-align: center;">
            <div style="font-size: 14px; opacity: 0.75;">Prediction</div>
            <div style="font-size: 32px; font-weight: 700; color: {color};">{label}</div>
            <div style="font-size: 16px;">Confidence: {confidence:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_chart(probabilities, predicted_label):
    colors = [
        result_color(lbl) if lbl == predicted_label else NEUTRAL_COLOR
        for lbl in CLASS_LABELS
    ]

    fig = go.Figure(go.Bar(
        x=probabilities,
        y=CLASS_LABELS,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probabilities],
        textposition="outside",
    ))

    fig.update_layout(
        xaxis=dict(range=[0, 1.15], tickformat=".0%", title="Probability"),
        yaxis=dict(autorange="reversed"),
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    return fig