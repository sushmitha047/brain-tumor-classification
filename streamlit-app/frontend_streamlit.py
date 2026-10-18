import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SAMPLES_DIR = os.path.join(PROJECT_ROOT, "assets", "samples")

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")


def sidebar_scan_picker():
    st.sidebar.header("MRI scan")
    uploaded = st.sidebar.file_uploader("Upload a scan", type=["jpg", "jpeg", "png"])
    
    samples = sorted(
    f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
) if os.path.isdir(SAMPLES_DIR) else []
    sample = st.sidebar.selectbox(
        "...or try a sample",
        options=samples,
        index=None,
        placeholder="Choose a sample scan",
        disabled=uploaded is not None,
    )

    if uploaded is not None:
        st.session_state["scan_name"] = uploaded.name
        st.session_state["scan_bytes"] = uploaded.getvalue()
    elif sample is not None:
        with open(os.path.join(SAMPLES_DIR, sample), "rb") as f:
            st.session_state["scan_bytes"] = f.read()
        st.session_state["scan_name"] = sample
    else:
        st.session_state.pop("scan_bytes", None)
        st.session_state.pop("scan_name", None)

    st.sidebar.caption("Research demo. Not a medical device.")


pages = [
    st.Page("classify.py", title="Classify", icon="🔍", default=True),
    st.Page("compare_models.py", title="Compare models", icon="⚖️"),
    st.Page("how_it_works.py", title="How it works", icon="📘"),
]

page = st.navigation(pages)
sidebar_scan_picker()
page.run()