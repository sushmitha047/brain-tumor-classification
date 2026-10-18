import os

import streamlit as st

from app_utils import analyze_scan, get_current_scan, probability_chart, result_card
from src.models import MODEL_CONFIGS, PROJECT_ROOT

ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

st.title("Compare models")
st.write("A transfer-learning model (Xception, pre-trained on ImageNet) against a CNN trained from scratch, on the same scan.")

scan = get_current_scan()
if scan is None:
    st.info("Choose a scan in the sidebar to compare the models on it.")
else:
    scan_name, scan_bytes = scan
    st.image(scan_bytes, caption=f"Scan: {scan_name}", width=260)

    columns = st.columns(len(MODEL_CONFIGS))
    for column, model_name in zip(columns, MODEL_CONFIGS):
        with column:
            st.subheader(model_name)
            label, confidence, probabilities, _ = analyze_scan(model_name, scan_bytes)
            result_card(label, confidence)
            st.plotly_chart(probability_chart(probabilities, label), width="stretch")

st.divider()

st.subheader("Test-set performance")
st.table({
    "Model": ["Xception (transfer learning)", "Custom CNN"],
    "Test accuracy": ["94.13%", "86.12%"],
    "Input size": ["299 × 299", "224 × 224"],
})
st.caption("800 held-out test images (200 per class) that the models never saw during training or tuning.")

matrix_files = {
    "Xception (transfer learning)": "confusion_matrix_xception.png",
    "Custom CNN": "confusion_matrix_cnn.png",
}
columns = st.columns(2)
for column, (model_name, file_name) in zip(columns, matrix_files.items()):
    path = os.path.join(ASSETS_DIR, file_name)
    if os.path.exists(path):
        column.image(path, caption=f"Confusion matrix: {model_name}", width="stretch")