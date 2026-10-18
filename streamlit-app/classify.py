import streamlit as st

from app_utils import analyze_scan, explain_scan, get_current_scan, probability_chart, result_card
from src.models import MODEL_CONFIGS

st.title("Classify an MRI scan")
st.write("Upload a brain MRI slice or pick a sample in the sidebar. The model predicts one of four classes and shows which pixels drove its decision.")

scan = get_current_scan()
if scan is None:
    st.info("Choose a scan in the sidebar to begin.")
    st.stop()

scan_name, scan_bytes = scan

model_name = st.radio("Model", list(MODEL_CONFIGS.keys()), horizontal=True)

try:
    with st.spinner("Analysing scan..."):
        label, confidence, probabilities, saliency = analyze_scan(model_name, scan_bytes)
except Exception as e:
    st.error(f"Could not analyse this image: {e}")
    st.stop()

left, right = st.columns(2)
left.image(scan_bytes, caption=f"Scan: {scan_name}", width="stretch")
right.image(saliency, caption="Saliency map: red and yellow = strongest influence", width="stretch")

left, right = st.columns([1, 2])
with left:
    result_card(label, confidence)
with right:
    st.plotly_chart(probability_chart(probabilities, label), width="stretch")

st.subheader("Why did the model say this?")
explanation_key = f"explanation::{model_name}::{scan_name}"

if st.button("Explain with Gemini"):
    try:
        with st.spinner("Asking Gemini..."):
            st.session_state[explanation_key] = explain_scan(saliency, label, confidence)
    except Exception as e:
        st.error(f"Could not generate an explanation: {e}")

if explanation_key in st.session_state:
    st.write(st.session_state[explanation_key])

st.caption("AI-generated description of the model's focus. Not medical advice.")