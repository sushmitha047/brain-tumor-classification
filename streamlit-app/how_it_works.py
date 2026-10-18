import streamlit as st

st.title("How it works")

st.header("Data")
st.markdown("""
- **Source:** the public [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) on Kaggle.
- **Size:** 7,200 2D MRI slices in four classes: glioma, meningioma, pituitary tumor and no tumor.
- **Split:** 5,600 images for training; the Kaggle test folder (1,600 images) split in half, by class, into validation (800) and test (800).
""")

st.header("Models")
st.markdown("""
- **Xception (transfer learning):** a network pre-trained on ImageNet, with a new classification head
  (dropout, a 128-unit dense layer, a 4-way softmax). It reuses visual features learned from millions of photos.
- **Custom CNN:** convolutional layers trained from scratch on this dataset only.
- The comparison shows why transfer learning wins on a small dataset: 94.13% vs 86.12% test accuracy on the same 800 test images.
""")

st.header("Saliency maps")
st.markdown("""
For the predicted class, the app computes the **gradient** of the class score with respect to every pixel:
how much the score would change if that pixel changed slightly. Pixels with the largest gradients influenced
the prediction most. Only the brain area is considered, and the top 20% of pixels are kept, smoothed, coloured
from blue (weak) to red (strongest) and laid over the original scan.
""")

st.header("Explanations")
st.markdown("""
On request, the saliency map and the prediction are sent to Google's Gemini model, which describes in plain language
which brain regions the model focused on.
""")

st.header("Limitations")
st.markdown("""
- Trained and tested on one public dataset; scans from other hospitals or scanners may perform worse.
- Works on single 2D slices, while real diagnosis uses full 3D scans and clinical context.
- Saliency maps show where the model looked, not why a tumor is present.
- **This is a research demo, not a medical device, and nothing here is medical advice.**
""")