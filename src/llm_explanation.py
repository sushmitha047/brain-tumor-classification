import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def build_prompt(label, confidence):
    return f"""You are helping explain an image-classification model to a general audience.

The image is a brain MRI slice with a saliency map overlaid. A deep learning model classified
the scan into one of four categories: glioma, meningioma, pituitary tumor, or no tumor.
It predicted '{label}' with {confidence:.1%} confidence.

In the overlay, red and yellow areas are the pixels that influenced the prediction most;
cyan and blue areas influenced it less; uncoloured areas had little influence.

In at most 4 sentences:
- Describe which regions of the brain the model focused on.
- Suggest why those regions could be consistent with the predicted category.
Do not give medical advice, diagnosis or treatment recommendations."""


def generate_explanation(saliency_image, label, confidence):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to the .env file.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[build_prompt(label, confidence), saliency_image],
    )
    return response.text