from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import io


# Generate a saliency map
def generate_saliency_map(model, img_array, class_index, img_size, image_bytes):
    # Load the original image to get its size
    original_image = Image.open(io.BytesIO(image_bytes))
    original_img_size = original_image.size  # (width, height)

    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]

    # Compute gradients
    gradients = tape.gradient(target_class, img_tensor)
    gradients = tf.math.abs(gradients)
    gradients = tf.reduce_max(gradients, axis=-1)
    gradients = gradients.numpy().squeeze()

    # Resize for display
    gradients = cv2.resize(gradients, original_img_size)

    # Mask for the brain region
    center_y, center_x = gradients.shape[0] // 2, gradients.shape[1] // 2
    radius = min(center_y, center_x) - 10
    y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    gradients = gradients * mask

    # Normalize inside mask
    brain_gradients = gradients[mask]
    if brain_gradients.max() > brain_gradients.min():
        brain_gradients = (brain_gradients - brain_gradients.min()) / (brain_gradients.max() - brain_gradients.min())
    gradients[mask] = brain_gradients

    # Threshold and blur
    threshold = np.percentile(gradients[mask], 80)
    gradients[gradients < threshold] = 0
    gradients = cv2.GaussianBlur(gradients, (11, 11), 0)
    if gradients.max() > 0:
        gradients = gradients / gradients.max()


    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose on original
    heatmap = cv2.resize(heatmap, img_size)
    original_img = img_array[0]
    original_img = (original_img * 255).astype(np.uint8)
    superimposed_img = heatmap * 0.7 + original_img * 0.3
    superimposed_img = superimposed_img.astype(np.uint8)

    return superimposed_img