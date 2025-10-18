import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Dog Breed Identifier 🐶",
    page_icon="🐾",
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_full_model(model_path):
    """
    Loads the full Keras model from the .h5 file.
    This is the most reliable method as it loads the exact saved architecture,
    optimizer, and weights without manual reconstruction.
    """
    try:
        # Provide the custom_objects dictionary to tell Keras how to handle the KerasLayer
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
        return model
        
    except Exception as e:
        st.error(f"Error loading model '{model_path}': {e}")
        st.error("This is often due to a version mismatch between the TensorFlow/Keras used to save the model and the one loading it. Ensure you are using the correct environment.")
        return None

@st.cache_data
def load_labels(labels_path):
    """Loads the dog breed labels from a CSV file."""
    try:
        labels_df = pd.read_csv(labels_path)
        return labels_df["breed"].tolist()
    except FileNotFoundError:
        st.error(f"Error: '{labels_path}' not found. Please ensure the file is in the correct directory.")
        return None

# --- Main App ---
# Define file paths
MODEL_PATH = 'extendedresults.h5'
LABELS_PATH = 'labels.csv'

# Load resources
model = load_full_model(MODEL_PATH)
class_names = load_labels(LABELS_PATH)

# UI Layout
st.title("🐾 Dog Vision: Breed Identifier")
st.write(f"Upload an image of a dog to predict its breed using the `{MODEL_PATH}` model!")

uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None and class_names is not None:
    image = Image.open(uploaded_file)
    
    # --- THIS IS THE ONLY CHANGE ---
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_class_index = np.argmax(prediction)
    predicted_breed = class_names[predicted_class_index].replace("_", " ").title()

    st.success(f"**Prediction:** {predicted_breed}")
    st.info(f"**Confidence:** {confidence:.2%}")

    with st.expander("Show Top 5 Predictions"):
        top_5_indices = prediction[0].argsort()[-5:][::-1]
        top_5_breeds = [class_names[i].replace("_", " ").title() for i in top_5_indices]
        top_5_confidences = prediction[0][top_5_indices]
        df = pd.DataFrame({"Breed": top_5_breeds, "Confidence": top_5_confidences})
        st.dataframe(df.style.format({"Confidence": "{:.2%}"}), use_container_width=True)
