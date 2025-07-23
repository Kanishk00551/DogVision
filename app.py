import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image

# --- App Configuration ---
# Set the page title, icon, and layout for a better user experience.
st.set_page_config(
    page_title="Dog Vision 🐶",
    page_icon="🐾",
    layout="centered"
)

# --- UI Elements ---
st.title("🐶 Dog Vision: Breed Classifier")
st.write(
    "Welcome! Upload an image of a dog, and this app will use a deep learning model "
    "to predict its breed."
)

# --- Caching Functions ---
# Using decorators to cache the model and labels prevents reloading them on every
# user interaction, which significantly speeds up the app after the first run.

@st.cache_resource
def load_model(model_path):
    """
    Loads the pre-trained Keras model from the specified path.
    The @st.cache_resource decorator ensures the model is loaded only once.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_labels(labels_path):
    """
    Loads the breed labels from a CSV file into a list.
    The @st.cache_data decorator caches the labels data.
    """
    try:
        df = pd.read_csv(labels_path)
        return df['breed'].tolist()
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return []

# --- Load Model and Labels ---
# Define the paths to your model and labels file.
MODEL_PATH = "dog_vision_model.h5"
LABELS_PATH = "unique_breeds.csv"

model = load_model(MODEL_PATH)
labels = load_labels(LABELS_PATH)

# --- Image Preprocessing Function ---
def preprocess_image(image):
    """
    Takes a PIL image, resizes it to the model's expected input size (224x224),
    converts it to a NumPy array, and adds a batch dimension.
    """
    img = image.resize((224, 224))
    img_array = np.array(img)
    # The model expects a batch of images, so we add a dimension.
    # Shape becomes: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Main Application Logic ---
# Create the file uploader widget.
uploaded_file = st.file_uploader(
    "Choose an image of a dog...",
    type=["jpg", "jpeg", "png"]
)

# This block runs only if a file has been uploaded and the assets are loaded.
if uploaded_file is not None and model is not None and labels:
    # Display the image the user uploaded.
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Uploaded Doggo!", use_column_width=True)
    
    # Show a spinner while the model is making a prediction.
    with st.spinner("Classifying breed..."):
        preprocessed_image = preprocess_image(image)
        
        # Get the model's predictions (an array of probabilities).
        predictions = model.predict(preprocessed_image)
        
        # Find the top prediction's index and confidence score.
        top_pred_index = np.argmax(predictions)
        top_pred_confidence = np.max(predictions)
        
        # Get the corresponding breed name from the labels list.
        # We replace underscores with spaces and capitalize for better readability.
        predicted_breed = labels[top_pred_index].replace("_", " ").title()

    # Display the final prediction.
    st.success("Prediction Complete!")
    st.metric(
        label="Predicted Breed",
        value=predicted_breed,
        delta=f"Confidence: {top_pred_confidence:.2%}" # Formats confidence as a percentage.
    )
    
    # Add an expander to show more details about the top predictions.
    with st.expander("See Top 5 Predictions"):
        # Get indices of the top 5 predictions in descending order.
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_confidences = predictions[0][top_5_indices]
        top_5_breeds = [labels[i].replace("_", " ").title() for i in top_5_indices]

        # Create a clean DataFrame for display.
        df = pd.DataFrame({
            'Breed': top_5_breeds,
            'Confidence': top_5_confidences
        })
        st.dataframe(df, use_container_width=True)

# Show an error message if the model or labels failed to load.
elif model is None or not labels:
    st.error(
        "Model or labels could not be loaded. Please ensure the necessary files "
        "(`dog_vision_model.h5` and `unique_breeds.csv`) are in the same directory as `app.py`."
    )
    