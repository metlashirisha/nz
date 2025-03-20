import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import traceback

# Page configuration
st.set_page_config(
    page_title="Autism Facial Recognition",
    page_icon="🧠",
    layout="centered"
)

# App title and description
st.title("🧠 Autism Facial Recognition Model")
st.write("Upload an image to classify.")

try:
    # Load the trained model
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("model.h5")
    
    model = load_model()
    
    # Get model's expected input shape
    if hasattr(model, 'input_shape'):
        if isinstance(model.input_shape, tuple):
            input_shape = model.input_shape
        elif isinstance(model.input_shape, list):
            input_shape = model.input_shape[0]
        else:
            input_shape = None
            
        st.write(f"Model input shape: {input_shape}")
        
        if input_shape is not None and len(input_shape) >= 3:
            # Extract dimensions, handling both (None, H, W, C) and (H, W, C) formats
            if len(input_shape) == 4:
                _, img_height, img_width, img_channels = input_shape
            else:
                img_height, img_width, img_channels = input_shape
                
            st.write(f"Expected image dimensions: {img_height}x{img_width}x{img_channels}")
        else:
            st.error("Could not determine model's expected input dimensions.")
            img_height, img_width, img_channels = 224, 224, 3  # Fallback to common values
    else:
        st.warning("Model input shape not available. Using default 224x224x3")
        img_height, img_width, img_channels = 224, 224, 3
    
    # Define image preprocessing function
    def preprocess_image(image):
        # Convert to RGB to ensure 3 channels
        image = image.convert("RGB")
        
        # Resize to match model's expected input size
        image = image.resize((img_width, img_height))
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_array, axis=0)
        
        return image_batch
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process image and make prediction
        with st.spinner("Processing image..."):
            try:
                # Preprocess image
                processed_image = preprocess_image(image)
                st.write(f"Processed image shape: {processed_image.shape}")
                
                # Make prediction
                prediction = model.predict(processed_image)
                
                # Display results
                st.success("Prediction complete!")
                
                if prediction.size == 1:
                    # For binary classification models
                    confidence = float(prediction[0][0])
                    st.write(f"Confidence score: {confidence:.4f}")
                    
                    threshold = 0.5
                    predicted_class = "Autism" if confidence >= threshold else "Non-Autism"
                    st.write(f"Predicted class: {predicted_class}")
                    
                    # Visual indicator
                    st.progress(confidence)
                else:
                    # For multi-class or other output formats
                    st.write("Prediction result:")
                    st.write(prediction)
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.write("Debug information:")
                st.code(traceback.format_exc())

except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.write("Debug information:")
    st.code(traceback.format_exc())