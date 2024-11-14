import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import os
from PIL import Image
import numpy as np




@st.cache_resource
def load_model():
    return tf.keras.models.load_model('your_model.h5')


# Set up ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Streamlit App Title
st.title("Image Augmentation Demo")
st.subheader("Upload Your Image")

# File uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image using PIL
    img = Image.open(uploaded_file)
    
    # Display the uploaded image in the Streamlit app
    st.subheader("Original Image")
    st.image(img, caption="Original Image", use_column_width=True)
    
    # Convert the image to a numpy array and reshape it for the augmentation
    x = np.array(img)
    if x.shape[2] == 4:  # If the image has an alpha channel (RGBA), convert to RGB
        x = x[:, :, :3]
    x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels)

    # Set output directory for augmented images
    output_dir = r"C:\Users\Ravi\OneDrive\Pictures\Images\New folder (2)"
    os.makedirs(output_dir, exist_ok=True)

    # Generate augmented images and display one of them
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='augmented', save_format='jpeg'):
        i += 1
        if i > 30:  # Display first 5 augmented images
            break  # Exit the loop after generating the required number of images

    # List all images in the output directory
    st.subheader("Augmented Images")
    
    # List all files in the output directory
    augmented_files = [f for f in os.listdir(output_dir) if f.startswith('augmented') and f.endswith('.jpeg')]
    
    # Display all augmented images
    for file in augmented_files:
        augmented_img_path = os.path.join(output_dir, file)
        augmented_img = Image.open(augmented_img_path)
        st.image(augmented_img, caption=f"Augmented Image: {file}", use_column_width=True)

