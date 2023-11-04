import streamlit as st
import requests
import cv2
import numpy as np
from io import BytesIO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

# Set your Azure subscription key and endpoint
subscription_key = "988d5cda57184a0c992cb9a02f5ddd0a"
azure_endpoint = "https://cv-api.cognitiveservices.azure.com/"

# Create a ComputerVisionClient
computervision_client = ComputerVisionClient(azure_endpoint, CognitiveServicesCredentials(subscription_key))

# Streamlit App
st.title("Object Detection with Microsoft Azure Computer Vision")

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Initialize variables
original_image = None
output_image = None

# Process the image and display the result
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    original_image = np.array(bytearray(file_bytes), dtype=np.uint8)
    original_image = cv2.imdecode(original_image, cv2.IMREAD_COLOR)

    # Convert the image to RGB (OpenCV uses BGR by default)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Convert bytes to stream
    image_stream = BytesIO(file_bytes)

    # Call the API for object detection
    objects = computervision_client.detect_objects_in_stream(image_stream)

    output_image = np.copy(original_image_rgb)

    # Draw rectangles around the detected objects
    for obj in objects.objects:
        rect = obj.rectangle
        cv2.rectangle(output_image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0, 255, 0), 2)

# Display images
if original_image is not None:
    st.subheader("Original Image")
    st.image(original_image_rgb, caption="Original Image")

if output_image is not None:
    output_image_path = "output_image.jpg"
    cv2.imwrite(output_image_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    st.subheader("Output Image with Detected Objects")
    st.image(output_image_path, caption="Output Image")