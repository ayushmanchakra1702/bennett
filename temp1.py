import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Load your model architecture
def load_model(model_path, device):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Make prediction
def predict(model, image_tensor, device):
    with torch.no_grad():
        output = torch.sigmoid(model(image_tensor.to(device)))
        prediction = (output > 0.5).long().item()
    return "Fake" if prediction == 1 else "Real"

# Main Streamlit app
def main():
    st.title("DeepFake Detection 🕵️‍♂️")
    st.markdown("Upload an image and the model will predict if it's **Fake** or **Real**.")

    model_path = "deepfake_detector.pth"  # Your saved model file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If model doesn't exist, notify user
    if not os.path.exists(model_path):
        st.error("Model file not found! Please train your model and save it as 'model.pth'.")
        return

    # Load model
    model = load_model(model_path, device)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess and predict
        image_tensor = preprocess_image(image)
        prediction = predict(model, image_tensor, device)

        st.markdown(f"### Prediction: `{prediction}`")

if __name__ == "__main__":
    main()
