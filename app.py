import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --------------------
# CONFIG
# --------------------
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")

device = "cuda" if torch.cuda.is_available() else "cpu"
class_names = ["Bengin cases", "Malignant cases", "Normal cases"]

# --------------------
# LOAD MODEL
# --------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load("lung_cancer_model_40.pth", map_location=device))
    model.eval()
    return model.to(device)

model = load_model()

# --------------------
# TRANSFORM
# --------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------------------
# PREDICT FUNCTION
# --------------------
def predict_image(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)
        prob = torch.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)
    return class_names[pred.item()], conf.item()

# --------------------
# GRADCAM FUNCTION
# --------------------
def gradcam_image(img):
    img_np = np.array(img.convert("RGB"))
    img_resized = cv2.resize(img_np, (224,224))
    input_tensor = transform(Image.fromarray(img_resized)).unsqueeze(0).to(device)

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    cam_img = show_cam_on_image(img_resized/255.0, grayscale_cam, use_rgb=True)
    
    # Create figure with colorbar
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [4, 0.2]})
    ax[0].imshow(cam_img)
    ax[0].axis('off')
    ax[0].set_title('Affected Area Heatmap')
    
    # Add colorbar
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax[1])
    cb.set_label('Importance Level', rotation=270, labelpad=20)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(['0.0\nLow', '0.25', '0.5\nModerate', '0.75', '1.0\nHigh'])
    
    plt.tight_layout()
    return fig

# --------------------
# STREAMLIT UI
# --------------------
st.title(" Lung Cancer Detection from CT Scan")
st.write("Upload a CT scan image to detect **Normal / Benign / Malignant** cases")

uploaded_file = st.file_uploader("Upload CT Image", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded CT Scan", use_container_width=True)

    if st.button("Analyze"):
        label, confidence = predict_image(image)

        st.subheader("🔍 Prediction Result")
        st.write(f"**Class:** {label}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

        if label != "Normal cases":
            cam_fig = gradcam_image(image)

            st.subheader("Affected Area (Grad-CAM)")
            st.pyplot(cam_fig)
