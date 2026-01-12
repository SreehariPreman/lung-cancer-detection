# Lung Cancer Detection from CT Scans

A deep learning-powered web application that analyzes CT scan images to detect and classify lung cancer cases. The system uses a ResNet18 convolutional neural network to classify scans into three categories: Normal, Benign, and Malignant, with visual explanations through Grad-CAM heatmaps.

## Features

- **Automated Classification**: Classifies CT scans into three categories:
  - Normal cases
  - Benign cases
  - Malignant cases
- **Confidence Scores**: Provides confidence percentages for each prediction
- **Visual Explanations**: Generates Grad-CAM heatmaps to highlight affected areas in abnormal cases
- **User-Friendly Interface**: Built with Streamlit for easy interaction
- **GPU Support**: Automatically uses GPU acceleration when available

## Demo

Upload a CT scan image through the web interface, and the system will:
1. Analyze the image using a trained ResNet18 model
2. Predict the classification with confidence score
3. Generate a heatmap showing the most relevant areas for the prediction (for benign/malignant cases)

## Technology Stack

- **Deep Learning Framework**: PyTorch
- **Model Architecture**: ResNet18 (pretrained and fine-tuned)
- **Web Framework**: Streamlit
- **Visualization**: Grad-CAM, Matplotlib
- **Image Processing**: PIL, OpenCV, NumPy

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Install required dependencies:
```bash
pip install git+https://github.com/jacobgil/pytorch-grad-cam.git

pip install -r requirements.txt
```

2. Install PyTorch Grad-CAM:
```bash
pip install grad-cam
```

3. Ensure the model file `lung_cancer_model_40.pth` is in the project directory.

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the displayed local URL (typically `http://localhost:8501`)

3. Upload a CT scan image (PNG, JPG, or JPEG format)

4. Click the "Analyze" button to get the prediction

5. View the results:
   - Classification (Normal/Benign/Malignant)
   - Confidence percentage
   - Affected area heatmap (for abnormal cases)

## Model Details

- **Architecture**: ResNet18 modified for 3-class classification
- **Input Size**: 224x224 pixels
- **Input Format**: Grayscale CT images (converted to 3-channel)
- **Normalization**: Mean and std of 0.5 across all channels
- **Output**: Softmax probabilities for 3 classes

## Project Structure

```
lung-cancer-detection/
├── app.py                      # Main Streamlit application
├── lung_cancer_model_40.pth    # Trained model weights
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## How It Works

1. **Image Preprocessing**: Uploaded images are converted to grayscale, resized to 224x224 pixels, and normalized
2. **Prediction**: The ResNet18 model processes the image and outputs class probabilities
3. **Grad-CAM Visualization**: For abnormal cases, Grad-CAM highlights the regions that influenced the model's decision
4. **Results Display**: Shows the predicted class, confidence score, and heatmap overlay

## Grad-CAM Heatmap

The Grad-CAM (Gradient-weighted Class Activation Mapping) visualization helps interpret the model's decision by highlighting:
- **Red/Yellow areas**: High importance regions that strongly influenced the prediction
- **Blue/Purple areas**: Low importance regions with minimal impact

This provides transparency and helps medical professionals understand which areas the model focused on.



