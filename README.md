# Unet-Background-Changer

This project uses a UNET model for binary segmentation to create a background remover and changer tool. The tool is implemented in Python and is accessible via a Streamlit web interface. It allows users to upload an image, remove its background, and optionally replace it with a new background.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Performance](#performance)
- [Technologies](#technologies)
- [Acknowledgements](#acknowledgements)

## Introduction

The purpose of this project is to provide a tool for segmenting and modifying image backgrounds using a deep learning approach. A UNET model is trained to accurately segment the foreground object from the background. Once the background is removed, users can choose to replace it with a new background.

## Features

- Binary image segmentation using a UNET model.
- Remove the background of an image.
- Option to replace the background with a new image.
- Simple and interactive web interface built with Streamlit.

## Installation

To run this project locally, follow the steps below:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/PritiyaxShukla/Unet-Background-Chnager.git
    cd Unet-Background-Chnager
    ```

2. **Install the required dependencies:**

    Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

    Install the dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the trained UNET model:**

    Download the pre-trained UNET model Binary_Unet_Model.h5 and place it in the `models/` directory.

4. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

    The app should now be accessible at `http://localhost:8501`.

## Usage

1. Open the web app.
2. Upload an image file (JPEG or PNG).
3. Use the "Remove Background" button to remove the background of the uploaded image.
4. (Optional) Upload a new background image to replace the removed background.
5. Download the processed image.

## Model Training

The UNET model is trained using the [DUTS Saliency Detection Dataset]. Below are the training details:

- **Training Accuracy:** 98.16%
- **Validation Accuracy:** 92.86%
- **Training Loss:** 0.0462
- **Validation Loss:** 0.3004
- **Epochs:** 50
- **Model Architecture:** UNET

The model was trained to achieve a balance between performance and computational efficiency, with a primary focus on foreground-background separation for image editing tasks.

## Performance

The model performed well on segmentation tasks, producing the following results:

- **Training Accuracy:** 98.16%
- **Validation Accuracy:** 92.86%
- **Training Loss:** 0.0462
- **Validation Loss:** 0.3004

Further improvements such as adding Intersection over Union (IoU) as a metric were considered but omitted due to GPU time constraints.

## Technologies

- **Python**: Programming language for model development and backend.
- **TensorFlow/Keras**: Used for building and training the UNET model.
- **OpenCV**: For image processing tasks.
- **Streamlit**: For creating the interactive web app.
- **Matplotlib**: For visualizing images and results.

## Acknowledgements

- [UNET Architecture](https://arxiv.org/abs/1505.04597) for image segmentation.
- [Streamlit](https://streamlit.io/) for providing a fast and simple way to deploy the app.
- [DUTS Saliency Detection Dataset](https://www.kaggle.com/datasets/balraj98/duts-saliency-detection-dataset) for training data.

## License

