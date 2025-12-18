# Dangerous Farm Insects Classification

This project aims to classify various species of dangerous farm insects using Deep Learning techniques. It includes a data preprocessing pipeline for splitting and augmenting the dataset, as well as implementations of several popular Convolutional Neural Network (CNN) architectures.

## 🐛 Dataset

The dataset consists of images of the following farm insects:

*   Africanized Honey Bees (Killer Bees)
*   Aphids
*   Armyworms
*   Brown Marmorated Stink Bugs
*   Cabbage Loopers
*   Citrus Canker
*   Colorado Potato Beetles
*   Corn Borers
*   Corn Earworms
*   Fall Armyworms
*   Fruit Flies
*   Spider Mites
*   Thrips
*   Tomato Hornworms
*   Western Corn Rootworms

## 🏗️ Project Structure

*   **`src/main.py`**: Entry point for the data preprocessing pipeline.
*   **`src/data/`**: Contains the `Data_Preprocessor` class which handles:
    *   Stratified K-Fold splitting (Train/Val/Test).
    *   Class balancing via image augmentation.
*   **`src/architectures/`**: Jupyter notebooks containing model implementations and training loops for:
    *   AlexNet
    *   GoogLeNet (Inception v1)
    *   InceptionV3
    *   LeNet-5
    *   ResNet50
    *   VGG16
*   **`data/`**: Stores raw and processed image data.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed along with the following libraries:

*   TensorFlow / Keras
*   NumPy
*   scikit-learn
*   Pillow (PIL)

### Usage

1.  **Data Preprocessing**:
    Run the main script to process the raw data, create splits, and apply augmentation.
    ```bash
    python src/main.py
    ```
    This will generate the processed dataset in `data/processed/farm_insects/`.

2.  **Model Training**:
    Navigate to the `src/architectures/` directory and open the Jupyter notebook corresponding to the model you wish to train (e.g., `ResNet50.ipynb`, `VGG16 (4).ipynb`). Run the cells to train and evaluate the model.

## 🧠 Architectures Implemented

The project explores the performance of the following architectures on the insect classification task:

*   **LeNet-5**: A classic CNN architecture.
*   **AlexNet**: A deeper architecture that popularized CNNs.
*   **VGG16**: Known for its simplicity and depth.
*   **GoogLeNet / InceptionV3**: Efficient architectures using Inception modules.
*   **ResNet50**: Uses residual connections to allow training of very deep networks.