# Cats vs Dogs Classifier Using Convolutional Neural Networks (CNN)

## Overview
This project implements a CNN-based binary image classifier to distinguish between images of cats and dogs. The model is built using **TensorFlow** and **Keras** libraries and trained on the popular **Dogs vs Cats** dataset from Kaggle.

## Features
- Preprocessing of images using TensorFlow's `image_dataset_from_directory`.
- Data normalization for faster convergence.
- A CNN architecture with:
  - Convolutional and MaxPooling layers.
  - Batch Normalization for stable training.
  - Dense layers with Dropout for regularization.
- Model performance evaluation with loss and accuracy visualization.

## Dataset
The dataset used is the **Dogs vs Cats** dataset, which you can download from [Kaggle](https://www.kaggle.com/salader/dogs-vs-cats).  
It contains labeled images of cats and dogs, split into `train` and `test` directories.

## Prerequisites
Before running this project, ensure you have the following installed:
- Python 3.8+
- TensorFlow
- Keras
- Matplotlib
- Kaggle API (for dataset download)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Aishjahan/Cat-vs-Dog-classifier-CNN.git
   cd cats-vs-dogs-classifier
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle:
   Place your kaggle.json file in the ~/.kaggle directory.
   Run the following commands
   mkdir -p ~/.kaggle
   ```bash
      cp kaggle.json ~/.kaggle/
      kaggle datasets download -d salader/dogs-vs-cats
   ```

## Model Architecture
The CNN model consists of:

- **Convolutional Layers:** Extract spatial features.
- **MaxPooling Layers:** Reduce spatial dimensions.
- **Batch Normalization:** Improve training stability.
- **Dense Layers:** Fully connected layers with Dropout for classification.

### Summary of the Model:
- **Input:** Images of size `(256, 256, 3)`.
- **Output:** Binary classification (Cat: 0, Dog: 1).

## Training
Run the following script to train the model:
```bash
python train.py
```

### Model Hyperparameters:
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy
- **Epochs:** 10
- **Batch Size:** 32

### Performance Visualization:
After training, the script will generate plots for:
- Training and validation accuracy.
- Training and validation loss.

### Results
- The model achieves high accuracy on the validation set.
- Training and validation performance is visualized using Matplotlib.

### Usage
1. Place your own images in a directory for testing.
2. Modify the code to load and preprocess your images.
3. Use the trained model to predict whether the image is a cat or a dog.

### Visualizations
Sample plots of accuracy and loss:



