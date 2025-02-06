# Car Brands Classification using Advanced Training Methods

This repository contains the code and resources for a car brand classification project. The goal of this project is to classify images of cars into different brands using deep learning techniques. The project leverages advanced training methods to significantly improve the model's accuracy from 87% to an impressive 99%.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Advanced Training Methods](#advanced-training-methods)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The project focuses on classifying car images into various brands using a deep learning model. The initial model achieved an accuracy of 87%, which was significantly improved to 99% by employing advanced training techniques. These techniques include data augmentation, transfer learning, and fine-tuning the model architecture.

## Dataset

The dataset used in this project consists of images of cars from different brands. The dataset is organized into folders, each representing a specific car brand. The dataset includes a variety of car models and angles to ensure robust training.

### Dataset Structure
```
datasets/
    car_brands/
        brand1/
            image1.jpg
            image2.jpg
            ...
        brand2/
            image1.jpg
            image2.jpg
            ...
        ...
```

## Advanced Training Methods

To achieve the remarkable improvement in accuracy, several advanced training methods were employed:

1. **Data Augmentation**: Techniques such as rotation, flipping, and zooming were used to artificially increase the diversity of the training data, helping the model generalize better.

2. **Transfer Learning**: A pre-trained model (e.g., ResNet, VGG) was used as the base model. This approach leverages the knowledge learned from a large dataset (e.g., ImageNet) and fine-tunes it for the specific task of car brand classification.

3. **Fine-Tuning**: The pre-trained model was fine-tuned by unfreezing some of the top layers and training them with a lower learning rate. This allows the model to adapt more specifically to the car brand classification task.

4. **Triplet Loss**: An advanced loss function, triplet loss, was used to improve the model's ability to distinguish between different car brands. This method involves training the model to minimize the distance between images of the same brand while maximizing the distance between images of different brands.

5. **Learning Rate Scheduling**: A dynamic learning rate scheduler was implemented to adjust the learning rate during training, ensuring that the model converges more effectively.

## Results

The advanced training methods led to a significant improvement in the model's performance:

- **Initial Accuracy**: 87%
- **Final Accuracy**: 99%

The model's ability to correctly classify car brands improved dramatically, demonstrating the effectiveness of the advanced training techniques employed.

## Usage

To use this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/James7799/car_brands_classification.git
   cd car-brands-classification
   ```

2. **Install Dependencies**:
   Ensure you have the required dependencies installed. You can install them using the following command:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:
   Organize your dataset as described in the [Dataset](#dataset) section.

4. **Train the Model**:
   Run the training script to train the model using the advanced training methods:
   ```bash
   python train.py
   ```

5. **Evaluate the Model**:
   After training, you can evaluate the model's performance on the test dataset:
   ```bash
   python evaluate.py
   ```

6. **Make Predictions**:
   Use the trained model to make predictions on new images:
   ```bash
   python predict.py --image_path path_to_image.jpg
   ```

## Dependencies

The project requires the following dependencies:

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- OpenCV
- Pandas

