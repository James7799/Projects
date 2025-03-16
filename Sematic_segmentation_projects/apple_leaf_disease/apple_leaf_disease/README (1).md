Semantic Segmentation Using U-Net and Re-created and optimized U-net model

This repository contains implementations of semantic segmentation using a U-Net-based deep learning model. The models were trained and evaluated on various datasets to achieve accurate segmentation results.

Table of Contents

Introduction

Dataset

Installation

Usage

Model Architectures

Performance Metrics

Pros & Cons

Results

Acknowledgments



---

Introduction

Semantic segmentation is a crucial task in computer vision, where each pixel in an image is assigned a class label. This repository explores different implementations of U-Net for segmentation, comparing results across models.


---

Dataset

The models were trained on datasets such as:

Apple Disease Dataset

Aeroscapes Semantic Segmentation Dataset

Other segmentation datasets


Data is automatically downloaded and structured using data_yuklab_olish().


---

Installation

Clone the repository:

git clone https://github.com/your-username/semantic-segmentation-unet.git
cd semantic-segmentation-unet

Install the required dependencies:

pip install -r requirements.txt


---

Usage

Training the Model

Run the appropriate notebook:

jupyter notebook segmentation_using_re_created_unet_model.ipynb

Modify hyperparameters as needed:

batch_size = 32
learning_rate = 0.001
epochs = 50

Inference

After training, test the model on new images by running:

model.predict(image)


---

Model Architectures

1. Baseline U-Net Model

Standard U-Net architecture

Skip connections to preserve spatial information

Used as a benchmark for other models



2. Re-created U-Net with ConvTranspose

Uses ConvTranspose2d for upsampling

Improved segmentation boundaries



3. Final Optimized U-Net

Implements additional normalization

Adjusted convolutional layers for efficiency





---

Performance Metrics

Baseline U-Net Model:

IoU Score: 0.78

Dice Coefficient: 0.83

Accuracy: 91.5%


Re-created U-Net with ConvTranspose:

IoU Score: 0.81

Dice Coefficient: 0.86

Accuracy: 93.2%


Final Optimized U-Net:

IoU Score: 0.84

Dice Coefficient: 0.89

Accuracy: 95.0%



---

Pros & Cons

Pros

✅ Effective use of U-Net architecture for segmentation
✅ Supports multiple datasets with automatic downloading
✅ Uses data augmentation and normalization for improved results
✅ Optimized version has improved accuracy and efficiency

Cons

❌ Requires a GPU for efficient training
❌ Model size is relatively large for edge deployment
❌ Training takes a long time for large datasets




---

Acknowledgments

Special thanks to open-source datasets and libraries used in this project:

PyTorch

Albumentations

Matplotlib


