# Ants & Bees Image Classification

This project is an image classification model that distinguishes between ants and bees using deep learning. It utilizes a convolutional neural network (CNN) built from scratch as well as a pre-trained ResNet18 model.

## Features
- Downloads dataset from Google Drive
- Custom PyTorch dataset class
- CNN model for classification
- Pre-trained ResNet18 for transfer learning
- Training with early stopping
- Model evaluation with visualization

## Installation
Ensure you have Python installed along with the required dependencies.

```bash
pip install torch torchvision timm gdown matplotlib tqdm
```

## Dataset
The dataset is automatically downloaded from Google Drive using the function `download_from_cloud`. It is then loaded and preprocessed for training.

## Model Training
The project provides two models:
1. A custom CNN (`conv_model`)
2. A pre-trained ResNet18 (`timm.create_model('resnet18')`)

Training is handled by `do_train`, which implements early stopping to prevent overfitting.

## Usage
Run the training script:

```python
python train.py
```

After training, the best model is saved in `best_model_dir/best_model.pt`.

## Inference
To test the model on new images:

```python
python inference.py
```

The script will display predictions for random test images.

## Results
The model achieves high accuracy in classifying ants and bees. The inference script visualizes predictions with labels.

## Acknowledgments
- [PyTorch](https://pytorch.org/)
- [TIMM (Torchvision Image Models)](https://github.com/rwightman/pytorch-image-models)
- Google Drive for dataset storage

## License
This project is licensed under the MIT License.

