# Rice Leaf Disease Classification Project

This project focuses on classifying images of rice leaves to detect diseases using deep learning techniques. The dataset used in this project is the **Rice Leaf Disease Classification Dataset**, which contains images of healthy and diseased rice leaves. The goal is to build and train a model that can accurately classify these images into their respective categories.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training and Validation](#training-and-validation)
5. [Results](#results)
6. [Inference](#inference)
7. [Usage](#usage)
8. [Dependencies](#dependencies)
9. [License](#license)

## Project Overview

The project involves the following steps:
- **Data Downloading**: The dataset is downloaded from Kaggle using the `kaggle datasets download` command.
- **Data Preprocessing**: The dataset is preprocessed and split into training, validation, and test sets.
- **Model Building**: A pre-trained model (`rexnet_150`) from the `timm` library is fine-tuned for the classification task.
- **Training and Validation**: The model is trained and validated using the training and validation datasets.
- **Inference**: The trained model is used to make predictions on the test dataset, and the results are visualized using Grad-CAM (Gradient-weighted Class Activation Mapping).

## Dataset

The dataset used in this project is the **Rice Leaf Disease Classification Dataset**, which can be downloaded from Kaggle using the following command:

```bash
kaggle datasets download -d killa92/rice-leaf-disease-classification-dataset
```

The dataset contains images of rice leaves with various diseases, and the goal is to classify these images into their respective categories.

## Model Architecture

### Pre-trained Model (`rexnet_150`)
The pre-trained model used in this project is `rexnet_150` from the `timm` library. This model is fine-tuned on the rice leaf disease dataset for classification.

## Training and Validation

The training and validation process involves the following steps:
- **Data Splitting**: The dataset is split into training (80%), validation (10%), and test (10%) sets.
- **Data Augmentation**: Data augmentation techniques such as resizing, normalization, and random cropping are applied to the training data.
- **Training**: The model is trained using the Adam optimizer and CrossEntropyLoss as the loss function.
- **Validation**: The model's performance is evaluated on the validation set after each epoch.
- **Early Stopping**: Training is stopped early if the validation accuracy does not improve for a specified number of epochs (patience).

## Results

The results of the training and validation process are visualized using learning curves, which show the training and validation loss and accuracy over epochs. The best model is saved based on the highest validation accuracy.

### Learning Curves
- **Training Loss vs. Validation Loss**: Shows the decrease in loss over epochs for both training and validation sets.
- **Training Accuracy vs. Validation Accuracy**: Shows the increase in accuracy over epochs for both training and validation sets.

## Inference

The trained model is used to make predictions on the test dataset. The predictions are visualized using Grad-CAM, which highlights the regions of the image that the model focused on to make the prediction.

### Grad-CAM Visualization
- **Heatmap**: A heatmap is overlaid on the original image to show the regions that contributed most to the model's prediction.
- **Ground Truth vs. Prediction**: The ground truth label and the predicted label are displayed for each image.

## Usage

To use this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/rice-leaf-disease-classification.git
   cd rice-leaf-disease-classification
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   ```bash
   kaggle datasets download -d killa92/rice-leaf-disease-classification-dataset
   unzip rice-leaf-disease-classification-dataset.zip -d datasets_leaf
   ```

4. **Run the Training Script**:
   ```bash
   python train.py
   ```

5. **Run the Inference Script**:
   ```bash
   python inference.py
   ```

## Dependencies

The following dependencies are required to run this project:
- Python 3.x
- PyTorch
- torchvision
- timm
- numpy
- matplotlib
- opencv-python
- tqdm

You can install the dependencies using the following command:
```bash
pip install torch torchvision timm numpy matplotlib opencv-python tqdm
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

This README provides an overview of the Rice Leaf Disease Classification Project. For more detailed information, please refer to the code and comments in the respective scripts.
