# Semantic Segmentation with PyTorch

## Overview
This project focuses on **semantic segmentation** using PyTorch. It includes dataset handling, data augmentation, model training, and evaluation. The dataset can be downloaded from Kaggle, and the model is trained using various transformations and optimizations.

## Features
- Custom dataset class for image-mask pairing.
- Data augmentation using `albumentations`.
- Model training and evaluation using PyTorch.
- Grad-CAM visualization for interpretability.
- Performance tracking with metrics like **IoU (Intersection over Union), Dice Coefficient, and Accuracy**.

## Installation

Ensure you have Python 3.7+ installed, then install dependencies:

```bash
pip install -r requirements.txt
```

Additional dependencies:
```bash
pip install torch torchvision albumentations numpy matplotlib pillow kaggle
```

## Dataset Preparation
This project uses the **Cardiac Segmentation Dataset**. You can download and extract it using:

```python
from data_loader import data_yuklab_olish
data_yuklab_olish(saqlash_uchun_papka='dataset_folder', data_nomi='cardiac')
```

## Model Training

### Training the Model
Run the following command to train the segmentation model:

```python
python train.py --dataset cardiac --epochs 50 --batch_size 16 --lr 0.001
```

### Training Details
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **Learning Rate**: 0.001 (default, adjustable)
- **Batch Size**: 16 (can be adjusted based on GPU memory)
- **Epochs**: 50 (recommended)

### Performance Scores (Cardiac Dataset)
| Metric            | Score  |
|------------------|--------|
| IoU Score       | 0.85   |
| Dice Coefficient | 0.91   |
| Accuracy        | 94.3%  |

## Evaluating the Model
```python
python evaluate.py --dataset cardiac
```
This will compute the IoU, Dice coefficient, and accuracy for the test dataset.

## Visualizing Results
To visualize predictions using Grad-CAM:

```python
python visualize.py --dataset cardiac
```

## Folder Structure
```
├── data_loader.py      # Handles dataset loading
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── visualize.py        # Grad-CAM visualization
├── models/             # Model architectures
├── results/            # Stores output visualizations
├── logs/               # Training logs
├── data/               # Dataset folder
└── README.md           # Project documentation
```

## Contributing
Feel free to fork this repository and submit a pull request with improvements.

## License
This project is licensed under the MIT License.

