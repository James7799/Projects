# Math Score Prediction

## Overview
This project aims to predict students' math scores based on various demographic and academic factors using machine learning. The dataset includes features such as gender, race/ethnicity, parental education level, lunch type, and test preparation course completion.

## Features
- Data preprocessing (handling categorical and numerical data)
- Machine learning pipeline using Scikit-learn
- Linear Regression model for prediction
- Performance evaluation using R² score

## Dataset
The dataset used in this project is publicly available and can be downloaded from Google Drive. The following datasets are supported:
- Salaries
- Exams
- College
- Cars
- Mall

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/math-score-prediction.git
   cd math-score-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the script to download the dataset:
   ```python
   python data_download.py --dataset exams
   ```

## Usage
1. Load and preprocess the data:
   ```python
   from data_processing import load_data
   df = load_data("exams")
   ```
2. Train the model:
   ```python
   from model import train_model
   model, score = train_model(df)
   print("Model Performance (R² Score):", score)
   ```
3. Make predictions:
   ```python
   predictions = model.predict(new_data)
   ```

## Project Structure
```
math-score-prediction/
│── data_download.py      # Dataset downloader
│── data_processing.py    # Data preprocessing
│── model.py              # Machine learning model training
│── requirements.txt      # Required dependencies
│── README.md             # Project documentation
```

## Results
The trained Linear Regression model achieves an R² score of **0.856** on the test set, indicating good predictive performance.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

