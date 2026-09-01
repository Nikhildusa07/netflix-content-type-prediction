# Netflix Content Type Prediction

A machine learning classification system that predicts whether a Netflix title is a **Movie** or **TV Show** using available content metadata.

## Features

- Data preprocessing
- Exploratory Data Analysis
- Feature engineering
- TF-IDF feature extraction
- Logistic Regression classification
- Model evaluation
- Flask web application
- Movie / TV Show prediction

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Flask
- HTML
- CSS

## Machine Learning Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
TF-IDF Vectorization
   ↓
Train/Test Split
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Flask Application
```

## Features Used

The model uses:

```text
director
country
rating
listed_in
```

The target variable is:

```text
type
```

where:

```text
Movie   → 0
TV Show → 1
```

## Project Structure

```text
netflix-content-type-prediction/
│
├── app/
│   ├── __init__.py
│   └── app.py
│
├── data/
│   └── Dataset.csv
│
├── models/
│   ├── type_prediction_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── train.py
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   ├── index.html
│   └── results.html
│
├── screenshots/
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

### Clone Repository

```bash
git clone https://github.com/Nikhildusa07/netflix-content-type-prediction.git
```

### Navigate to Project

```bash
cd netflix-content-type-prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python -m src.train
```

The trained files will be saved in:

```text
models/
├── type_prediction_model.pkl
└── tfidf_vectorizer.pkl
```

## Run the Application

```bash
python -m app.app
```

Open:

```text
http://127.0.0.1:5000
```

Enter the content metadata and click **Predict Content Type**.

## Model

The project uses:

**TF-IDF Vectorization**

to convert text-based metadata into numerical features.

**Logistic Regression**

is used as the classification algorithm.

## Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Limitations

The prediction quality depends on the metadata available in the dataset.

The model does not use:

- User viewing history
- User preferences
- Watch history
- User ratings

## Future Improvements

- Add additional metadata features
- Compare multiple classification algorithms
- Hyperparameter tuning
- Add cross-validation
- Improve feature engineering
- Add model performance visualization
- Deploy the application

## Author

**Nikhil Dusa**

GitHub:

https://github.com/Nikhildusa07