# Sprint Project 02 - Home Credit Default Risk

Build a binary classification model to predict loan repayment risk using machine learning.

## Tech Stack

- **Python 3.9+** - Main programming language
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Feature engineering and ML models
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive experimentation
- **Pytest** - Testing framework
- **Black & isort** - Code formatting

## Setup & Run

1. Install dependencies (use virtual environment recommended):
```bash
pip install -r requirements.txt
```

2. Open and run the Jupyter notebook:
```bash
jupyter notebook "AnyoneAI - Sprint Project 02.ipynb"
```

3. Complete the TODO sections in the notebook (data will auto-download in Section 1).

4. Run tests:
```bash
pytest tests/
```

5. Format code:
```bash
isort --profile=black . && black --line-length 88 .
```

## Project Structure

```
├── dataset/                 # Training and test data (auto-downloaded)
├── src/                     # Source code modules
├── tests/                   # Unit tests
├── AnyoneAI - Sprint Project 02.ipynb
├── README.md
└── requirements.txt
```

## Key Concepts Covered

- **Binary Classification** - Predicting loan default risk
- **Feature Engineering** - Creating and selecting relevant features
- **Model Training** - Building ML models with Scikit-learn
- **Model Evaluation** - ROC-AUC metric and performance analysis
- **Data Preprocessing** - Handling missing values and encoding
- **Exploratory Data Analysis** - Understanding data patterns

## Business Problem

Predict whether a loan applicant will have payment difficulties (binary classification):
- **1**: Client will have late payment (>X days) on at least one of the first Y installments
- **0**: All other cases

**Evaluation Metric**: Area Under the ROC Curve (ROC-AUC)

**Data**: Home credit application data (`application_train_aai.csv` and `application_test_aai.csv`)
