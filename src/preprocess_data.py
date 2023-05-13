import pandas as pd
import requests
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# TODO Update warning imports
    
def preprocess_data(data):
    """
    Preprocess the data, including handling missing values, encoding categorical variables,
    and scaling numerical variables.
    
    :param data: pandas DataFrame containing the golf data
    :return: pandas DataFrame with preprocessed data
    """

    # Handle missing values
    # Replace missing numerical values with the median value of the column
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    numerical_cols.remove('sg_total')  # exclude target variable from scaling
    
    for col in numerical_cols:
        data[col].fillna(data[col].median(), inplace=True)

    # Replace missing categorical values with the mode of the column
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Encode categorical variables
    # One-hot encode categorical variables with multiple categories
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Scale numerical variables
    # Normalize numerical variables to the range [0, 1]
    for col in numerical_cols:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    return data
