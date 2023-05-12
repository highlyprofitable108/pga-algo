import pandas as pd
import numpy as np
from src.preprocess_data import preprocess_data

def test_preprocess_data():
    # Test case 1: Check that missing values are handled correctly
    data = pd.DataFrame({'A': [1, 2, np.nan], 'B': ['a', 'b', 'c']})
    processed_data = preprocess_data(data)
    assert processed_data.isnull().sum().sum() == 0, "Missing values were not handled correctly"

    # Test case 2: Check that categorical variables are encoded correctly
    # Adjust this test case based on how your function handles categorical variables
    data = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    processed_data = preprocess_data(data)
    assert processed_data['B'].dtype == 'int', "Categorical variables were not encoded correctly"
