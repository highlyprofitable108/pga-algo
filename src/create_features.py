import pandas as pd
import requests
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Import custom functions
from src.preprocess_data import preprocess_data
from src.train_model import train_model
from datetime import datetime

# TODO: Implement function for feature engineering
def create_features(data):
    """
    Create new features from the existing data.
    
    :param data: pandas DataFrame containing the golf data
    :return: pandas DataFrame with new features added
    """
    # Create interaction features between variables
    data['strokes_gained_per_course_length'] = data['strokes_gained'] / data['course_length']
    data['strokes_gained_per_hazards'] = data['strokes_gained'] / data['hazards']

    # Calculate the average temperature during the tournament
    data['avg_temperature'] = (data['temp_min'] + data['temp_max']) / 2

    # Convert wind speed to wind force category using the Beaufort scale
    data['wind_force'] = pd.cut(data['wind_speed'],
                                bins=[0, 1, 4, 7, 11, 17, 24, 31, 39, 47, 56, 64, float('inf')],
                                labels=list(range(0, 12)))

    # Feature scaling (normalize the data to the range [0, 1])
    for col in ['strokes_gained', 'course_length', 'hazards', 'avg_temperature', 'wind_speed']:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    return data
# TODO:
# 1. Investigate additional interaction features: Explore other meaningful interactions
#    between variables that could improve the model's performance.
# 2. Investigate polynomial features: Instead of just looking at interactions, consider
#    creating polynomial features (squared, cubic, etc.) of the existing variables.
# 3. Incorporate player-specific features: Include player-specific features such as
#    player ranking, age, or past performance to capture individual differences.
# 4. Create categorical features for weather: Convert continuous weather variables like
#    temperature, wind speed, and humidity into categorical features (e.g., temperature range)
#    to capture non-linear relationships.
# 5. Perform feature selection: Use techniques like recursive feature elimination or
#    regularization methods (Lasso, Ridge) to identify the most important features for the model.
# 6. Use domain knowledge to create additional relevant features: Leverage your understanding
#    of golf to create features that could be relevant to player performance.
# 7. Experiment with different feature scaling methods: Try standardization or other
#    scaling techniques to see if they improve the model's performance.
