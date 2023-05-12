import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import Lasso

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
                                bins=[0, 1, 4, 7, 11, 17, 24, 31, 39, 47, 56, 64, np.inf],
                                labels=list(range(0, 12)))

    # Feature scaling (normalize the data to the range [0, 1])
    for col in ['strokes_gained', 'course_length', 'hazards', 'avg_temperature', 'wind_speed']:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    # Investigate additional interaction features
    # data['interaction_feature_1'] = data['feature1'] * data['feature2']

    # Investigate polynomial features
    # poly = PolynomialFeatures(degree=2, include_bias=False)
    # poly_features = poly.fit_transform(data[['feature1', 'feature2']])
    # poly_columns = poly.get_feature_names(['feature1', 'feature2'])
    # can you get medata[poly_columns] = pd.DataFrame(poly_features, columns=poly_columns)

    # Incorporate player-specific features
    data['player_ranking_normalized'] = (data['player_ranking'] - data['player_ranking'].min()) / (
                data['player_ranking'].max() - data['player_ranking'].min())

    # Create categorical features for weather
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    temperature_bins = discretizer.fit_transform(data['avg_temperature'].values.reshape(-1, 1))
    data['temperature_category'] = temperature_bins

    # Perform feature selection (example using Lasso regularization)
    # Perform feature selection (example using Lasso regularization)
    X = data.drop(['strokes_gained', 'course_length'], axis=1)
    y = data[['strokes_gained', 'course_length']]  # Separate the target columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_scaled, y)
    selected_features = X.columns[lasso.coef_ != 0]
    data = pd.concat([data[selected_features], y], axis=1)  # Include the target columns in the result

