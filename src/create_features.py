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
    data['sg_putt_gir'] = data['sg_putt'] * data['gir']
    data['sg_app_prox_fw'] = data['sg_app'] * data['prox_fw']
    data['sg_ott_driving_dist'] = data['sg_ott'] * data['driving_dist']

    # Create polynomial features
    poly = PolynomialFeatures(2, interaction_only=True)
    numerical_cols = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'driving_dist', 'driving_acc', 'gir', 'scrambling', 'prox_rgh', 'prox_fw']
    poly_data = poly.fit_transform(data[numerical_cols])
    target_feature_names = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(numerical_cols,p) for p in poly.powers_]]
    poly_data = pd.DataFrame(poly_data, columns=target_feature_names)

    # Merge the polynomial features with the original data
    data = pd.concat([data, poly_data], axis=1)
    
    # Calculate the average performance per season
    data['avg_sg_season'] = data.groupby('season')['sg_total'].transform('mean')
    
    # Convert year to "years since start of data"
    data['years_since_start'] = data['year'] - data['year'].min()

    # Add player performance per course
    data['avg_sg_player_course'] = data.groupby(['player_name', 'course_name'])['sg_total'].transform('mean')


    # TODO: Incorporate weather data. You can fetch historical weather data based on the date and location of each event, 
    # and add this data as new features. Consider weather conditions like temperature, precipitation, and wind speed.
    
    # Perform feature selection (example using Lasso regularization)
    X = data.drop('sg_total', axis=1)  # Remove the target variable
    y = data['sg_total'].values.reshape(-1, 1)  # Separate the target column
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_scaled, y)
    feature_coef = pd.Series(lasso.coef_, index=X.columns)
    selected_features = feature_coef[feature_coef != 0].index.tolist()

    # Include only the selected features in the result
    selected_data = data[selected_features]

    return selected_data
