import pandas as pd
import requests
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

def train_model(data, target):
    """
    Train the model using Ridge regression with polynomial features and cross-validation
    to find the best hyperparameters.
    
    :param data: pandas DataFrame containing the preprocessed golf data
    :param target: pandas Series containing the target variable
    :return: trained model
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Create a pipeline with PolynomialFeatures and Ridge regression
    model = Pipeline([
        ('poly', PolynomialFeatures()),
        ('ridge', Ridge())
    ])

    # Define the hyperparameters and their possible values for fine-tuning
    params = {
        'poly__degree': range(1, 4),
        'ridge__alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100]
    }

    # Perform a grid search with cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', random_state=42)
    grid_search.fit(X_train, y_train)

    # Train the model with the best hyperparameters on the entire training set
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae:.2f}')

    return best_model

# TODO:
# 1. Experiment with different regression models, such as Lasso, ElasticNet, or
#    more advanced models like RandomForestRegressor or XGBoost, and compare their performance.
# 2. Optimize the cross-validation process by using different techniques, like
#    KFold, StratifiedKFold, or TimeSeriesSplit, depending on the data's characteristics.
# 3. Include additional hyperparameters in the grid search to further fine-tune the model.
# 4. Implement a more efficient search algorithm, such as RandomizedSearchCV or
#    Bayesian optimization, to reduce the computation time required for hyperparameter tuning.
# 5. Incorporate a custom scoring function in the grid search to better align with
#    the project's objectives (e.g., minimize absolute error instead of squared error).
# 6. Evaluate the model using multiple performance metrics to gain a more comprehensive
#    understanding of its performance.
# 7. Implement an early stopping mechanism during training to prevent overfitting
#    and reduce computation time.
# 8. Explore feature selection techniques to identify the most important features
#    and improve the model's efficiency and interpretability.
