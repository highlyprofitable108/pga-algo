import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def evaluate_model(models, X_test, y_test):
    """
    Evaluate the performance of an ensemble of models on a test set.

    :param models: a list of trained models
    :param X_test: pandas DataFrame, test data (features)
    :param y_test: pandas Series, test data (target)
    :return: None
    """
    # Make predictions on the test set with each model
    predictions = [model.predict(X_test) for model in models]

    # Average the predictions to get the final prediction
    y_pred = np.mean(predictions, axis=0)

    # Calculate the Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error (MAE): {mae:.2f}')

    # Calculate the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

    # Plot actual vs predicted values
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title('Actual vs Predicted Golf Scores')
    plt.show()

    return mae, rmse
