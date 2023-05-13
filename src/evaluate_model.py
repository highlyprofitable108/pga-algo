from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(models, X_test, y_test):
    """
    Evaluate the performance of an ensemble of models on a test set.

    :param models: a list of trained models
    :param X_test: pandas DataFrame, test data (features)
    :param y_test: pandas Series, test data (target)
    :return: MAE, RMSE, R2 Score, and Predicted values
    """
    try:
        # Make predictions on the test set with each model
        if isinstance(models, list):
            predictions = [model.predict(X_test) for model in models]
        else:
            # If a single model is passed
            predictions = [models.predict(X_test)]

        # Average the predictions to get the final prediction
        y_pred = np.mean(predictions, axis=0)

        # Calculate the Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, y_pred)
        print(f'Mean Absolute Error (MAE): {mae:.2f}')

        # Calculate the Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

        # Calculate the R2 Score
        r2 = r2_score(y_test, y_pred)
        print(f'R2 Score: {r2:.2f}')

        # Plot actual vs predicted values
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual Scores')
        plt.ylabel('Predicted Scores')
        plt.title('Actual vs Predicted Golf Scores')
        plt.show()

        return mae, rmse, r2, y_pred

    except Exception as e:
        print(f"Error occurred during model evaluation: {str(e)}")
        return None

# TODO: Implement cross-validation for more robust model evaluation
# TODO: Calculate confidence intervals for the model's predictions
# TODO: Implement residual plots to visualize model errors
# TODO: Add functionality to compare performance of different models
# TODO: Implement learning curves to understand model performance over time
