import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from src.evaluate_model import evaluate_model

class TestEvaluateModel(unittest.TestCase):
    
    def setUp(self):
        # Create a simple regression dataset
        X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train several models on the data
        self.models = [
            LinearRegression().fit(self.X_train, self.y_train),
            Lasso().fit(self.X_train, self.y_train),
            Ridge().fit(self.X_train, self.y_train),
        ]

    def test_evaluate_model(self):
        # Test that the evaluate_model function runs without error
        mae, rmse = evaluate_model(self.models, self.X_test, self.y_test)
        
        # Test that the function returns reasonable results
        self.assertTrue(0 <= mae < np.inf)
        self.assertTrue(0 <= rmse < np.inf)
        
        # Test that the function returns the same results when run multiple times on the same data
        mae2, rmse2 = evaluate_model(self.models, self.X_test, self.y_test)
        self.assertEqual(mae, mae2)
        self.assertEqual(rmse, rmse2)


if __name__ == '__main__':
    unittest.main()
