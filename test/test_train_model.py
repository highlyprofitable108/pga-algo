import unittest
import pandas as pd
import numpy as np
from src.train_model import train_model

class TestTrainModel(unittest.TestCase):

    def test_train_model(self):
        # Create a sample DataFrame for testing
        data = pd.DataFrame({
            'strokes_gained': np.random.rand(100),
            'course_length': np.random.randint(400, 600, 100)
        })

        target = data['strokes_gained'] * 2 + data['course_length'] * 3 + np.random.rand(100)

        # Apply train_model
