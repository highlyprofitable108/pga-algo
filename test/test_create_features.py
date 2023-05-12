import unittest
import pandas as pd
from src.create_features import create_features

class TestCreateFeatures(unittest.TestCase):

    def test_create_features(self):
        # Create a sample DataFrame for testing
        data = pd.DataFrame({
            'strokes_gained': [10, 20, 30],
            'course_length': [400, 500, 600],
            'hazards': [5, 10, 15],
            'temp_min': [10, 15, 20],
            'temp_max': [20, 25, 30],
            'wind_speed': [5, 10, 15],
            'player_ranking': [1, 2, 3]
        })

        expected_data = pd.DataFrame({
            'strokes_gained': [0.0, 0.5, 1.0],  # Update with scaled values
            'course_length': [400, 500, 600]
        }, dtype=float)

        # Apply create_features function
        result = create_features(data)

        # Assert that the result matches the expected output
        pd.testing.assert_frame_equal(result, expected_data)

if __name__ == '__main__':
    unittest.main()
