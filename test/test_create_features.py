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
            'player_ranking': [1,2,3]
        })

        expected_data = pd.DataFrame({
            'strokes_gained': [10, 20, 30],
            'course_length': [400, 500, 600],
            'hazards': [5, 10, 15],
            'temp_min': [10, 15, 20],
            'temp_max': [20, 25, 30],
            'wind_speed': [5, 10, 15],
            'strokes_gained_per_course_length': [0.025, 0.04, 0.05],
            'strokes_gained_per_hazards': [2.0, 2.0, 2.0],
            'avg_temperature': [15.0, 20.0, 25.0],
            'wind_force': [1, 2, 2],
            'player_ranking': [1, 2, 3]
        })

        # Apply create_features function
        result = create_features(data)

        # Check if 'player_ranking_normalized' column exists
        if 'player_ranking_normalized' in expected_data.columns:
            # Remove 'player_ranking_normalized' column from expected_data
            expected_data = expected_data.drop('player_ranking_normalized', axis=1)

        # Assert that the result matches the expected output
        pd.testing.assert_frame_equal(result, expected_data)

if __name__ == '__main__':
    unittest.main()
