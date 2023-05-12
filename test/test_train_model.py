class TestTrainModel(unittest.TestCase):

    def test_train_model(self):
        # Create a sample DataFrame for testing
        data = pd.DataFrame({
            'strokes_gained': np.random.rand(100),
            'course_length': np.random.randint(400, 600, 100)
        })

        target = data['strokes_gained'] * 2 + data['course_length'] * 3 + np.random.rand(100)

        # Apply train_model function
        model = train_model(data, target)

        # Assert that the model has been trained, i.e., it has coefficients
        self.assertTrue(hasattr(model.named_steps['ridge'], 'coef_'))

if __name__ == '__main__':
    unittest.main()
