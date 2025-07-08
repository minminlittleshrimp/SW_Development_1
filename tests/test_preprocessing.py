import unittest
from src.preprocessing.preprocess import preprocess_data

class TestPreprocessing(unittest.TestCase):

    def test_preprocess_data(self):
        # Example input data
        raw_data = {
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        }
        
        # Expected processed data
        expected_data = {
            'feature1': [0.0, 0.5, 1.0],  # Example normalization
            'feature2': [0.0, 0.5, 1.0]   # Example normalization
        }
        
        # Call the preprocessing function
        processed_data = preprocess_data(raw_data)
        
        # Assert that the processed data matches the expected data
        self.assertEqual(processed_data, expected_data)

if __name__ == '__main__':
    unittest.main()