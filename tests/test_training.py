import unittest
from src.training.train import train_model
from src.training.utils import load_data

class TestTraining(unittest.TestCase):

    def setUp(self):
        self.train_data, self.val_data = load_data()

    def test_train_model(self):
        model = train_model(self.train_data, self.val_data)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

    def test_data_loading(self):
        self.assertGreater(len(self.train_data), 0)
        self.assertGreater(len(self.val_data), 0)

if __name__ == '__main__':
    unittest.main()