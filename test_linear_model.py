# test_linear_model.py

import unittest
import numpy as np
import pandas as pd
from linear_model import train_model

class TestLinearModel(unittest.TestCase):

    def test_train_model_output(self):
        # Dummy dataset
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100)

        model = train_model(X_train, y_train)
        predictions = model.predict(X_train)

        self.assertEqual(len(predictions), 100)
        self.assertTrue(np.allclose(predictions.shape, (100,)))

if __name__ == '__main__':
    unittest.main()
