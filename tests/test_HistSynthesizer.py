import pandas as pd

import unittest
from diffprivlib.utils import PrivacyLeakWarning

from synthesis.hist_synthesis import HistSynthesizer


class TestHistSynthesizer(unittest.TestCase):
    def test_not_none(self):
        self.assertIsNotNone(HistSynthesizer)

    def test_no_params(self):
        synthesizer = HistSynthesizer()

        X = pd.DataFrame(
            {
                'gender': ['female', 'female', 'male', 'female', 'male'],
                'age': [27, 28, 40, 12, 22]
            }
        )

        with self.assertWarns(PrivacyLeakWarning):
            synthesizer.fit(X)


if __name__ == '__main__':
    unittest.main()