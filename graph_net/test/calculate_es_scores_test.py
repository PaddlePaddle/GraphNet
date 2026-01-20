import unittest
import os
from os.path import dirname
import graph_net
from graph_net.fault_locator.calculate_es_scores import calculate_es_scores


class TestCalculateESScores(unittest.TestCase):
    def setUp(self):
        """Initialize the log file path relative to the graph_net package."""
        self.log_file = os.path.join(
            dirname(graph_net.__file__),
            "test",
            "data_calculate_es_scores",
            "evaluation.log",
        )

    def test_calculate_es_scores_monotonicity(self):
        """
        Verifies:
        1. Type is dict[int, float].
        2. Keys and values are within range [-10, 4].
        3. Monotonicity: result[i-1] < result[i] for all keys.
        """
        self.assertTrue(
            os.path.exists(self.log_file), f"Log file missing: {self.log_file}"
        )

        results = calculate_es_scores(self.log_file)

        # 1. Basic Type and Range Verification
        self.assertIsInstance(results, dict)
        sorted_keys = sorted(results.keys())

        # Check range [-10, 4] for both keys and values
        for k in sorted_keys:
            self.assertIsInstance(k, int)
            self.assertTrue(-10 <= k <= 4, f"Key {k} out of range")

            val = results[k]
            self.assertIsInstance(val, float)
            self.assertTrue(-10 <= val <= 4, f"Value {val} out of range")

        # 2. Monotonicity Check: result[i-1] < result[i]
        # We iterate through the sorted keys to compare consecutive entries
        for idx in range(1, len(sorted_keys)):
            prev_key = sorted_keys[idx - 1]
            curr_key = sorted_keys[idx]

            # Ensure keys are consecutive (optional, based on your range -10 to 4)
            # If they aren't strictly consecutive, the logic still holds for sorted keys
            self.assertLessEqual(
                results[prev_key],
                results[curr_key],
                f"Monotonicity failed: result[{prev_key}] ({results[prev_key]}) "
                f"is not less than result[{curr_key}] ({results[curr_key]})",
            )


if __name__ == "__main__":
    unittest.main()
