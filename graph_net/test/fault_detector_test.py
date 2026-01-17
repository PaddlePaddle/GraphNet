import unittest
from graph_net.fault_locator.fault_detector import has_fault_at, ES


class TestHasFaultAt(unittest.TestCase):
    def test_fault_detected(self):
        """Case: ES[t-1] is less than ES[max], should return True."""
        # tolerance = 5, so we look at ES[4]
        es_data: ES = {0: 0.1, 4: 0.5, 10: 0.9}
        tolerance = 5
        # 0.5 < 0.9 is True
        self.assertTrue(has_fault_at(es_data, tolerance))

    def test_no_fault_detected(self):
        """Case: ES[t-1] is equal to or greater than ES[max], should return False."""
        # tolerance = 11, we look at ES[10]. Max key is 10.
        es_data: ES = {0: 0.1, 5: 0.5, 10: 0.9}
        tolerance = 11
        # 0.9 < 0.9 is False
        self.assertFalse(has_fault_at(es_data, tolerance))

    def test_assertion_error(self):
        """Case: tolerance-1 is not in ES, should raise AssertionError."""
        es_data: ES = {0: 0.1, 1: 0.2}
        tolerance = 5  # 4 is not in ES
        with self.assertRaises(AssertionError):
            has_fault_at(es_data, tolerance)

    def test_edge_case_single_element(self):
        """Case: Only one element in ES, max is itself."""
        es_data: ES = {0: 0.5}
        tolerance = 1
        # 0.5 < 0.5 is False
        self.assertFalse(has_fault_at(es_data, tolerance))


if __name__ == "__main__":
    unittest.main()
