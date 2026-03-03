import unittest
from graph_net.fault_locator.bi_search import bi_search

# Assuming bi_search is imported or defined above
# from your_module import bi_search


class TestFaultLocator(unittest.TestCase):
    def setUp(self):
        """Setup mock data for a graph with 10 operators (0-9)."""
        self.fault_index = 7
        self.model_path = "/mock/path/model.pdmodel"

    def mock_truncator(self):
        def func(path, split_point):
            """Simulates slicing; returns the index as a proxy for the sub-model."""
            return split_point

        return func

    def mock_evaluator(self, sub_model_id):
        """If the sub-model includes index 7, it returns an error score."""
        if sub_model_id >= self.fault_index:
            return [0.99]
        return [0.01]

    def mock_predicator(self, es_scores, tolerance):
        return any(score > tolerance for score in es_scores)

    def mock_stoper(self, history, high=None):
        """Stops when the search interval converges (range is 0 or 1)."""
        if len(history) < 2:
            return False
        return abs(history[-1][0] - history[-2][0]) <= 1

    def test_bi_search_finds_correct_index(self):
        """Verifies that the fault at index 7 is located."""
        # Use a simple lambda or function with an attribute if needed,
        # or just let bi_search handle the default/getattr.
        truncator = self.mock_truncator()
        setattr(truncator, "total_steps", 9)

        history, faulty_operator_index, faulty_model_path = bi_search(
            relative_model_path=self.model_path,
            truncator=truncator,
            evaluator=self.mock_evaluator,
            es_scores_calculator=lambda x: x,  # Mock ES calculator
            predicator=self.mock_predicator,
            stoper=self.mock_stoper,
            tolerance=0.5,
        )

        print(f"\nFault Test History: {history}")
        print(f"Detected faulty operator index: {faulty_operator_index}")

        # The result of the fault localization is directly provided by the function
        actual_fault_index = faulty_operator_index

        print(f"\nIdentified Fault Index: {actual_fault_index}")
        self.assertEqual(actual_fault_index, self.fault_index)

    def test_no_fault_scenario(self):
        """Verifies behavior when no fault is present in the graph."""

        # We define a local function to allow attribute assignment
        def clean_truncator(path, split_point):
            return split_point

        clean_truncator.total_steps = 5

        def healthy_evaluator(sub_model_id):
            return [0.01]

        history, faulty_operator_index, faulty_model_path = bi_search(
            relative_model_path=self.model_path,
            truncator=clean_truncator,
            evaluator=healthy_evaluator,
            es_scores_calculator=lambda x: x,  # Mock ES calculator
            predicator=self.mock_predicator,
            stoper=self.mock_stoper,
            tolerance=0.5,
        )

        print(f"No-Fault Test History: {history}")
        print(f"Detected faulty operator index: {faulty_operator_index}")

        # No fault should be detected
        final_status = history[-1][1]
        self.assertFalse(final_status)
        self.assertEqual(faulty_operator_index, -1)  # -1 indicates no fault found


if __name__ == "__main__":
    unittest.main()
