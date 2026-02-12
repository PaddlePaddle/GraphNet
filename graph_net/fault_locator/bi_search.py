def bi_search(
    relative_model_path,
    truncator,  # Signature: (relative_model_path, split_point) -> relative_model_path
    evaluator,  # Signature: (relative_model_path) -> log_content
    es_scores_calculator,  # Signature: (log_content) -> ES
    predicator,  # Signature: (ES, tolerance) -> bool
    stoper,  # Signature: (history_list) -> bool
    tolerance=0,
) -> (list[(int, bool)], int):
    """
    Binary Search Algorithm for Automatic Fault Location with Faulty Operator Detection.

    This algorithm locates the first faulty operation in a computational graph
    by iteratively narrowing the search range through graph truncation and
    backend execution comparison.

    Args:
        relative_model_path (str): Path to the original computational graph.
        truncator (callable): Logic to slice the model at a specific split point.
        evaluator (callable): Logic to Evaluate a given model.
        es_scores_calculator (callable): Logic to calculate Error Score (ES) for a given log file.
        predicator (callable): Logic to determine fault status from ES and tolerance.
        stoper (callable): Logic to evaluate termination based on search history.
        tolerance (int): Numerical threshold for fault detection.

    Returns:
        tuple: (search_history, faulty_operator_index)
        - search_history: list of (split_point, is_fault) tuples
        - faulty_operator_index: index of the first faulty operator, or -1 if no fault found
    """
    search_history = []
    faulty_operator_index = -1  # Initialize as -1 meaning no fault found

    # Initialize boundaries.
    # 'high' usually represents the total number of operators in the graph.
    low = 0
    if hasattr(truncator, "get_total_steps"):
        high = truncator.get_total_steps(relative_model_path)
    else:
        high = getattr(truncator, "total_steps", 1024)

    truncate_pos = high

    while True:
        # Extract subgraph: generates relative_model_path for prefix [0:truncate_pos]
        sub_model_path = truncator(relative_model_path, truncate_pos)

        # Evaluation: runs the sub-graph to get Error Score (ES)
        es_scores = es_scores_calculator(evaluator(sub_model_path))

        # Predication: checks if current ES is within acceptable tolerance
        is_fault = predicator(es_scores, tolerance)

        # Log results
        current_step = (truncate_pos, is_fault)
        search_history.append(current_step)

        # Termination check
        if stoper(search_history, high=high):
            break

        # Interval update
        if is_fault:
            # Fault detected in current prefix; search earlier for the root cause
            high = truncate_pos
        else:
            # Current prefix is healthy; the first fault must be in the suffix
            low = truncate_pos + 1

        # Determine current split point
        truncate_pos = (low + high) // 2

        # Safety break for boundary convergence
        if low >= high:
            # Ensure the final point is captured if the loop terminates via boundary
            if not any(h[0] == low for h in search_history):
                truncated_model_path = truncator(relative_model_path, low)
                final_es = es_scores_calculator(evaluator(truncated_model_path))
                final_is_fault = predicator(final_es, tolerance)
                search_history.append((low, final_is_fault))

                if final_is_fault:
                    faulty_operator_index = low
            break

    faulty_positions = [pos for pos, is_fault in search_history if is_fault]
    if faulty_positions:
        faulty_operator_index = min(faulty_positions)
        faulty_model_path = truncator(relative_model_path, faulty_operator_index)
    else:
        faulty_operator_index = -1
        faulty_model_path = ""

    return search_history, faulty_operator_index, faulty_model_path
