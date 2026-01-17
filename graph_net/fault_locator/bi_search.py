def bi_search(
    model_path,
    truncator,  # Signature: (model_path, split_point) -> model_path
    evaluator,  # Signature: (model_path) -> ES
    predicator,  # Signature: (ES, tolerance) -> bool
    stoper,  # Signature: (history_list) -> bool
    tolerance=0,
):
    """
    Binary Search Algorithm for Automatic Fault Location.

    This algorithm locates the first faulty operation in a computational graph
    by iteratively narrowing the search range through graph truncation and
    backend execution comparison.

    Args:
        model_path (str): Path to the original computational graph.
        truncator (callable): Logic to slice the model at a specific split point.
        evaluator (callable): Logic to calculate Error Score (ES) for a given model.
        predicator (callable): Logic to determine fault status from ES and tolerance.
        stoper (callable): Logic to evaluate termination based on search history.
        tolerance (int): Numerical threshold for fault detection.

    Returns:
        list: Search history as a list of (split_point, is_fault) tuples.
    """
    search_history = []

    # Initialize boundaries.
    # 'high' usually represents the total number of operators in the graph.
    low = 0
    high = getattr(truncator, "total_steps", 1024)

    while True:
        # 1. Determine current split point
        mid = (low + high) // 2

        # 2. Extract subgraph: generates model_path for prefix [0:mid]
        sub_model_path = truncator(model_path, mid)

        # 3. Evaluation: runs the sub-graph to get Error Score (ES)
        es_scores = evaluator(sub_model_path)

        # 4. Predication: checks if current ES is within acceptable tolerance
        is_fault = predicator(es_scores, tolerance)

        # 5. Log results
        current_step = (mid, is_fault)
        search_history.append(current_step)

        # 6. Termination check
        if stoper(search_history):
            break

        # 7. Interval update
        if is_fault:
            # Fault detected in current prefix; search earlier for the root cause
            high = mid
        else:
            # Current prefix is healthy; the first fault must be in the suffix
            low = mid + 1

        # Safety break for boundary convergence
        if low >= high:
            # Ensure the final point is captured if the loop terminates via boundary
            if not any(h[0] == low for h in search_history):
                final_es = evaluator(truncator(model_path, low))
                search_history.append((low, predicator(final_es, tolerance)))
            break

    return search_history
