import os
import json
import argparse
import re
from datetime import datetime
from huggingface_hub import list_models, model_info
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError


def normalize_name(name: str) -> str:
    """
    Normalizes a name by replacing all non-alphanumeric characters with underscores.
    """
    return re.sub(r"[^a-zA-Z0-9]", "_", name)


def validate_match(local_name: str, hub_name: str) -> bool:
    """
    Validates if a found Hub model is a correct match using a two-step process.
    1. Suffix Check: Checks if the Hub name ends with the local name.
    2. Component Set Check: Checks if all parts of the local name exist in the Hub name (handles reordering).
    """
    normalized_local = normalize_name(local_name)
    normalized_hub = normalize_name(hub_name)

    if normalized_hub.endswith(normalized_local):
        return True

    local_components = set(normalized_local.split("_"))
    hub_components = set(normalized_hub.split("_"))
    if local_components.issubset(hub_components):
        return True

    return False


def update_json_file(path: str, data: dict):
    """
    Writes updated data to a JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def find_best_fuzzy_match(search_term: str, local_name: str) -> dict | None:
    """
    Searches the Hub for the top 3 candidates and returns the first one that passes validation.
    """
    # Phase 1: Try a precise fuzzy search if the name contains a potential author
    author, term = None, search_term
    if "/" in search_term:
        parts = search_term.split("/", 1)
        author, term = parts[0], parts[1]

    candidates = list(
        list_models(search=term, author=author, sort="downloads", limit=3)
    )
    for model_candidate in candidates:
        if validate_match(local_name=local_name, hub_name=model_candidate.id):
            return model_candidate

    # Phase 2: Fallback to a broader search if the precise one failed
    if author:
        candidates = list(list_models(search=search_term, sort="downloads", limit=3))
        for model_candidate in candidates:
            if validate_match(local_name=local_name, hub_name=model_candidate.id):
                return model_candidate

    return None


def process_single_model(model_dir_path: str, root_dir: str, failures: list):
    """
    Processes a single identified model directory.
    """
    json_path = os.path.join(model_dir_path, "graph_net.json")
    relative_path = os.path.relpath(model_dir_path, root_dir)
    potential_model_id = relative_path.replace(os.path.sep, "/")

    print(f"\n{'='*20}\nProcessing: {potential_model_id} (at {model_dir_path})")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        best_match_info = None

        # --- Stage 1: Precise Lookup ---
        # Phase 1a: Try the name derived directly from the path
        print(f"  [Phase 1a] Trying precise lookup for '{potential_model_id}'...")
        try:
            best_match_info = model_info(potential_model_id)
            print(f"  [Phase 1a] Success! Found exact match from path.")
        except RepositoryNotFoundError:
            print(f"  [Phase 1a] Precise lookup failed.")
            # Phase 1b: If that fails, and it's a single-level dir, try replacing the first '_' with '/'
            if "/" not in potential_model_id and "_" in potential_model_id:
                hypothetical_id = potential_model_id.replace("_", "/", 1)
                print(
                    f"  [Phase 1b] Trying alternative precise lookup for '{hypothetical_id}'..."
                )
                try:
                    best_match_info = model_info(hypothetical_id)
                    print(
                        f"  [Phase 1b] Success! Found exact match by replacing underscore."
                    )
                except RepositoryNotFoundError:
                    print(f"  [Phase 1b] Alternative lookup failed.")

        # --- Stage 2: Advanced Fuzzy Search (Fallback) ---
        if not best_match_info:
            print("  [Phase 2] Falling back to advanced fuzzy search...")
            best_match_info = find_best_fuzzy_match(
                search_term=potential_model_id, local_name=potential_model_id
            )

        # --- Process Final Result ---
        if best_match_info:
            full_model_id = best_match_info.id
            tags = best_match_info.tags or []
            print(
                f"  [✅ SUCCESS] Verified Match: '{potential_model_id}' -> '{full_model_id}'"
            )
            data["model_name"] = full_model_id
            data["source"] = "huggingface_hub"
            data["original_tag"] = tags
        else:
            print(
                f"  [❌ FAILED] Could not find a valid match for '{potential_model_id}' on Hugging Face Hub."
            )
            failures.append(
                {
                    "directory": model_dir_path,
                    "reason": "No valid match found after all search phases",
                }
            )
            data["model_name"] = "NO_VALID_MATCH_FOUND"
            data["source"] = "huggingface_hub"
            data["original_tag"] = []

        update_json_file(json_path, data)
        print(f"  [📝 UPDATED] Successfully updated '{json_path}'.")

    except json.JSONDecodeError as e:
        failures.append(
            {
                "directory": model_dir_path,
                "reason": "Invalid JSON format",
                "error_message": str(e),
            }
        )
    except HfHubHTTPError as e:
        failures.append(
            {
                "directory": model_dir_path,
                "reason": "API Request Failed",
                "error_message": str(e),
            }
        )
    except Exception as e:
        failures.append(
            {
                "directory": model_dir_path,
                "reason": "Unexpected Script Error",
                "error_message": str(e),
            }
        )


def process_model_directories(root_dir: str):
    """
    Scans for directories containing 'graph_net.json' at any level under the root path
    and processes them to find model information.
    """
    if not root_dir or not os.path.isdir(root_dir):
        print(
            f"❌ ERROR: Root directory not provided or not found. Please use the --directory argument."
        )
        return

    failures = []
    model_paths_to_process = []

    print(
        f"🔍 Scanning for model directories containing 'graph_net.json' under '{root_dir}'..."
    )
    for dirpath, _, filenames in os.walk(root_dir):
        if "graph_net.json" in filenames:
            model_paths_to_process.append(dirpath)

    if not model_paths_to_process:
        print("🤷 No directories containing 'graph_net.json' were found.")
        return

    print(
        f"🚀 Found {len(model_paths_to_process)} model directories. Starting process..."
    )

    for model_path in model_paths_to_process:
        process_single_model(model_path, root_dir, failures)

    print(f"\n{'='*40}\n🎉 All directories processed!")

    if failures:
        log_and_print_failures(failures)


def log_and_print_failures(failures: list):
    log_filename = "processing_failures.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"--- Processing Failure Report ({timestamp}) ---\n\n"
    summary += f"Total failures: {len(failures)}\n\n"
    for failure in failures:
        summary += f"Directory: {failure['directory']}\n"
        summary += f"Reason:    {failure['reason']}\n"
        if "error_message" in failure:
            summary += f"Details:   {failure['error_message']}\n"
        summary += "-" * 40 + "\n"
    print("\n\n" + summary)
    try:
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"A detailed failure report has been saved to '{log_filename}'")
    except IOError as e:
        print(f"❌ ERROR: Could not write failure report to log file. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan a directory (including subdirectories) for models, find their info on Hugging Face Hub, and update their local graph_net.json file."
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="The root directory to start scanning from.",
    )
    args = parser.parse_args()

    process_model_directories(args.directory)
