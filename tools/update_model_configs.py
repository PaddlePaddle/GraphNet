import os
import json
import argparse
import re
from datetime import datetime
from huggingface_hub import list_models
from huggingface_hub.utils import HfHubHTTPError


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

    # 1. Suffix Check (fast and handles most cases)
    if normalized_hub.endswith(normalized_local):
        return True

    # 2. Component Set Check (handles reordered names like opus-mt-en-st)
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


def find_best_match(
    search_term: str, author: str = None, local_name: str = None
) -> dict | None:
    """
    Searches the Hub for the top 3 candidates and returns the first one that passes validation.
    """
    candidates = list(
        list_models(search=search_term, author=author, sort="downloads", limit=3)
    )
    for model_candidate in candidates:
        if validate_match(local_name=local_name, hub_name=model_candidate.id):
            return model_candidate  # Return the first valid match
    return None


def process_model_directories(root_dir: str):
    """
    Scans model subdirectories and updates their graph_net.json file using a
    multi-stage search and validation strategy.
    """
    if not root_dir or not os.path.isdir(root_dir):
        print(
            f"❌ ERROR: Root directory not provided or not found. Please use the --directory argument."
        )
        return

    failures = []  # A list to collect details of all failed directories

    try:
        subdirectories = [
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        ]
    except OSError as e:
        print(
            f"❌ ERROR: Cannot read directory '{root_dir}'. Please check permissions. Error: {e}"
        )
        return

    print(f"🚀 Found {len(subdirectories)} model directories. Starting process...")

    for subdir_name in subdirectories:
        print(f"\n{'='*20}\nProcessing: {subdir_name}")
        json_path = os.path.join(root_dir, subdir_name, "graph_net.json")

        if not os.path.exists(json_path):
            print(f"  [🟡 SKIPPING] 'graph_net.json' file not found.")
            failures.append(
                {"directory": subdir_name, "reason": "'graph_net.json' not found"}
            )
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            best_match = None

            # --- Phase 1: Precise Search (Author + Model Name) ---
            if "_" in subdir_name:
                parts = subdir_name.split("_", 1)
                author, model_search_term = parts[0], parts[1]
                print(f"  [Phase 1] Trying precise search with author: '{author}'")
                best_match = find_best_match(
                    search_term=model_search_term, author=author, local_name=subdir_name
                )

            # --- Phase 2: Broad Search (Fallback) ---
            if not best_match:
                print(f"  [Phase 2] Falling back to broad search for '{subdir_name}'")
                best_match = find_best_match(
                    search_term=subdir_name, local_name=subdir_name
                )

            # --- Process Final Result ---
            if best_match:
                full_model_id = best_match.id
                tags = best_match.tags or []
                print(
                    f"  [✅ SUCCESS] Verified Match: '{subdir_name}' -> '{full_model_id}'"
                )
                data["model_name"] = full_model_id
                data["source"] = "huggingface_hub"
                data["original_tag"] = tags
            else:
                print(
                    f"  [❌ FAILED] Could not find a valid match for '{subdir_name}' on Hugging Face Hub."
                )
                failures.append(
                    {
                        "directory": subdir_name,
                        "reason": "No valid match found after all search phases",
                    }
                )
                data["model_name"] = subdir_name
                data["source"] = "huggingface_hub"
                data["original_tag"] = []

            update_json_file(json_path, data)
            print(f"  [📝 UPDATED] Successfully updated '{json_path}'.")

        except json.JSONDecodeError as e:
            print(f"  [🔴 ERROR] File '{json_path}' is not a valid JSON. Skipping.")
            failures.append(
                {
                    "directory": subdir_name,
                    "reason": "Invalid JSON format",
                    "error_message": str(e),
                }
            )
        except HfHubHTTPError as e:
            print(f"  [🔴 ERROR] Network or API request failed.")
            failures.append(
                {
                    "directory": subdir_name,
                    "reason": "API Request Failed",
                    "error_message": str(e),
                }
            )
        except Exception as e:
            print(f"  [🔴 ERROR] An unexpected error occurred.")
            failures.append(
                {
                    "directory": subdir_name,
                    "reason": "Unexpected Script Error",
                    "error_message": str(e),
                }
            )

    print(f"\n{'='*40}\n🎉 All directories processed!")

    if failures:
        log_and_print_failures(failures)


def log_and_print_failures(failures: list):
    """
    Prints a summary of all failures to the console and saves it to a log file.
    """
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
        description="Scan a directory of models, find their info on Hugging Face Hub using a multi-stage strategy, and update their local graph_net.json file."
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="The root directory containing the model subdirectories to process.",
    )
    args = parser.parse_args()

    process_model_directories(args.directory)
