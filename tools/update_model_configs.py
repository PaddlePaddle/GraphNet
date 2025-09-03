import os
import json
import argparse
from huggingface_hub import list_models
from huggingface_hub.utils import HfHubHTTPError


def process_model_directories(root_dir):
    """
    Scans all model subdirectories in a root path, searches for their tags on
    Hugging Face Hub, and updates their corresponding graph_net.json file.
    """
    if not root_dir or not os.path.isdir(root_dir):
        print(
            f"❌ ERROR: Root directory not provided or not found. Please use the --directory argument."
        )
        return

    # Get the names of all immediate subdirectories
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
            continue

        # Use the subdirectory name as the search term
        search_term = subdir_name

        try:
            # Pre-process the term: replace the first underscore with a slash
            # to accommodate the common 'user_model' format.
            corrected_term = search_term.replace("_", "/", 1)

            # Fuzzy search for the top-downloaded matching model
            found_models = list(
                list_models(search=corrected_term, sort="downloads", limit=1)
            )

            # Read the existing JSON file
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if found_models:
                best_match = found_models[0]
                full_model_id = best_match.id
                tags = best_match.tags or []  # Ensure tags is a list

                print(f"  [✅ SUCCESS] Matched: '{search_term}' -> '{full_model_id}'")

                # Update the JSON data dictionary with model name, source, and original tags
                data["model_name"] = full_model_id
                data["source"] = "huggingface_hub"
                data["original_tag"] = tags
            else:
                print(
                    f"  [❌ FAILED] Could not find a match for '{search_term}' on Hugging Face Hub."
                )

                # Also update the JSON to record the failure status
                data["model_name"] = "NOT_FOUND"
                data["source"] = "huggingface_hub"
                data["original_tag"] = []

            # Write the updated data back to the JSON file
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            print(f"  [📝 UPDATED] Successfully updated '{json_path}'.")

        except json.JSONDecodeError:
            print(f"  [🔴 ERROR] File '{json_path}' is not a valid JSON. Skipping.")
        except HfHubHTTPError as e:
            print(f"  [🔴 ERROR] Network or API request failed: {e}")
        except Exception as e:
            print(f"  [🔴 ERROR] An unexpected error occurred during processing: {e}")

    print(f"\n{'='*20}\n🎉 All directories processed!")


if __name__ == "__main__":
    # --- Setup command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Scan a directory of models, find their info on Hugging Face Hub, and update their local graph_net.json file."
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="The root directory containing the model subdirectories to process.",
    )
    args = parser.parse_args()

    # --- Run the main processing function with the provided directory ---
    process_model_directories(args.directory)
