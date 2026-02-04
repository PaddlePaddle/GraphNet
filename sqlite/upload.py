#!/usr/bin/env python3
import argparse
from orm_models import (
    get_session,
    Repo,
    GraphSample,
    SubgraphSource,
    DimensionGeneralizationSource,
    DataTypeGeneralizationSource,
    BackwardGraphSource,
)
from sqlalchemy.exc import IntegrityError
import os
from pathlib import Path
from huggingface_hub import HfApi

FULL_GRAPH_PATH = (
    "/work/clone/GraphNet/subgraph_dataset_workspace_small10_torch_samples/full_graph"
)
TYPICAL_GRAPH_PATH = "/work/clone/GraphNet/subgraph_dataset_workspace_small10_torch_samples/typical_graph"
FUSIBLE_GRAPH_PATH = "/work/clone/GraphNet/subgraph_dataset_workspace_small10_torch_samples/fusible_graph"
SOLE_OP_GRAPH_PATH = "/work/clone/GraphNet/subgraph_dataset_workspace_small10_torch_samples/sole_op_graph"

HF_REPO_ID = "PaddlePaddle/GraphNet"
HF_REPO_TYPE = "dataset"
HF_BRANCH = "main"
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")
HF_LARGE_THRESHOLD_MB = 1


def upload_to_huggingface(api: HfApi, local_path: str, repo_id: str, path_in_repo: str):
    if not os.path.exists(local_path):
        print(f"Warning: Local path not found, skipping upload: {local_path}")
        return

    print(f"Uploading {local_path} to HF: {repo_id}/{path_in_repo} ...")
    try:
        folder_size = sum(
            f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file()
        )
        size_mb = folder_size / (1024 * 1024)
        print(f"  Size: {size_mb:.2f}MB, uploading files individually...")
        for file_path in Path(local_path).rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(local_path)
                remote_path = f"{path_in_repo}/{rel_path}"
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type=HF_REPO_TYPE,
                    revision=HF_BRANCH,
                    commit_message=f"Upload {rel_path}",
                )
        print(f"Success: Uploaded {path_in_repo} ({size_mb:.2f}MB)")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")


def merge_databases(main_db_path: str, new_db_path: str):
    main_session = get_session(main_db_path)
    new_session = get_session(new_db_path)

    stats = {
        "repo": 0,
        "graph_sample": 0,
        "subgraph_source": 0,
        "dimension_generalization_source": 0,
        "datatype_generalization_source": 0,
        "backward_graph_source": 0,
    }

    try:
        existing_repo_uids = {
            r.repo_uid for r in main_session.query(Repo.repo_uid).all()
        }
        new_repos = (
            new_session.query(Repo)
            .filter(Repo.repo_uid.notin_(existing_repo_uids))
            .all()
        )
        for repo in new_repos:
            new_repo = Repo(
                repo_uid=repo.repo_uid,
                repo_type=repo.repo_type,
                repo_name=repo.repo_name,
                repo_url=repo.repo_url,
            )
            main_session.add(new_repo)
            stats["repo"] += 1
        main_session.commit()

        subgraph_map = {
            s.subgraph_uuid: s for s in new_session.query(SubgraphSource).all()
        }
        dim_gen_map = {
            s.generalized_graph_uuid: s
            for s in new_session.query(DimensionGeneralizationSource).all()
        }
        dtype_gen_map = {
            s.generalized_graph_uuid: s
            for s in new_session.query(DataTypeGeneralizationSource).all()
        }
        backward_map = {
            s.backward_graph_uuid: s
            for s in new_session.query(BackwardGraphSource).all()
        }

        existing_graph_uuids = {
            g.uuid for g in main_session.query(GraphSample.uuid).all()
        }
        existing_paths = {
            (g.relative_model_path, g.repo_uid)
            for g in main_session.query(
                GraphSample.relative_model_path, GraphSample.repo_uid
            ).all()
        }
        new_graph_samples = (
            new_session.query(GraphSample)
            .filter(GraphSample.uuid.notin_(existing_graph_uuids))
            .all()
        )
        new_graph_samples.sort(key=lambda x: 0 if x.sample_type == "full_graph" else 1)

        for sample in new_graph_samples:
            if (sample.relative_model_path, sample.repo_uid) in existing_paths:
                continue
            new_sample = GraphSample(
                uuid=sample.uuid,
                repo_uid=sample.repo_uid,
                relative_model_path=sample.relative_model_path,
                sample_type=sample.sample_type,
                is_subgraph=sample.is_subgraph,
                num_ops=sample.num_ops,
                graph_hash=sample.graph_hash,
                order_value=sample.order_value,
                create_at=sample.create_at,
                deleted=sample.deleted,
                delete_at=sample.delete_at,
            )
            main_session.add(new_sample)
            stats["graph_sample"] += 1

            if sample.uuid in subgraph_map:
                src = subgraph_map[sample.uuid]
                new_subgraph = SubgraphSource(
                    subgraph_uuid=src.subgraph_uuid,
                    full_graph_uuid=src.full_graph_uuid,
                    range_start=src.range_start,
                    range_end=src.range_end,
                    create_at=src.create_at,
                    deleted=src.deleted,
                    delete_at=src.delete_at,
                )
                main_session.add(new_subgraph)
                stats["subgraph_source"] += 1

            if sample.uuid in dim_gen_map:
                src = dim_gen_map[sample.uuid]
                new_dim = DimensionGeneralizationSource(
                    generalized_graph_uuid=src.generalized_graph_uuid,
                    original_graph_uuid=src.original_graph_uuid,
                    total_element_size=src.total_element_size,
                    create_at=src.create_at,
                    deleted=src.deleted,
                    delete_at=src.delete_at,
                )
                main_session.add(new_dim)
                stats["dimension_generalization_source"] += 1

            if sample.uuid in dtype_gen_map:
                src = dtype_gen_map[sample.uuid]
                new_dtype = DataTypeGeneralizationSource(
                    generalized_graph_uuid=src.generalized_graph_uuid,
                    original_graph_uuid=src.original_graph_uuid,
                    data_type=src.data_type,
                    create_at=src.create_at,
                    deleted=src.deleted,
                    delete_at=src.delete_at,
                )
                main_session.add(new_dtype)
                stats["datatype_generalization_source"] += 1

            if sample.uuid in backward_map:
                src = backward_map[sample.uuid]
                new_back = BackwardGraphSource(
                    backward_graph_uuid=src.backward_graph_uuid,
                    forward_graph_uuid=src.forward_graph_uuid,
                    create_at=src.create_at,
                    deleted=src.deleted,
                    delete_at=src.delete_at,
                )
                main_session.add(new_back)
                stats["backward_graph_source"] += 1

        main_session.commit()

        total_merged = sum(stats.values())
        print(
            f"Total merged: {total_merged} records from {new_db_path} into {main_db_path}"
        )
        print(f"  Breakdown: {stats}")

        return stats

    except IntegrityError as e:
        main_session.rollback()
        print(f"IntegrityError during merge: {e}")
        return stats
    except Exception as e:
        main_session.rollback()
        print(f"Error during merge: {e}")
        return stats
    finally:
        new_session.close()
        main_session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Merge new database into main GraphNet database"
    )
    parser.add_argument(
        "--new_db_path",
        type=str,
        required=True,
        help="Path to new database file (e.g., /path/to/new.db)",
    )
    parser.add_argument(
        "--main_db_path",
        type=str,
        required=True,
        help="Path to main GraphNet.db (e.g., /path/to/GraphNet.db)",
    )

    args = parser.parse_args()

    print("SREP 1: Merging databases...")
    stats = merge_databases(args.main_db_path, args.new_db_path)
    print("Merge Summary:")
    total_merged = sum(stats.values())
    print(f"  Total records merged: {total_merged}")
    for table, count in stats.items():
        print(f"  - {table}: {count}")

    print("STEP 2: Uploading samples to Hugging Face...")
    api = HfApi(token=HF_TOKEN)
    folders_to_upload = [
        (FULL_GRAPH_PATH, "full_graph"),
        (TYPICAL_GRAPH_PATH, "typical_graph"),
        (FUSIBLE_GRAPH_PATH, "fusible_graph"),
        (SOLE_OP_GRAPH_PATH, "sole_op_graph"),
    ]
    for local_p, remote_p in folders_to_upload:
        upload_to_huggingface(api, local_p, HF_REPO_ID, remote_p)

    print("STEP 3: Uploading database to Hugging Face...")
    # Upload main database to HF
    api.upload_file(
        path_or_fileobj=args.main_db_path,
        path_in_repo="GraphNet.db",
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        revision=HF_BRANCH,
        commit_message="Update GraphNet.db database",
    )
    print(f"Success: Uploaded database {args.main_db_path}")


if __name__ == "__main__":
    main()
