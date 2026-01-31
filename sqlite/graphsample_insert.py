import sqlite3
import json
import argparse
from pathlib import Path
from datetime import datetime
import uuid as uuid_lib
import re
from orm_models import (
    get_session,
    GraphSample,
    SubgraphSource,
    DimensionGeneralizationSource,
    DataTypeGeneralizationSource,
)
from sqlalchemy.exc import IntegrityError


# graph_sample insert func
def get_graph_sample_data(
    model_path_prefix: str,
    relative_model_path: str,
    repo_uid: str,
    sample_type: str,
    order_value: int,
) -> dict:
    model_path = Path(model_path_prefix) / relative_model_path
    data = {
        "uuid": _get_uuid(),
        "repo_uid": repo_uid,
        "relative_model_path": relative_model_path,
        "sample_type": sample_type,
        "is_subgraph": _is_subgraph(sample_type),
        "num_ops": _get_num_ops(model_path, sample_type),
        "graph_hash": _get_graph_hash(model_path),
        "order_value": order_value,
        "create_at": _get_create_at(),
        "deleted": False,
        "delete_at": None,
    }
    return data


def insert_graph_sample(db_path: str, data: dict, model_path_prefix: str):
    session = get_session(db_path)
    try:
        graph_sample = GraphSample(**data)
        session.add(graph_sample)
        session.commit()
        return graph_sample
    except IntegrityError as e:
        session.rollback()
        raise e
    finally:
        session.close()


# subgraph source insert func
def insert_subgraph_source(
    subgraph_uuid: str, model_path_prefix: str, relative_model_path: str, db_path: str
):
    session = get_session(db_path)
    try:
        parent_relative_path = get_parent_relative_path(relative_model_path)
        full_graph = (
            session.query(GraphSample)
            .filter(
                GraphSample.relative_model_path == parent_relative_path,
                GraphSample.sample_type == "full_graph",
            )
            .first()
        )

        if not full_graph:
            raise ValueError(f"Full graph not found for path: {parent_relative_path}")

        range_info = _get_range_info(model_path_prefix, relative_model_path)
        subgraph_source = SubgraphSource(
            subgraph_uuid=subgraph_uuid,
            full_graph_uuid=full_graph.uuid,
            range_start=range_info["start"],
            range_end=range_info["end"],
            create_at=datetime.now(),
            deleted=False,
            delete_at=None,
        )
        session.add(subgraph_source)
        session.commit()

        return {
            "subgraph_uuid": subgraph_source.subgraph_uuid,
            "full_graph_uuid": subgraph_source.full_graph_uuid,
            "range_start": subgraph_source.range_start,
            "range_end": subgraph_source.range_end,
        }
    except IntegrityError as e:
        session.rollback()
        raise e
    finally:
        session.close()


def _get_range_info(model_path_prefix: str, relative_model_path: str):
    model_path = Path(model_path_prefix) / relative_model_path
    subgraph_sources_file = model_path / "subgraph_sources.json"
    if not subgraph_sources_file.exists():
        return {"start": -1, "end": -1}

    try:
        with open(subgraph_sources_file) as f:
            data = json.load(f)
        for key, ranges in data.items():
            if isinstance(ranges, list):
                r = ranges[0]
                if isinstance(r, list) and len(r) == 2:
                    return {"start": r[0], "end": r[1]}
        return {"start": -1, "end": -1}
    except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
        print(f"Warning: Failed to parse {subgraph_sources_file}: {e}")
        return {"start": -1, "end": -1}


def get_parent_relative_path(relative_path: str) -> str:
    if "_decomposed" not in relative_path:
        return None

    parts = relative_path.split("/")
    if len(parts) < 2:
        return None

    parent_parts = []
    for part in parts:
        if part == "_decomposed":
            break
        parent_parts.append(part)

    return "/".join(parent_parts)


# full_graph insert func
def _get_uuid() -> str:
    return uuid_lib.uuid4().hex


def _is_subgraph(sample_type: str) -> bool:
    return sample_type not in ("full_graph")


def _get_num_ops(model_path: Path, sample_type: str):
    if sample_type == "full_graph":
        return -1
    subgraph_sources_file = model_path / "subgraph_sources.json"
    if not subgraph_sources_file.exists():
        return -1

    try:
        with open(subgraph_sources_file) as f:
            data = json.load(f)
        for key, ranges in data.items():
            if isinstance(ranges, list):
                r = ranges[0]
                if isinstance(r, list) and len(r) == 2:
                    return r[1] - r[0]

        return -1
    except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
        print(f"Warning: Failed to parse {subgraph_sources_file}: {e}")
        return -1


def _get_graph_hash(model_path: Path) -> str:
    hash_file = model_path / "graph_hash.txt"
    if hash_file.exists():
        return hash_file.read_text().strip()
    return ""


def _get_create_at() -> datetime:
    return datetime.now()


# DimensionGeneralizationSource insert func
def insert_dimension_generalization_source(
    generalized_graph_uuid: str,
    original_graph_uuid: str,
    model_path_prefix: str,
    relative_model_path: str,
    db_path: str,
):
    session = get_session(db_path)
    try:
        dimension_source = DimensionGeneralizationSource(
            generalized_graph_uuid=generalized_graph_uuid,
            original_graph_uuid=original_graph_uuid,
            total_element_size=_get_total_element_size(
                model_path_prefix, relative_model_path
            ),
            create_at=datetime.now(),
            deleted=False,
            delete_at=None,
        )
        session.add(dimension_source)
        session.commit()
    except IntegrityError as e:
        session.rollback()
        raise e
    finally:
        session.close()


def _get_total_element_size(model_path_prefix: str, relative_model_path: str):
    model_path = Path(model_path_prefix) / relative_model_path
    weight_meta_file = model_path / "weight_meta.py"
    try:
        with open(weight_meta_file) as f:
            content = f.read()

        shape_matches = re.findall(
            r"shape\s*=\s*\[([0-9,\s\.]+(?:\d+)?[^\]]+)\s*\]", content
        )
        total_element_size = 0
        for match in shape_matches:
            shape_str = match.strip()
            shape_element_size = 1
            numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", shape_str)
            for num_str in numbers:
                num = float(num_str) if "." in num_str else int(num_str)
                shape_element_size *= num

            total_element_size += shape_element_size

        return total_element_size
    except Exception as e:
        print(f"Warning: Failed to parse {weight_meta_file}: {e}")
        return -1


# DataTypeGeneralizationSource insert func
def insert_datatype_generalization_source(
    generalized_graph_uuid: str,
    original_graph_uuid: str,
    model_path_prefix: str,
    relative_model_path: str,
    db_path: str,
):
    session = get_session(db_path)
    try:
        data_type_source = DataTypeGeneralizationSource(
            generalized_graph_uuid=generalized_graph_uuid,
            original_graph_uuid=original_graph_uuid,
            data_type=_get_data_type(model_path_prefix, relative_model_path),
            create_at=datetime.now(),
            deleted=False,
            delete_at=None,
        )
        session.add(data_type_source)
        session.commit()
    except IntegrityError as e:
        session.rollback()
        raise e
    finally:
        session.close()


def _get_data_type(model_path_prefix: str, relative_model_path: str):
    return "todo"


# main func
def main(args):
    data = get_graph_sample_data(
        model_path_prefix=args.model_path_prefix,
        relative_model_path=args.relative_model_path,
        repo_uid=args.repo_uid,
        sample_type=args.sample_type,
        order_value=args.order_value,
    )
    print(f"\ninsert into database: {args.db_path}")
    try:
        insert_graph_sample(args.db_path, data, args.model_path_prefix)
        if data["is_subgraph"]:
            subgraph_source_data = insert_subgraph_source(
                data["uuid"],
                args.model_path_prefix,
                data["relative_model_path"],
                args.db_path,
            )
            if args.sample_type in ["fusible_graph"]:
                insert_dimension_generalization_source(
                    subgraph_source_data["subgraph_uuid"],
                    subgraph_source_data["full_graph_uuid"],
                    args.model_path_prefix,
                    args.relative_model_path,
                    args.db_path,
                )
                insert_datatype_generalization_source(
                    subgraph_source_data["subgraph_uuid"],
                    subgraph_source_data["full_graph_uuid"],
                    args.model_path_prefix,
                    args.relative_model_path,
                    args.db_path,
                )
        print(f"success insert: {data['relative_model_path']}")
    except sqlite3.IntegrityError as e:
        print("insert failed: integrity error (possible duplicate uuid or graph_hash)")
        print(f"error info: {e}")
    except Exception as e:
        print(f"insert failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="insert graph sample to database")
    parser.add_argument(
        "--model_path_prefix",
        type=str,
        required=True,
        default="GraphNet",
        help="Prefix of model path root'",
    )
    parser.add_argument(
        "--relative_model_path",
        type=str,
        required=True,
        help="Path to model folder e.g '../../samples/torch/resnet18'",
    )
    parser.add_argument(
        "--repo_uid",
        type=str,
        required=True,
        help="Repository uid e.g 'github torch samples', 'github_paddle_samples'",
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        required=True,
        default="full_graph",
        help="Sample type e.g 'full_graph', 'fusible_graph'",
    )
    parser.add_argument(
        "--order_value",
        type=int,
        required=True,
        help="Order value e.g '1'",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=False,
        default="graphnet.db",
        help="Database file path e.g 'graphnet.db'",
    )
    args = parser.parse_args()
    main(args)
