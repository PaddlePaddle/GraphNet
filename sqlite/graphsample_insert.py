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
    SampleOpName,
    SampleOpNameList,
    SampleInputTensorMeta,
)
from sqlalchemy import delete as sql_delete
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
    subgraph_uuid: str,
    model_path_prefix: str,
    sample_type: str,
    relative_model_path: str,
    db_path: str,
):
    session = get_session(db_path)
    try:
        parent_relative_path = get_parent_relative_path(relative_model_path)
        if sample_type == "fusible_graph" or sample_type == "typical_graph":
            parent_parts = parent_relative_path.split("/")
            parent_parts = parent_parts[2:]
            parent_relative_path = "/".join(parent_parts)
        if sample_type == "sole_op_graph":
            parent_parts = parent_relative_path.split("/")
            parent_parts = parent_parts[1:]
            parent_relative_path = "/".join(parent_parts)

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

        range_info = _get_parent_key_and_range(model_path_prefix, relative_model_path)
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
    return relative_model_path.split("/")[0]


# SampleOpNameList and SampleOpName insert func
def _get_parent_key_and_range(model_path_prefix: str, relative_model_path: str) -> dict:
    model_path = Path(model_path_prefix) / relative_model_path
    subgraph_sources_file = model_path / "subgraph_sources.json"
    if not subgraph_sources_file.exists():
        return {"parent_key": "", "start": -1, "end": -1}

    try:
        with open(subgraph_sources_file) as f:
            data = json.load(f)
        for key, ranges in data.items():
            if isinstance(ranges, list) and len(ranges) > 0:
                r = ranges[0]
                if isinstance(r, list) and len(r) == 2:
                    return {"parent_key": key, "start": r[0], "end": r[1]}
        return {"parent_key": "", "start": -1, "end": -1}
    except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
        print(f"Warning: Failed to parse {subgraph_sources_file}: {e}")
        return {"parent_key": "", "start": -1, "end": -1}


def insert_sample_op_name_list(
    sample_uuid: str,
    model_path_prefix: str,
    op_names_path_prefix: str,
    relative_model_path: str,
    db_path: str,
):
    if not op_names_path_prefix:
        print("op_names_path_prefix not provided, skipping insert_sample_op_name_list")
        return

    range_info = _get_parent_key_and_range(model_path_prefix, relative_model_path)
    parent_key = range_info["parent_key"]
    start = range_info["start"]
    end = range_info["end"]

    if start == -1 or end == -1 or not parent_key:
        print(
            f"Invalid range info for {relative_model_path}, skipping insert_sample_op_name_list"
        )
        return

    op_size = end - start
    op_names_file = Path(op_names_path_prefix) / parent_key / "op_names.txt"
    if not op_names_file.exists():
        print(
            f"op_names.txt not found at {op_names_file}, skipping insert_sample_op_name_list"
        )
        return

    try:
        with open(op_names_file) as f:
            all_op_names = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Warning: Failed to read {op_names_file}: {e}")
        return

    op_start = start
    op_end = end
    if op_end > len(all_op_names):
        print(f"Warning: op_end {op_end} exceeds total ops {len(all_op_names)}")
        op_end = len(all_op_names)
    if op_start >= op_end:
        print(f"Warning: op_start {op_start} >= op_end {op_end}")
        return

    selected_op_names = all_op_names[op_start:op_end]
    op_names_json = json.dumps(
        [{"op_name": name, "op_idx": i} for i, name in enumerate(selected_op_names)]
    )
    session = get_session(db_path)
    try:
        session.execute(
            sql_delete(SampleOpNameList).where(
                SampleOpNameList.sample_uuid == sample_uuid
            )
        )
        session.execute(
            sql_delete(SampleOpName).where(SampleOpName.sample_uuid == sample_uuid)
        )
        sample_op_name_list = SampleOpNameList(
            sample_uuid=sample_uuid,
            op_names_json=op_names_json,
            create_at=datetime.now(),
            deleted=False,
            delete_at=None,
        )
        session.add(sample_op_name_list)

        for idx, op_name in enumerate(selected_op_names):
            sample_op_name = SampleOpName(
                sample_uuid=sample_uuid,
                op_name=op_name,
                op_idx=idx,
                op_size=op_size,
                create_at=datetime.now(),
                deleted=False,
                delete_at=None,
            )
            session.add(sample_op_name)

        session.commit()
        print(
            f"Inserted {len(selected_op_names)} op_names for sample_uuid={sample_uuid}"
        )
    except IntegrityError as e:
        session.rollback()
        raise e
    finally:
        session.close()


# SampleInputTensorMeta insert func
def insert_sample_input_tensor_meta(
    sample_uuid: str,
    model_path_prefix: str,
    relative_model_path: str,
    db_path: str,
):
    from graph_net.tensor_meta import TensorMeta

    model_path = Path(model_path_prefix) / relative_model_path
    weight_meta_file = model_path / "weight_meta.py"

    try:
        tensor_metas = TensorMeta.unserialize_from_py_file(str(weight_meta_file))
    except Exception as e:
        print(f"Warning: Failed to parse {weight_meta_file}: {e}")
        return

    input_tensor_metas = []
    for input_idx, tensor_meta in enumerate(tensor_metas):
        input_tensor_metas.append(
            {
                "input_name": tensor_meta.original_name or tensor_meta.name,
                "input_idx": input_idx,
                "shape": str(tensor_meta.shape),
                "dtype": tensor_meta.dtype,
            }
        )

    if not input_tensor_metas:
        print(f"No tensor meta found in {weight_meta_file}")
        return

    session = get_session(db_path)
    try:
        session.execute(
            sql_delete(SampleInputTensorMeta).where(
                SampleInputTensorMeta.sample_uuid == sample_uuid
            )
        )
        for meta in input_tensor_metas:
            sample_input_tensor_meta = SampleInputTensorMeta(
                sample_uuid=sample_uuid,
                input_name=meta["input_name"],
                input_idx=meta["input_idx"],
                shape=meta["shape"],
                dtype=meta["dtype"],
                create_at=datetime.now(),
                deleted=False,
                delete_at=None,
            )
            session.add(sample_input_tensor_meta)

        session.commit()
        print(
            f"Inserted {len(input_tensor_metas)} input tensor meta(s) for sample_uuid={sample_uuid}"
        )
    except IntegrityError as e:
        session.rollback()
        print(f"Error inserting input tensor meta: {e}")
        raise e
    finally:
        session.close()


# main func
def insert_one_sample(
    model_path_prefix: str,
    relative_model_path: str,
    repo_uid: str,
    sample_type: str,
    order_value: int,
    db_path: str,
    op_names_path_prefix: str = "",
):
    model_path_prefix = model_path_prefix.strip()
    relative_model_path = relative_model_path.strip()
    repo_uid = repo_uid.strip()
    sample_type = sample_type.strip()
    db_path = db_path.strip()
    op_names_path_prefix = op_names_path_prefix.strip() if op_names_path_prefix else ""

    data = get_graph_sample_data(
        model_path_prefix=model_path_prefix,
        relative_model_path=relative_model_path,
        repo_uid=repo_uid,
        sample_type=sample_type,
        order_value=order_value,
    )
    print(f"\ninsert into database: {db_path=}, {sample_type=}")
    try:
        insert_graph_sample(db_path, data, model_path_prefix)
        if data["is_subgraph"]:
            subgraph_source_data = insert_subgraph_source(
                subgraph_uuid=data["uuid"],
                model_path_prefix=model_path_prefix,
                sample_type=sample_type,
                relative_model_path=relative_model_path,
                db_path=db_path,
            )
            insert_sample_op_name_list(
                sample_uuid=data["uuid"],
                model_path_prefix=model_path_prefix,
                op_names_path_prefix=op_names_path_prefix,
                relative_model_path=relative_model_path,
                db_path=db_path,
            )
            insert_sample_input_tensor_meta(
                sample_uuid=data["uuid"],
                model_path_prefix=model_path_prefix,
                relative_model_path=relative_model_path,
                db_path=db_path,
            )
            if sample_type in ["fusible_graph", "typical_graph"]:
                insert_dimension_generalization_source(
                    subgraph_source_data["subgraph_uuid"],
                    subgraph_source_data["full_graph_uuid"],
                    model_path_prefix,
                    relative_model_path,
                    db_path,
                )
            insert_datatype_generalization_source(
                subgraph_source_data["subgraph_uuid"],
                subgraph_source_data["full_graph_uuid"],
                model_path_prefix,
                relative_model_path,
                db_path,
            )
        print(f"insert {sample_type} success: {data['relative_model_path']}")
    except sqlite3.IntegrityError as e:
        print(
            "insert {sample_type} failed: integrity error (possible duplicate uuid or graph_hash)"
        )
        print(f"error info: {e}")
        return False
    except Exception as e:
        print(f"insert {sample_type} failed: {e}")
        return False

    return True


def main(args):
    insert_one_sample(
        model_path_prefix=args.model_path_prefix,
        relative_model_path=args.relative_model_path,
        repo_uid=args.repo_uid,
        sample_type=args.sample_type,
        order_value=args.order_value,
        db_path=args.db_path,
        op_names_path_prefix=args.op_names_path_prefix,
    )


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
    parser.add_argument(
        "--op_names_path_prefix",
        type=str,
        required=False,
        help="Path prefix of op names file",
    )
    args = parser.parse_args()
    main(args)
