"""
GraphNet Sample Bucketing Generator

Viba definition:
    SampleBucketInfo :=
        Object
      * $sample_uid str
      * $op_seq_bucket_id list[str]
      * $input_shapes_bucket_id list[list[int]]
      * $input_dtypes_bucket_id list[str]

Output: intermediate artifact (sample_type_uids, bucket_info_map)
Downstream: ai4c_sample_generator.Ai4cSampleGenerator
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Tuple
from collections import defaultdict

from orm_models import (
    get_session,
    GraphNetSampleBucket,
    GraphSample,
    SampleOpNameList,
    SampleInputTensorMeta,
)


@dataclass
class SampleBucketInfo:
    """Bucket info for a single sample."""

    sample_uid: str = ""
    op_seq_bucket_id: List[str] = field(default_factory=list)
    input_shapes_bucket_id: List[List[int]] = field(default_factory=list)
    input_dtypes_bucket_id: List[str] = field(default_factory=list)

    def get_bucket_key(self) -> str:
        return ".".join(self.op_seq_bucket_id) if self.op_seq_bucket_id else "unknown"


@dataclass
class SampleTypeBucketResult:
    """Bucketing results grouped by sample_type."""

    sample_type: str
    is_sole_op: bool  # whether this is a sole-operator type
    sample_uids: List[str]
    bucket_info_map: Dict[str, SampleBucketInfo]

    def __len__(self):
        return len(self.sample_uids)


def get_default_dimension_bucket(dim: int) -> int:
    """Default bucket function: log2(dim) // 4"""
    return -1 if dim <= 0 else int(math.log2(dim)) // 4


def generate_sample_buckets(
    session,
    get_dimension_bucket: Callable[[int], int] = None,
) -> Tuple[List[SampleTypeBucketResult], Dict[str, SampleBucketInfo]]:
    """
    Generate bucketing info from GraphNet.db.

    Args:
        session: SQLAlchemy session
        get_dimension_bucket: bucket function, defaults to log2(dim) // 4

    Returns:
        sample_type_results: List[SampleTypeBucketResult], results grouped by sample_type
        all_bucket_info_map: Dict[sample_uid, SampleBucketInfo], bucket info for all samples
    """
    if get_dimension_bucket is None:
        get_dimension_bucket = get_default_dimension_bucket

    # Query samples (exclude full_graph, only process subgraphs)
    samples = (
        session.query(GraphSample)
        .filter(GraphSample.deleted.is_(False))
        .filter(GraphSample.sample_type != "full_graph")
        .all()
    )
    samples_map = {s.uuid: s.sample_type for s in samples}

    # Query operator sequences
    op_seqs = {}
    op_name_lists = (
        session.query(SampleOpNameList)
        .filter(SampleOpNameList.deleted.is_(False))
        .all()
    )
    for op_list in op_name_lists:
        op_data = json.loads(op_list.op_names_json)
        op_seqs[op_list.sample_uuid] = [op["op_name"] for op in op_data]

    # Query input tensor metadata
    input_metas = defaultdict(list)
    tensor_metas = (
        session.query(SampleInputTensorMeta)
        .filter(SampleInputTensorMeta.deleted.is_(False))
        .order_by(SampleInputTensorMeta.input_idx)
        .all()
    )
    for t in tensor_metas:
        dtype = t.dtype
        if dtype == "todo":
            dtype = "unknown"
        else:
            dtype = dtype.replace("torch.", "")

        shape_str = t.shape
        if shape_str == "todo" or shape_str == "[]":
            shape = []
        else:
            shape = json.loads(shape_str)

        input_metas[t.sample_uuid].append(
            {
                "shape": shape,
                "dtype": dtype,
            }
        )

    # Group by sample_type
    grouped_by_type = defaultdict(list)
    for uid, stype in samples_map.items():
        grouped_by_type[stype].append(uid)

    # Generate bucket info for each sample
    all_bucket_info_map = {}
    for uid in samples_map:
        shapes = input_metas.get(uid, [])
        all_bucket_info_map[uid] = SampleBucketInfo(
            sample_uid=uid,
            op_seq_bucket_id=op_seqs.get(uid, []),
            input_shapes_bucket_id=[
                [get_dimension_bucket(d) for d in m["shape"]] for m in shapes
            ],
            input_dtypes_bucket_id=[m["dtype"] for m in shapes],
        )

    # Build results grouped by sample_type
    sample_type_results = []
    for stype, uids in grouped_by_type.items():
        stype_bucket_map = {uid: all_bucket_info_map[uid] for uid in uids}
        sample_type_results.append(
            SampleTypeBucketResult(
                sample_type=stype,
                is_sole_op=(stype == "sole_op_graph"),
                sample_uids=uids,
                bucket_info_map=stype_bucket_map,
            )
        )

    return sample_type_results, all_bucket_info_map


def save_bucket_results(
    session,
    bucket_info_map: Dict[str, SampleBucketInfo],
) -> int:
    """
    Save bucket results to graph_net_sample_buckets table in the same DB.

    Uses INSERT OR REPLACE: each sample_uid keeps only the latest row.
    The deleted/delete_at fields are reserved for downstream business logic.

    Args:
        session: SQLAlchemy session
        bucket_info_map: Dict[sample_uid, SampleBucketInfo] from generate_sample_buckets()

    Returns:
        number of rows inserted
    """
    now = datetime.now()
    count = 0

    for uid, info in bucket_info_map.items():
        # Check if exists
        existing = (
            session.query(GraphNetSampleBucket)
            .filter(GraphNetSampleBucket.sample_uid == uid)
            .first()
        )

        if existing:
            existing.op_seq_bucket_id = json.dumps(
                info.op_seq_bucket_id, ensure_ascii=False
            )
            existing.input_shapes_bucket_id = json.dumps(info.input_shapes_bucket_id)
            existing.input_dtypes_bucket_id = json.dumps(info.input_dtypes_bucket_id)
            existing.deleted = False
            existing.delete_at = None
        else:
            bucket = GraphNetSampleBucket(
                sample_uid=uid,
                op_seq_bucket_id=json.dumps(info.op_seq_bucket_id, ensure_ascii=False),
                input_shapes_bucket_id=json.dumps(info.input_shapes_bucket_id),
                input_dtypes_bucket_id=json.dumps(info.input_dtypes_bucket_id),
                create_at=now,
                deleted=False,
            )
            session.add(bucket)
        count += 1

    session.commit()
    return count


def generate_buckets(db_path, dry_run=False):
    """Generate buckets and save to DB."""
    session = get_session(db_path)
    try:
        print("=" * 70)
        print("Step 1: Generating bucket info from graph_sample...")
        _, bucket_info_map = generate_sample_buckets(session)
        if dry_run:
            print("Dry run mode - skipping database insert")
            print(
                f"  Would insert {len(bucket_info_map)} records into graph_net_sample_buckets"
            )
            return 0

        print("Step 2: Saving to database...")
        count = save_bucket_results(session, bucket_info_map)
        print(f"  Inserted {count} records into graph_net_sample_buckets")
        return count
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate graph_net_sample_buckets from graph_sample"
    )
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    generate_buckets(args.db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
