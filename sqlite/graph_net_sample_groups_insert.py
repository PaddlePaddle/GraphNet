import argparse
import json
import uuid as uuid_module
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional
from datetime import datetime

from orm_models import (
    get_session,
    GraphNetSampleBucket,
    GraphNetSampleGroup,
)


GraphNetSampleUid = str
GraphNetSampleType = str
BucketId = str


@dataclass
class SampleBucketInfo:
    sample_uid: str = ""
    op_seq_bucket_id: List[str] = field(default_factory=list)
    input_shapes_bucket_id: List[List[int]] = field(default_factory=list)
    input_dtypes_bucket_id: List[str] = field(default_factory=list)

    def get_bucket_key(self) -> str:
        return ".".join(self.op_seq_bucket_id) if self.op_seq_bucket_id else "unknown"


@dataclass
class Ai4cSample:
    graph_net_sample_uids: List[GraphNetSampleUid] = field(default_factory=list)
    shape_diversity_min_bucket_size: int = 0
    shape_diversity_max_bucket_size: int = 0
    dtype_diversity_min_bucket_size: int = 0
    dtype_diversity_max_bucket_size: int = 0
    op_seq_bucket_id: List[str] = field(default_factory=list)
    group_type: str = ""

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.shape_diversity_min_bucket_size > self.shape_diversity_max_bucket_size:
            raise ValueError(
                f"shape_diversity_min_bucket_size ({self.shape_diversity_min_bucket_size}) "
                f"> shape_diversity_max_bucket_size ({self.shape_diversity_max_bucket_size})"
            )
        if self.dtype_diversity_min_bucket_size > self.dtype_diversity_max_bucket_size:
            raise ValueError(
                f"dtype_diversity_min_bucket_size ({self.dtype_diversity_min_bucket_size}) "
                f"> dtype_diversity_max_bucket_size ({self.dtype_diversity_max_bucket_size})"
            )


class Ai4cSampleGenerator:
    def __init__(
        self,
        shape_diversity_min: int = 1,
        shape_diversity_max: int = 2,
        dtype_diversity_min: int = 1,
        dtype_diversity_max: int = 2,
    ):
        self.shape_diversity_min = shape_diversity_min
        self.shape_diversity_max = shape_diversity_max
        self.dtype_diversity_min = dtype_diversity_min
        self.dtype_diversity_max = dtype_diversity_max

        # Inline function hooks (can be overridden)
        self.get_max_num_graph_net_samples: Optional[
            Callable[[Ai4cSample, BucketId], int]
        ] = None
        self.get_min_num_graph_net_samples: Optional[
            Callable[[Ai4cSample, BucketId], int]
        ] = None
        self._shape_diversity_sample_fn: Optional[
            Callable[[List[SampleBucketInfo], int, int], Ai4cSample]
        ] = None
        self._dtype_diversity_sample_fn: Optional[
            Callable[[List[SampleBucketInfo], int, int], Ai4cSample]
        ] = None

    def __call__(
        self,
        bucket_infos: List[SampleBucketInfo],
    ) -> List[Ai4cSample]:
        results: List[Ai4cSample] = []

        same_op_seq_buckets: Dict[str, List[SampleBucketInfo]] = {}

        for bucket_info in bucket_infos:
            bucket_key = bucket_info.get_bucket_key()
            if bucket_key not in same_op_seq_buckets:
                same_op_seq_buckets[bucket_key] = []
            same_op_seq_buckets[bucket_key].append(bucket_info)

        for bucket_key, bucket_info_list in same_op_seq_buckets.items():
            op_seq = bucket_info_list[0].op_seq_bucket_id if bucket_info_list else []
            temp_sample = Ai4cSample(op_seq_bucket_id=op_seq)
            max_samples = self._get_max_num_graph_net_samples(temp_sample, bucket_key)
            min_samples = self._get_min_num_graph_net_samples(temp_sample, bucket_key)

            shape_sample = self._shape_diversity_sample(
                bucket_info_list, max_samples, min_samples
            )
            self._validate_shape_diversity_sample(shape_sample)
            if shape_sample.graph_net_sample_uids:
                results.append(shape_sample)

            dtype_sample = self._dtype_diversity_sample(
                bucket_info_list, max_samples, min_samples
            )
            self._validate_dtype_diversity_sample(dtype_sample)
            if dtype_sample.graph_net_sample_uids:
                results.append(dtype_sample)

        return results

    def _get_max_num_graph_net_samples(
        self, sample: Ai4cSample, bucket_key: str
    ) -> int:
        if self.get_max_num_graph_net_samples:
            return self.get_max_num_graph_net_samples(sample, bucket_key)
        return 10

    def _get_min_num_graph_net_samples(
        self, sample: Ai4cSample, bucket_key: str
    ) -> int:
        if self.get_min_num_graph_net_samples:
            return self.get_min_num_graph_net_samples(sample, bucket_key)
        return 1

    def _shape_diversity_sample(
        self,
        bucket_infos: List[SampleBucketInfo],
        max_samples: int,
        min_samples: int,
    ) -> Ai4cSample:
        if self._shape_diversity_sample_fn:
            return self._shape_diversity_sample_fn(
                bucket_infos, max_samples, min_samples
            )

        shape_groups: Dict[str, List[GraphNetSampleUid]] = {}

        for bucket_info in bucket_infos[:max_samples]:
            shape_sig = str(bucket_info.input_shapes_bucket_id)
            if shape_sig not in shape_groups:
                shape_groups[shape_sig] = []
            shape_groups[shape_sig].append(bucket_info.sample_uid)

        filtered_uids: List[GraphNetSampleUid] = []
        bucket_sizes: List[int] = []

        for shape_sig, uids in shape_groups.items():
            if len(uids) < self.shape_diversity_min:
                continue

            truncated = uids[: self.shape_diversity_max]
            filtered_uids.extend(truncated)
            bucket_sizes.append(len(truncated))

        op_seq = bucket_infos[0].op_seq_bucket_id if bucket_infos else []

        return Ai4cSample(
            graph_net_sample_uids=filtered_uids,
            shape_diversity_min_bucket_size=min(bucket_sizes) if bucket_sizes else 0,
            shape_diversity_max_bucket_size=max(bucket_sizes) if bucket_sizes else 0,
            dtype_diversity_min_bucket_size=0,
            dtype_diversity_max_bucket_size=0,
            op_seq_bucket_id=op_seq,
            group_type="shape_diversity",
        )

    def _dtype_diversity_sample(
        self,
        bucket_infos: List[SampleBucketInfo],
        max_samples: int,
        min_samples: int,
    ) -> Ai4cSample:
        if self._dtype_diversity_sample_fn:
            return self._dtype_diversity_sample_fn(
                bucket_infos, max_samples, min_samples
            )

        dtype_groups: Dict[str, List[GraphNetSampleUid]] = {}

        for bucket_info in bucket_infos[:max_samples]:
            dtype_sig = str(bucket_info.input_dtypes_bucket_id)
            if dtype_sig not in dtype_groups:
                dtype_groups[dtype_sig] = []
            dtype_groups[dtype_sig].append(bucket_info.sample_uid)

        filtered_uids: List[GraphNetSampleUid] = []
        bucket_sizes: List[int] = []

        for dtype_sig, uids in dtype_groups.items():
            if len(uids) < self.dtype_diversity_min:
                continue

            truncated = uids[: self.dtype_diversity_max]
            filtered_uids.extend(truncated)
            bucket_sizes.append(len(truncated))

        op_seq = bucket_infos[0].op_seq_bucket_id if bucket_infos else []

        return Ai4cSample(
            graph_net_sample_uids=filtered_uids,
            shape_diversity_min_bucket_size=0,
            shape_diversity_max_bucket_size=0,
            dtype_diversity_min_bucket_size=min(bucket_sizes) if bucket_sizes else 0,
            dtype_diversity_max_bucket_size=max(bucket_sizes) if bucket_sizes else 0,
            op_seq_bucket_id=op_seq,
            group_type="dtype_diversity",
        )

    def _validate_shape_diversity_sample(self, sample: Ai4cSample) -> None:
        pass

    def _validate_dtype_diversity_sample(self, sample: Ai4cSample) -> None:
        pass


def load_bucket_infos_from_db(session) -> List[SampleBucketInfo]:
    bucket_infos = []

    db_buckets = (
        session.query(GraphNetSampleBucket)
        .filter(GraphNetSampleBucket.deleted.is_(False))
        .all()
    )

    for db_bucket in db_buckets:
        op_seq = (
            json.loads(db_bucket.op_seq_bucket_id) if db_bucket.op_seq_bucket_id else []
        )
        input_shapes = (
            json.loads(db_bucket.input_shapes_bucket_id)
            if db_bucket.input_shapes_bucket_id
            else []
        )
        input_dtypes = (
            json.loads(db_bucket.input_dtypes_bucket_id)
            if db_bucket.input_dtypes_bucket_id
            else []
        )

        bucket_info = SampleBucketInfo(
            sample_uid=db_bucket.sample_uid,
            op_seq_bucket_id=op_seq,
            input_shapes_bucket_id=input_shapes,
            input_dtypes_bucket_id=input_dtypes,
        )
        bucket_infos.append(bucket_info)

    return bucket_infos


def save_groups_to_db(
    session,
    ai4c_samples: List[Ai4cSample],
    policy_version: str = "v0.1",
) -> int:
    count = 0

    for sample in ai4c_samples:
        group_uid = str(uuid_module.uuid4())

        for sample_uid in sample.graph_net_sample_uids:
            group_record = GraphNetSampleGroup(
                sample_uid=sample_uid,
                group_uid=group_uid,
                group_type=sample.group_type,
                group_policy="by_bucket",
                policy_version=policy_version,
                create_at=datetime.now(),
                deleted=False,
            )
            session.add(group_record)
            count += 1

    session.commit()
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Generate graph_net_sample_groups from graph_net_sample_buckets"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--shape_diversity_min",
        type=int,
        default=2,
        help="Minimum shape diversity (default: 2)",
    )
    parser.add_argument(
        "--shape_diversity_max",
        type=int,
        default=3,
        help="Maximum shape diversity (default: 3)",
    )
    parser.add_argument(
        "--dtype_diversity_min",
        type=int,
        default=2,
        help="Minimum dtype diversity (default: 2)",
    )
    parser.add_argument(
        "--dtype_diversity_max",
        type=int,
        default=3,
        help="Maximum dtype diversity (default: 3)",
    )
    parser.add_argument(
        "--policy_version",
        type=str,
        default="v0.1",
        help="Policy version for the generated groups (default: v0.1)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be done, don't actually insert into database",
    )

    args = parser.parse_args()

    session = get_session(args.db_path)

    print("=" * 70)
    print("Step 1: Loading bucket info from database...")
    bucket_infos = load_bucket_infos_from_db(session)
    print(f"  Loaded {len(bucket_infos)} bucket records")

    print("=" * 70)
    print("Step 2: Generating Ai4cSample groups...")
    generator = Ai4cSampleGenerator(
        shape_diversity_min=args.shape_diversity_min,
        shape_diversity_max=args.shape_diversity_max,
        dtype_diversity_min=args.dtype_diversity_min,
        dtype_diversity_max=args.dtype_diversity_max,
    )
    ai4c_samples = generator(bucket_infos)
    print(f"  Generated {len(ai4c_samples)} Ai4cSample groups")

    print("=" * 70)
    print("Generated Ai4cSample groups:")
    for i, sample in enumerate(ai4c_samples):
        print(f"\n[{i}] op_seq: {sample.op_seq_bucket_id}")
        print(f"    group_type: {sample.group_type}")
        print(f"    sample_count: {len(sample.graph_net_sample_uids)}")
        print(
            f"    sample_uids: {sample.graph_net_sample_uids[:5]}{'...' if len(sample.graph_net_sample_uids) > 5 else ''}"
        )
        if sample.group_type == "shape_diversity":
            print(
                f"    shape_diversity: min={sample.shape_diversity_min_bucket_size}, max={sample.shape_diversity_max_bucket_size}"
            )
        else:
            print(
                f"    dtype_diversity: min={sample.dtype_diversity_min_bucket_size}, max={sample.dtype_diversity_max_bucket_size}"
            )

    print("=" * 70)
    if args.dry_run:
        print("Dry run mode - skipping database insert")
        total_records = sum(len(s.graph_net_sample_uids) for s in ai4c_samples)
        print(f"  Would insert {total_records} records into graph_net_sample_groups")
    else:
        print("Step 3: Saving to database...")
        count = save_groups_to_db(session, ai4c_samples, args.policy_version)
        print(f"  Inserted {count} records into graph_net_sample_groups")

    print("=" * 70)
    print("Done!")

    session.close()


if __name__ == "__main__":
    main()
