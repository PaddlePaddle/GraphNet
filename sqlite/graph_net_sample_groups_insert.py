import argparse
import sqlite3
import uuid as uuid_module
from datetime import datetime
from collections import namedtuple, defaultdict

from orm_models import (
    get_session,
    GraphNetSampleGroup,
)


class DB:
    def __init__(self, path):
        self.path = path

    def connect(self):
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.cur = self.conn.cursor()

    def query(self, sql, params=None):
        self.cur.execute(sql, params or ())
        return self.cur.fetchall()

    def close(self):
        self.conn.close()


# ── V1 types ──

SampleBucketInfo = namedtuple(
    "SampleBucketInfo",
    [
        "sample_uid",
        "op_seq_bucket_id",
        "input_shapes_bucket_id",
        "input_dtypes_bucket_id",
        "sample_type",
        "sample_uids",
    ],
)


# ── V2 types ──

CandidateGraph = namedtuple(
    "CandidateGraph",
    [
        "sample_uid",
        "op_seq_bucket_id",
        "input_shapes_bucket_id",
        "input_dtypes_bucket_id",
        "graph_hash",
    ],
)


# ═══════════════════════════════════════════════════════════════════
# V1: Rule 1 (bucket-internal stride-5) + Rule 2 (cross-shape aggregation)
# ═══════════════════════════════════════════════════════════════════


def get_v1_group_members(sample_bucket_infos: list[SampleBucketInfo]):
    """Rule 1: stride-5 sampling within each bucket, each sample is its own group.
    Rule 2: cross-shape aggregation by (op_seq, dtype), heads share one group_uid."""

    # Rule 1
    for bucket_info in sample_bucket_infos:
        head_sample_uid = bucket_info.sample_uid
        sample_uids = bucket_info.sample_uids.split(",")
        selected_other_sample_uids = [
            other_sample_uid
            for other_sample_uid in sample_uids[::5]
            if other_sample_uid != head_sample_uid
        ]
        for sample_uid in selected_other_sample_uids:
            new_uuid = str(uuid_module.uuid4())
            yield sample_uid, new_uuid

    # Rule 2
    grouped = defaultdict(list)
    for bucket_info in sample_bucket_infos:
        key = (bucket_info.op_seq_bucket_id, bucket_info.input_dtypes_bucket_id)
        grouped[key].append(bucket_info.sample_uid)

    grouped = dict(grouped)
    for key, sample_uids in grouped.items():
        new_uuid = str(uuid_module.uuid4())
        for sample_uid in sample_uids:
            yield sample_uid, new_uuid


def query_v1_candidates(db: DB):
    query_str = """
SELECT b.sample_uid, b.op_seq_bucket_id as op_seq, b.input_shapes_bucket_id, b.input_dtypes_bucket_id, b.sample_type, group_concat(b.sample_uid, ',') as sample_uids
FROM (
    SELECT
    s.uuid AS sample_uid,
    s.sample_type AS sample_type,
    b.op_seq_bucket_id AS op_seq_bucket_id,
    b.input_shapes_bucket_id AS input_shapes_bucket_id,
    b.input_dtypes_bucket_id AS input_dtypes_bucket_id
    FROM graph_sample s
    JOIN graph_net_sample_buckets b
        ON s.uuid = b.sample_uid
    order by s.create_at asc, s.uuid asc
) b
GROUP BY b.sample_type, b.op_seq_bucket_id, b.input_shapes_bucket_id, b.input_dtypes_bucket_id;
    """
    rows = db.query(query_str)
    return [SampleBucketInfo(*row) for row in rows]


def insert_v1(db: DB, session, db_path: str):
    print("=" * 60)
    print("V1: Rule 1 (stride-5) + Rule 2 (cross-shape aggregation)")
    print("=" * 60)

    candidates = query_v1_candidates(db)
    print(f"  Bucket groups: {len(candidates)}")

    count = 0
    for sample_uid, group_uid in get_v1_group_members(candidates):
        new_group = GraphNetSampleGroup(
            sample_uid=sample_uid,
            group_uid=group_uid,
            group_type="ai4c",
            group_policy="bucket_policy_v1",
            policy_version="1.0",
            create_at=datetime.now(),
            deleted=False,
        )
        session.add(new_group)
        count += 1

    session.commit()
    print(f"  Inserted {count} v1 group records.")
    return count


# ═══════════════════════════════════════════════════════════════════
# V2: Rule 3 (global sparse sampling) + Rule 4 (dtype coverage)
# ═══════════════════════════════════════════════════════════════════


def get_v2_group_members(candidates: list[CandidateGraph], num_dtypes: int):
    """Rule 3: global sparse sampling, window_size=num_dtypes*5, pick first num_dtypes per window.
    Rule 4: dtype coverage, per (op_seq, shape) pick up to num_dtypes different-dtype samples.
    """

    # Index candidates by op_seq
    by_op_seq = defaultdict(list)
    for c in candidates:
        by_op_seq[c.op_seq_bucket_id].append(c)

    rule3_selected_uids = set()

    # --- Rule 3: global sparse sampling ---
    window_size = num_dtypes * 5
    for op_seq, op_candidates in by_op_seq.items():
        sorted_candidates = sorted(op_candidates, key=lambda c: c.sample_uid)

        rule3_uids = []
        for order_value, c in enumerate(sorted_candidates):
            if (order_value % window_size) < num_dtypes:
                rule3_uids.append(c.sample_uid)
                rule3_selected_uids.add(c.sample_uid)

        if rule3_uids:
            group_uid = str(uuid_module.uuid4())
            for uid in rule3_uids:
                yield uid, group_uid

    # --- Rule 4: dtype coverage ---
    for op_seq, op_candidates in by_op_seq.items():
        remaining = [
            c for c in op_candidates if c.sample_uid not in rule3_selected_uids
        ]

        # Sub-group by shape
        by_shape = defaultdict(list)
        for c in remaining:
            by_shape[c.input_shapes_bucket_id].append(c)

        rule4_uids = []
        for shape, shape_candidates in by_shape.items():
            seen_dtypes = set()
            for c in shape_candidates:
                if (
                    c.input_dtypes_bucket_id not in seen_dtypes
                    and len(seen_dtypes) < num_dtypes
                ):
                    seen_dtypes.add(c.input_dtypes_bucket_id)
                    rule4_uids.append(c.sample_uid)

        if rule4_uids:
            group_uid = str(uuid_module.uuid4())
            for uid in rule4_uids:
                yield uid, group_uid


def query_v2_candidates(db: DB) -> list[CandidateGraph]:
    query_str = """
SELECT
    sub.sample_uid,
    sub.op_seq_bucket_id,
    sub.input_shapes_bucket_id,
    sub.input_dtypes_bucket_id,
    sub.graph_hash
FROM (
    SELECT
        s.uuid AS sample_uid,
        b.op_seq_bucket_id,
        b.input_shapes_bucket_id,
        b.input_dtypes_bucket_id,
        s.graph_hash,
        ROW_NUMBER() OVER (
            PARTITION BY b.op_seq_bucket_id, b.input_shapes_bucket_id, b.input_dtypes_bucket_id, s.graph_hash
            ORDER BY s.create_at ASC, s.uuid ASC
        ) AS rn
    FROM graph_sample s
    JOIN graph_net_sample_buckets b ON s.uuid = b.sample_uid
    WHERE s.deleted = 0
      AND s.sample_type != 'full_graph'
) sub
WHERE sub.rn = 1
  AND sub.sample_uid NOT IN (
    SELECT g.sample_uid
    FROM graph_net_sample_groups g
    WHERE g.group_policy = 'bucket_policy_v1'
      AND g.deleted = 0
  )
ORDER BY sub.op_seq_bucket_id, sub.input_shapes_bucket_id, sub.input_dtypes_bucket_id, sub.sample_uid;
    """
    rows = db.query(query_str)
    return [CandidateGraph(*row) for row in rows]


def insert_v2(db: DB, session, db_path: str, num_dtypes: int):
    print("=" * 60)
    print("V2: Rule 3 (sparse sampling) + Rule 4 (dtype coverage)")
    print("=" * 60)

    candidates = query_v2_candidates(db)
    print(f"  V2 candidate graphs: {len(candidates)}")

    if not candidates:
        print("  No v2 candidates found. Skipping.")
        return 0

    count = 0
    for sample_uid, group_uid in get_v2_group_members(candidates, num_dtypes):
        new_group = GraphNetSampleGroup(
            sample_uid=sample_uid,
            group_uid=group_uid,
            group_type="ai4c",
            group_policy="bucket_policy_v2",
            policy_version="1.0",
            create_at=datetime.now(),
            deleted=False,
        )
        session.add(new_group)
        count += 1

    session.commit()
    print(f"  Inserted {count} v2 group records.")
    return count


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Generate graph_net_sample_groups (v1 + v2)"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--num_dtypes",
        type=int,
        default=3,
        help="Number of different dtypes to pick per shape in v2 (default: 3)",
    )
    parser.add_argument(
        "--only_v1",
        action="store_true",
        help="Only run v1 (Rule 1 + Rule 2)",
    )
    parser.add_argument(
        "--only_v2",
        action="store_true",
        help="Only run v2 (Rule 3 + Rule 4), requires v1 already in DB",
    )

    args = parser.parse_args()
    db = DB(args.db_path)
    db.connect()

    session = get_session(args.db_path)

    try:
        run_v1 = not args.only_v2
        run_v2 = not args.only_v1

        if run_v1:
            insert_v1(db, session, args.db_path)

        if run_v2:
            insert_v2(db, session, args.db_path, args.num_dtypes)

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        db.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
