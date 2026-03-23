"""
bucket_policy_v2: Generate graph_net_sample_groups for v2 candidates.

v2 candidates = total_graph_buckets - v1_selected_graph_buckets

Graph bucket key: (op_seq_bucket_id, input_shapes_bucket_id, input_dtypes_bucket_id, graph_hash)

v2 grouping strategy (progressive, mutually exclusive with v1):
  Rule 3 (global sparse sampling): per op_seq, sort by sample_uid, every 5 pick 1.
  Rule 4 (dtype coverage): from Rule 3 remainder, per op_seq+shape, pick num_dtypes different-dtype samples.
"""

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


def query_v2_candidates(db: DB) -> list[CandidateGraph]:
    """
    Query v2 candidate graphs:
    total graph buckets - v1 selected graph buckets.

    Each (op_seq, shapes, dtypes, graph_hash) bucket picks one representative sample_uid
    (the earliest by create_at, uuid).

    v1 selected = sample_uids already in graph_net_sample_groups with group_policy='bucket_policy_v1'.
    """
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


def get_v2_group_members(candidates: list[CandidateGraph], num_dtypes: int):
    """
    Yield (sample_uid, group_uid) pairs for v2 grouping.

    Rule 3 (global sparse sampling):
      Sort all candidates by sample_uid, every 5 pick 1 sample.
      All picked samples under the same op_seq share one group_uid.

    Rule 4 (dtype coverage):
      From remaining candidates (not selected in Rule 3),
      per op_seq, for each shape, pick up to num_dtypes samples with different dtypes.
      All picked samples under the same op_seq share one group_uid.
    """
    # Index candidates by op_seq
    by_op_seq = defaultdict(list)
    for c in candidates:
        by_op_seq[c.op_seq_bucket_id].append(c)

    rule3_selected_uids = set()

    # --- Rule 3: global sparse sampling ---
    # Window size = num_dtypes * stride(5) = 15, pick first num_dtypes(3) per window
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
            # Pick up to num_dtypes samples with different dtypes
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate graph_net_sample_groups with bucket_policy_v2"
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
        help="Number of different dtypes to pick per shape (default: 3)",
    )

    args = parser.parse_args()
    db = DB(args.db_path)
    db.connect()

    print("Step 1: Querying v2 candidates...")
    candidates = query_v2_candidates(db)
    print(f"  v2 candidate graphs: {len(candidates)}")
    db.close()

    if not candidates:
        print("No v2 candidates found. Done!")
        return

    print(f"Step 2: Generating v2 groups (num_dtypes={args.num_dtypes})...")
    session = get_session(args.db_path)

    try:
        count = 0
        for sample_uid, group_uid in get_v2_group_members(candidates, args.num_dtypes):
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
        print(f"  Inserted {count} group records.")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print("Done!")


if __name__ == "__main__":
    main()
