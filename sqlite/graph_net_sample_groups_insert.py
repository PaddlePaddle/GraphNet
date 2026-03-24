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

BucketGroup = namedtuple(
    "BucketGroup",
    ["head_uid", "op_seq", "shapes", "sample_type", "all_uids_csv"],
)

# ── V2 types ──

V2Candidate = namedtuple(
    "V2Candidate",
    ["uid", "op_seq", "shapes", "dtypes"],
)


def _new_group_id():
    return str(uuid_module.uuid4())


def _print_stats(stats, rule_order):
    total_records = 0
    total_groups = 0
    for rule_name in rule_order:
        if rule_name in stats:
            r = stats[rule_name]["records"]
            g = len(stats[rule_name]["groups"])
            print(f"  {rule_name}: {r} records, {g} groups")
            total_records += r
            total_groups += g
    print(f"  Total: {total_records} records, {total_groups} groups.")


# ═══════════════════════════════════════════════════════════════════
# V1: Rule 1 (bucket-internal stride-5) + Rule 2 (cross-shape aggregation)
# ═══════════════════════════════════════════════════════════════════


def generate_v1_groups(bucket_groups: list[BucketGroup]):
    """Rule 1: stride-5 sampling, each sampled uid gets its own group.
    Rule 2: group all bucket heads that share the same op_seq.
    Yields (uid, group_id, rule_name)."""

    # Rule 1: stride-5 sampling
    for bucket in bucket_groups:
        members = bucket.all_uids_csv.split(",")
        sampled = [uid for uid in members[::5] if uid != bucket.head_uid]
        for uid in sampled:
            yield uid, _new_group_id(), "rule1"

    # Rule 2: group heads by op_seq
    op_seq_to_heads = defaultdict(list)
    for bucket in bucket_groups:
        op_seq_to_heads[bucket.op_seq].append(bucket.head_uid)

    for heads in op_seq_to_heads.values():
        group_id = _new_group_id()
        for uid in heads:
            yield uid, group_id, "rule2"


def query_v1_bucket_groups(db: DB) -> list[BucketGroup]:
    sql = """
SELECT
    sub.sample_uid,
    sub.op_seq_bucket_id,
    sub.input_shapes_bucket_id,
    sub.sample_type,
    group_concat(sub.sample_uid, ',') AS all_uids
FROM (
    SELECT
        s.uuid AS sample_uid,
        s.sample_type,
        b.op_seq_bucket_id,
        b.input_shapes_bucket_id
    FROM graph_sample s
    JOIN graph_net_sample_buckets b ON s.uuid = b.sample_uid
    ORDER BY s.create_at ASC, s.uuid ASC
) sub
GROUP BY sub.sample_type, sub.op_seq_bucket_id, sub.input_shapes_bucket_id;
    """
    return [BucketGroup(*row) for row in db.query(sql)]


def insert_v1_groups(db: DB, session):
    print("=" * 60)
    print("V1: Rule 1 (stride-5) + Rule 2 (cross-shape aggregation)")
    print("=" * 60)

    bucket_groups = query_v1_bucket_groups(db)
    print(f"  Bucket groups: {len(bucket_groups)}")

    stats = defaultdict(lambda: {"records": 0, "groups": set()})
    for uid, group_id, rule_name in generate_v1_groups(bucket_groups):
        session.add(
            GraphNetSampleGroup(
                sample_uid=uid,
                group_uid=group_id,
                group_type="ai4c",
                group_policy="bucket_policy_v1",
                policy_version="1.0",
                create_at=datetime.now(),
                deleted=False,
            )
        )
        stats[rule_name]["records"] += 1
        stats[rule_name]["groups"].add(group_id)

    session.commit()
    _print_stats(stats, ["rule1", "rule2"])


# ═══════════════════════════════════════════════════════════════════
# V2: Rule 4 (dtype coverage) + Rule 3 (global sparse sampling)
# ═══════════════════════════════════════════════════════════════════


def generate_v2_groups(candidates: list[V2Candidate], num_dtypes: int):
    """Rule 4 runs first to ensure full dtype coverage.
    Rule 3 then sparse-samples from the remaining candidates.
    Yields (uid, group_id, rule_name)."""

    candidates_by_op_seq = defaultdict(list)
    for c in candidates:
        candidates_by_op_seq[c.op_seq].append(c)

    dtype_covered_uids = set()

    # --- Rule 4: dtype coverage (runs first) ---
    for op_seq, op_candidates in candidates_by_op_seq.items():
        candidates_by_shape = defaultdict(list)
        for c in op_candidates:
            candidates_by_shape[c.shapes].append(c)

        selected_uids = []
        for shape, shape_candidates in candidates_by_shape.items():
            seen_dtypes = set()
            for c in shape_candidates:
                if c.dtypes not in seen_dtypes and len(seen_dtypes) < num_dtypes:
                    seen_dtypes.add(c.dtypes)
                    selected_uids.append(c.uid)
                    dtype_covered_uids.add(c.uid)

        if selected_uids:
            group_id = _new_group_id()
            for uid in selected_uids:
                yield uid, group_id, "rule4"

    # --- Rule 3: global sparse sampling (on remaining candidates) ---
    window_size = num_dtypes * 5
    for op_seq, op_candidates in candidates_by_op_seq.items():
        remaining = [c for c in op_candidates if c.uid not in dtype_covered_uids]
        remaining.sort(key=lambda c: c.uid)

        selected_uids = []
        for idx, c in enumerate(remaining):
            if (idx % window_size) < num_dtypes:
                selected_uids.append(c.uid)

        if selected_uids:
            group_id = _new_group_id()
            for uid in selected_uids:
                yield uid, group_id, "rule3"


def query_v2_candidates(db: DB) -> list[V2Candidate]:
    sql = """
SELECT
    s.uuid,
    b.op_seq_bucket_id,
    b.input_shapes_bucket_id,
    b.input_dtypes_bucket_id
FROM graph_sample s
JOIN graph_net_sample_buckets b ON s.uuid = b.sample_uid
WHERE s.deleted = 0
  AND s.sample_type != 'full_graph'
  AND s.uuid NOT IN (
    SELECT g.sample_uid
    FROM graph_net_sample_groups g
    WHERE g.group_policy = 'bucket_policy_v1'
      AND g.deleted = 0
  )
ORDER BY b.op_seq_bucket_id, b.input_shapes_bucket_id, b.input_dtypes_bucket_id, s.uuid;
    """
    return [V2Candidate(*row) for row in db.query(sql)]


def insert_v2_groups(db: DB, session, num_dtypes: int):
    print("=" * 60)
    print("V2: Rule 4 (dtype coverage) + Rule 3 (sparse sampling)")
    print("=" * 60)

    candidates = query_v2_candidates(db)
    print(f"  V2 candidates: {len(candidates)}")

    if not candidates:
        print("  No v2 candidates found. Skipping.")
        return

    stats = defaultdict(lambda: {"records": 0, "groups": set()})
    for uid, group_id, rule_name in generate_v2_groups(candidates, num_dtypes):
        session.add(
            GraphNetSampleGroup(
                sample_uid=uid,
                group_uid=group_id,
                group_type="ai4c",
                group_policy="bucket_policy_v2",
                policy_version="1.0",
                create_at=datetime.now(),
                deleted=False,
            )
        )
        stats[rule_name]["records"] += 1
        stats[rule_name]["groups"].add(group_id)

    session.commit()
    _print_stats(stats, ["rule4", "rule3"])


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
    args = parser.parse_args()

    db = DB(args.db_path)
    db.connect()
    session = get_session(args.db_path)

    try:
        insert_v1_groups(db, session)
        insert_v2_groups(db, session, args.num_dtypes)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        db.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
