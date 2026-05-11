import argparse
import sqlite3
import uuid as uuid_module
from collections import defaultdict, namedtuple
from datetime import datetime

from orm_models import get_session, GraphNetSampleGroup


BucketGroup = namedtuple(
    "BucketGroup",
    ["head_uid", "op_seq", "shapes", "sample_type", "all_uids_csv"],
)

Candidate = namedtuple(
    "Candidate",
    ["uid", "sample_type", "op_seq", "shapes", "dtypes"],
)


def _new_group_id():
    return str(uuid_module.uuid4())


def _merge_stats(dst, src):
    for key, val in src.items():
        dst[key]["records"] += val["records"]
        dst[key]["groups"].update(val["groups"])


def _print_stats(stats):
    rule_order = ["rule1", "rule2", "rule4", "rule3"]
    sample_types = sorted({st for st, _ in stats})
    total_records = 0
    total_groups = 0
    for sample_type in sample_types:
        print(f"\n  [{sample_type}]")
        for rule in rule_order:
            key = (sample_type, rule)
            if key in stats:
                n_records = stats[key]["records"]
                n_groups = len(stats[key]["groups"])
                print(f"    {rule}: {n_records} records, {n_groups} groups")
                total_records += n_records
                total_groups += n_groups
    print(f"\n  Total: {total_records} records, {total_groups} groups.")


class DB:
    def __init__(self, path):
        self.path = path

    def connect(self):
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def query(self, sql, params=None):
        self.cursor.execute(sql, params or ())
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()


def query_bucket_groups(db: DB) -> list[BucketGroup]:
    sql = """
SELECT
    MIN(sub.sample_uid) AS head_uid,
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
    WHERE s.deleted = 0 AND s.sample_type != 'full_graph'
    ORDER BY s.create_at ASC, s.uuid ASC
) sub
GROUP BY sub.sample_type, sub.op_seq_bucket_id, sub.input_shapes_bucket_id;
    """
    return [BucketGroup(*row) for row in db.query(sql)]


def query_v2_candidates(db: DB) -> list[Candidate]:
    sql = """
SELECT
    s.uuid,
    s.sample_type,
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
ORDER BY s.sample_type, b.op_seq_bucket_id, b.input_shapes_bucket_id,
         b.input_dtypes_bucket_id, s.uuid;
    """
    return [Candidate(*row) for row in db.query(sql)]


# ═══════════════════════════════════════════════════════════════════
# V1: Rule 1 (bucket-internal stride sampling) + Rule 2 (cross-shape)
# ═══════════════════════════════════════════════════════════════════


def generate_v1_groups(bucket_groups: list[BucketGroup]):
    """Yields (sample_type, uid, group_id, rule_name).

    Rule 1: stride-16 sampling within each bucket, one group per sample.
    Rule 2: aggregate all bucket heads sharing the same (sample_type, op_seq).
    """
    # Rule 1
    for bucket in bucket_groups:
        members = bucket.all_uids_csv.split(",")
        for uid in members[::16]:
            if uid != bucket.head_uid:
                yield bucket.sample_type, uid, _new_group_id(), "rule1"

    # Rule 2
    heads_by_type_op = defaultdict(list)
    for bucket in bucket_groups:
        heads_by_type_op[(bucket.sample_type, bucket.op_seq)].append(bucket.head_uid)
    for (sample_type, _op), heads in heads_by_type_op.items():
        gid = _new_group_id()
        for uid in heads:
            yield sample_type, uid, gid, "rule2"


# ═══════════════════════════════════════════════════════════════════
# V2: Rule 4 (dtype coverage) + Rule 3 (sparse sampling on remainder)
# ═══════════════════════════════════════════════════════════════════


def generate_v2_groups(candidates: list[Candidate], num_dtypes: int):
    """Yields (sample_type, uid, group_id, rule_name).

    Rule 4 (first): per (sample_type, op_seq, shape), pick up to
                     num_dtypes samples with distinct dtypes.
    Rule 3 (second): window-based sparse sampling on the remainder,
                     window_size = num_dtypes * 5, pick first num_dtypes.
    """
    by_type_op = defaultdict(list)
    for c in candidates:
        by_type_op[(c.sample_type, c.op_seq)].append(c)

    covered_uids = set()

    # Rule 4: dtype coverage
    for (sample_type, _op), group in by_type_op.items():
        by_shape = defaultdict(list)
        for c in group:
            by_shape[c.shapes].append(c)

        picked = []
        for _shape, shape_group in by_shape.items():
            seen_dtypes = set()
            for c in shape_group:
                if c.dtypes not in seen_dtypes and len(seen_dtypes) < num_dtypes:
                    seen_dtypes.add(c.dtypes)
                    picked.append(c.uid)
                    covered_uids.add(c.uid)

        if picked:
            gid = _new_group_id()
            for uid in picked:
                yield sample_type, uid, gid, "rule4"

    # Rule 3: sparse sampling on remainder
    window_size = num_dtypes * 5
    for (sample_type, _op), group in by_type_op.items():
        remaining = sorted(
            (c for c in group if c.uid not in covered_uids),
            key=lambda c: c.uid,
        )
        picked = [
            c.uid for i, c in enumerate(remaining) if (i % window_size) < num_dtypes
        ]
        if picked:
            gid = _new_group_id()
            for uid in picked:
                yield sample_type, uid, gid, "rule3"


# ═══════════════════════════════════════════════════════════════════
# Insert
# ═══════════════════════════════════════════════════════════════════


def _insert_groups(session, rows, policy):
    """Consume a generator of (sample_type, uid, group_id, rule_name),
    write to DB, and return per-(sample_type, rule) stats."""
    stats = defaultdict(lambda: {"records": 0, "groups": set()})
    for sample_type, uid, group_id, rule_name in rows:
        session.add(
            GraphNetSampleGroup(
                sample_uid=uid,
                group_uid=group_id,
                group_type="ai4c",
                group_policy=policy,
                policy_version="1.0",
                create_at=datetime.now(),
                deleted=False,
            )
        )
        stats[(sample_type, rule_name)]["records"] += 1
        stats[(sample_type, rule_name)]["groups"].add(group_id)
    session.commit()
    return stats


def generate_groups(db_path, num_dtypes=3):
    """Generate sample groups and save to DB."""
    db = DB(db_path)
    db.connect()
    session = get_session(db_path)
    all_stats = defaultdict(lambda: {"records": 0, "groups": set()})
    try:
        buckets = query_bucket_groups(db)
        print(f"Bucket groups: {len(buckets)}")
        v1 = _insert_groups(session, generate_v1_groups(buckets), "bucket_policy_v1")
        _merge_stats(all_stats, v1)

        candidates = query_v2_candidates(db)
        print(f"V2 candidates: {len(candidates)}")
        if candidates:
            v2 = _insert_groups(
                session,
                generate_v2_groups(candidates, num_dtypes),
                "bucket_policy_v2",
            )
            _merge_stats(all_stats, v2)
        else:
            print("No V2 candidates found. Skipping.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        db.close()

    print("=" * 60)
    _print_stats(all_stats)
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate graph_net_sample_groups (v1 + v2)"
    )
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--num_dtypes", type=int, default=3)
    args = parser.parse_args()
    generate_groups(args.db_path, args.num_dtypes)


if __name__ == "__main__":
    main()
