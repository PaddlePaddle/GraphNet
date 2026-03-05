import argparse
import sqlite3
import uuid as uuid_module
from datetime import datetime
from collections import namedtuple
from collections import defaultdict

from orm_models import (
    get_session,
    GraphNetSampleGroup,
)


GraphNetSampleUid = str
GraphNetSampleType = str
BucketId = str


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

    def exec(self, sql, params=None):
        self.cur.execute(sql, params or ())
        self.conn.commit()

    def close(self):
        self.conn.close()


SampleBucketInfo = namedtuple(
    "SampleBucketInfo",
    [
        "sample_uid",
        "op_seq_bucket_id",
        "input_shapes_bucket_id",
        "sample_type",
        "sample_uids",
    ],
)


def get_ai4c_group_members(sample_bucket_infos: list[SampleBucketInfo]):
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

    grouped = defaultdict(list)
    for bucket_info in sample_bucket_infos:
        grouped[bucket_info.op_seq_bucket_id].append(bucket_info.sample_uid)

    grouped = dict(grouped)
    for op_seq, sample_uids in grouped.items():
        new_uuid = str(uuid_module.uuid4())
        for sample_uid in sample_uids:
            yield sample_uid, new_uuid


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

    args = parser.parse_args()
    db = DB(args.db_path)
    db.connect()

    query_str = """
SELECT b.sample_uid, b.op_seq_bucket_id as op_seq, b.input_shapes_bucket_id, b.sample_type, group_concat(b.sample_uid, ',') as sample_uids
FROM (
    SELECT
    s.uuid AS sample_uid,
    s.sample_type AS sample_type,
    b.op_seq_bucket_id AS op_seq_bucket_id,
    b.input_shapes_bucket_id AS input_shapes_bucket_id
    FROM graph_sample s
    JOIN graph_net_sample_buckets b
        ON s.uuid = b.sample_uid
    order by s.create_at asc, s.uuid asc
) b
GROUP BY b.sample_type, b.op_seq_bucket_id, b.input_shapes_bucket_id;
    """

    query_results = db.query(query_str)
    print("Output:", len(query_results))

    query_results = [SampleBucketInfo(row) for row in query_results]

    session = get_session(args.db_path)

    try:
        for sample_uid, group_uid in get_ai4c_group_members(query_results):
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
        session.commit()

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
