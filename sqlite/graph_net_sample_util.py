# graph_net_sample_util.py
import json
from typing import Dict, List

from orm_models import get_session, GraphSample, SampleOpNameList


class GraphNetSampleTypeGetter:
    """
    Get sample_type for a given sample_uid.

    Viba:
        GraphNetSampleTypeGetter :=
            # __call__
            $sample_type str
          <- $sample_uid str
            # __init__
          <- $group_net_db_file_path str
          <- $fetch_cache dict[$sample_uid str, $sample_type str]
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._cache: Dict[str, str] = {}

    def __call__(self, sample_uid: str) -> str:
        """Get sample_type for the given sample_uid."""
        if sample_uid in self._cache:
            return self._cache[sample_uid]

        session = get_session(self.db_path)
        sample = (
            session.query(GraphSample).filter(GraphSample.uuid == sample_uid).first()
        )
        session.close()

        sample_type = sample.sample_type if sample else ""
        self._cache[sample_uid] = sample_type
        return sample_type

    def bulk_get(self, sample_uids: List[str]) -> Dict[str, str]:
        """Bulk get sample_types for multiple sample UIDs."""
        session = get_session(self.db_path)

        samples = (
            session.query(GraphSample).filter(GraphSample.uuid.in_(sample_uids)).all()
        )

        result = {}
        for s in samples:
            result[s.uuid] = s.sample_type
            self._cache[s.uuid] = s.sample_type

        for uid in sample_uids:
            if uid not in result:
                result[uid] = ""

        session.close()
        return result


class GraphNetSampleOpSeqGetter:
    """
    Get op_seq for a given sample_uid.

    Viba:
        GraphNetSampleOpSeqGetter :=
            # __call__
            $sample_op_seq list[str]
          <- $sample_uid str
            # __init__
          <- $group_net_db_file_path str
          <- $fetch_cache dict[$sample_uid str, $sample_op_seq list[str]]
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._cache: Dict[str, List[str]] = {}

    def __call__(self, sample_uid: str) -> List[str]:
        """Get op_seq for the given sample_uid."""
        if sample_uid in self._cache:
            return self._cache[sample_uid]

        session = get_session(self.db_path)
        op_list = (
            session.query(SampleOpNameList)
            .filter(SampleOpNameList.sample_uuid == sample_uid)
            .first()
        )
        session.close()

        if op_list and op_list.op_names_json:
            op_data = json.loads(op_list.op_names_json)
            op_seq = [op["op_name"] for op in op_data]
        else:
            op_seq = []

        self._cache[sample_uid] = op_seq
        return op_seq

    def bulk_get(self, sample_uids: List[str]) -> Dict[str, List[str]]:
        """Bulk get op_seqs for multiple sample UIDs."""
        session = get_session(self.db_path)

        op_lists = (
            session.query(SampleOpNameList)
            .filter(SampleOpNameList.sample_uuid.in_(sample_uids))
            .all()
        )

        result = {}
        for op_list in op_lists:
            if op_list.op_names_json:
                op_data = json.loads(op_list.op_names_json)
                op_seq = [op["op_name"] for op in op_data]
            else:
                op_seq = []
            result[op_list.sample_uuid] = op_seq
            self._cache[op_list.sample_uuid] = op_seq

        for uid in sample_uids:
            if uid not in result:
                result[uid] = []

        session.close()
        return result
