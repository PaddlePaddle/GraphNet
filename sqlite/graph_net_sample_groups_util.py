# graph_net_sample_groups_util.py
from typing import List, Set, Dict
from collections import defaultdict

from orm_models import get_session, GraphNetSampleGroup


def get_all_graph_net_sample_groups(
    db_path: str,
    group_types: List[str],
    group_policies: List[str],
    versions: List[str],
) -> List[Set[str]]:
    """
    Get all graph_net sample groups from database.

    Viba:
        get_all_graph_net_sample_groups :=
            list[set[$sample_uid str]]
          <- $group_net_db_file_path str
          <- $group_type list[str]
          <- $group_policy list[str]
          <- $version list[str]

    Args:
        db_path: Path to the SQLite database file.
        group_types: List of group types to filter (e.g., ["shape_diversity", "dtype_diversity"]).
        group_policies: List of group policies to filter (e.g., ["by_bucket"]).
        versions: List of policy versions to filter (e.g., ["v0.1"]).

    Returns:
        List of sets, each set contains sample UIDs belonging to one group.
    """
    session = get_session(db_path)

    query = session.query(GraphNetSampleGroup).filter(
        GraphNetSampleGroup.deleted.is_(False),
        GraphNetSampleGroup.group_type.in_(group_types),
        GraphNetSampleGroup.group_policy.in_(group_policies),
        GraphNetSampleGroup.policy_version.in_(versions),
    )

    groups_dict: Dict[str, List[str]] = defaultdict(list)
    for row in query.all():
        groups_dict[row.group_uid].append(row.sample_uid)

    session.close()

    return [set(uids) for uids in groups_dict.values()]
