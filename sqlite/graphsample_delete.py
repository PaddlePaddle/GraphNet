import argparse
from datetime import datetime
from orm_models import (
    get_session,
    GraphSample,
    SubgraphSource,
    DimensionGeneralizationSource,
    DataTypeGeneralizationSource,
    SampleOpNameList,
    SampleOpName,
)


def delete_graph_sample(db_path: str, relative_model_path: str, repo_uid: str = None):
    session = get_session(db_path)
    try:
        query = session.query(GraphSample).filter(
            GraphSample.relative_model_path == relative_model_path,
            GraphSample.repo_uid == repo_uid,
        )

        graph_sample = query.first()
        if not graph_sample:
            print(f"GraphSample not found: {relative_model_path}")
            return False

        if graph_sample.deleted:
            print(f"GraphSample already deleted: {relative_model_path}")
            return False

        delete_at = datetime.now()
        graph_sample.deleted = True
        graph_sample.delete_at = delete_at
        if graph_sample.is_subgraph:
            subgraph_source = (
                session.query(SubgraphSource)
                .filter(SubgraphSource.subgraph_uuid == graph_sample.uuid)
                .first()
            )
            if subgraph_source:
                subgraph_source.deleted = True
                subgraph_source.delete_at = delete_at

            dim_source = (
                session.query(DimensionGeneralizationSource)
                .filter(
                    DimensionGeneralizationSource.generalized_graph_uuid
                    == graph_sample.uuid
                )
                .first()
            )
            if dim_source:
                dim_source.deleted = True
                dim_source.delete_at = delete_at

            datatype_source = (
                session.query(DataTypeGeneralizationSource)
                .filter(
                    DataTypeGeneralizationSource.generalized_graph_uuid
                    == graph_sample.uuid
                )
                .first()
            )
            if datatype_source:
                datatype_source.deleted = True
                datatype_source.delete_at = delete_at

            session.query(SampleOpNameList).filter(
                SampleOpNameList.sample_uuid == graph_sample.uuid
            ).update({"deleted": True, "delete_at": delete_at})
            session.query(SampleOpName).filter(
                SampleOpName.sample_uuid == graph_sample.uuid
            ).update({"deleted": True, "delete_at": delete_at})

        session.commit()
        print(f"Successfully deleted: {relative_model_path}")
        return True

    except Exception as e:
        session.rollback()
        print(f"Delete failed: {e}")
        return False
    finally:
        session.close()


def main(args):
    result = delete_graph_sample(
        db_path=args.db_path,
        relative_model_path=args.relative_model_path,
        repo_uid=args.repo_uid,
    )

    if result:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="delete graph sample from database")
    parser.add_argument(
        "--repo_uid",
        type=str,
        required=True,
        help="Repository uid e.g 'github_torch_samples'",
    )
    parser.add_argument(
        "--relative_model_path",
        type=str,
        required=True,
        help="Relative model path to delete e.g 'samples/torch/resnet18'",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=False,
        default="graphnet.db",
        help="Database file path e.g 'graphnet.db'",
    )
    args = parser.parse_args()
    main(args)
