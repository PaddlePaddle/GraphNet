#!/usr/bin/env python3
import argparse
from orm_models import (
    get_session,
    Repo,
    GraphSample,
    SubgraphSource,
    DimensionGeneralizationSource,
    DataTypeGeneralizationSource,
    BackwardGraphSource,
    SampleOpName,
    SampleOpNameList,
    SampleInputTensorMeta,
)
from sqlalchemy.exc import IntegrityError


def merge_databases(main_db_path: str, new_db_path: str):
    main_session = get_session(main_db_path)
    new_session = get_session(new_db_path)

    stats = {
        "repo": 0,
        "graph_sample": 0,
        "subgraph_source": 0,
        "dimension_generalization_source": 0,
        "datatype_generalization_source": 0,
        "backward_graph_source": 0,
        "sample_op_name": 0,
        "sample_op_name_list": 0,
        "sample_input_tensor_meta": 0,
    }

    try:
        existing_repo_uids = {
            r.repo_uid for r in main_session.query(Repo.repo_uid).all()
        }
        new_repos = (
            new_session.query(Repo)
            .filter(Repo.repo_uid.notin_(existing_repo_uids))
            .all()
        )
        for repo in new_repos:
            new_repo = Repo(
                repo_uid=repo.repo_uid,
                repo_type=repo.repo_type,
                repo_name=repo.repo_name,
                repo_url=repo.repo_url,
            )
            main_session.add(new_repo)
            stats["repo"] += 1
        main_session.commit()

        subgraph_map = {
            s.subgraph_uuid: s for s in new_session.query(SubgraphSource).all()
        }
        dim_gen_map = {
            s.generalized_graph_uuid: s
            for s in new_session.query(DimensionGeneralizationSource).all()
        }
        dtype_gen_map = {
            s.generalized_graph_uuid: s
            for s in new_session.query(DataTypeGeneralizationSource).all()
        }
        backward_map = {
            s.backward_graph_uuid: s
            for s in new_session.query(BackwardGraphSource).all()
        }
        sample_op_name_map = {
            (s.sample_uuid, s.op_idx): s for s in new_session.query(SampleOpName).all()
        }
        sample_op_name_list_map = {
            s.sample_uuid: s for s in new_session.query(SampleOpNameList).all()
        }
        sample_input_tensor_meta_list_map = {}
        for s in new_session.query(SampleInputTensorMeta).all():
            if s.sample_uuid not in sample_input_tensor_meta_list_map:
                sample_input_tensor_meta_list_map[s.sample_uuid] = []
            sample_input_tensor_meta_list_map[s.sample_uuid].append(s)

        existing_graph_uuids = {
            g.uuid for g in main_session.query(GraphSample.uuid).all()
        }
        existing_paths = {
            (g.relative_model_path, g.repo_uid)
            for g in main_session.query(
                GraphSample.relative_model_path, GraphSample.repo_uid
            ).all()
        }
        new_graph_samples = (
            new_session.query(GraphSample)
            .filter(GraphSample.uuid.notin_(existing_graph_uuids))
            .all()
        )
        new_graph_samples.sort(key=lambda x: 0 if x.sample_type == "full_graph" else 1)

        for sample in new_graph_samples:
            if (sample.relative_model_path, sample.repo_uid) in existing_paths:
                continue
            new_sample = GraphSample(
                uuid=sample.uuid,
                repo_uid=sample.repo_uid,
                relative_model_path=sample.relative_model_path,
                sample_type=sample.sample_type,
                is_subgraph=sample.is_subgraph,
                num_ops=sample.num_ops,
                graph_hash=sample.graph_hash,
                order_value=sample.order_value,
                create_at=sample.create_at,
                deleted=sample.deleted,
                delete_at=sample.delete_at,
            )
            main_session.add(new_sample)
            stats["graph_sample"] += 1

            if sample.uuid in subgraph_map:
                src = subgraph_map[sample.uuid]
                new_subgraph = SubgraphSource(
                    subgraph_uuid=src.subgraph_uuid,
                    full_graph_uuid=src.full_graph_uuid,
                    range_start=src.range_start,
                    range_end=src.range_end,
                    create_at=src.create_at,
                    deleted=src.deleted,
                    delete_at=src.delete_at,
                )
                main_session.add(new_subgraph)
                stats["subgraph_source"] += 1

            if sample.uuid in dim_gen_map:
                src = dim_gen_map[sample.uuid]
                new_dim = DimensionGeneralizationSource(
                    generalized_graph_uuid=src.generalized_graph_uuid,
                    original_graph_uuid=src.original_graph_uuid,
                    total_element_size=src.total_element_size,
                    create_at=src.create_at,
                    deleted=src.deleted,
                    delete_at=src.delete_at,
                )
                main_session.add(new_dim)
                stats["dimension_generalization_source"] += 1

            if sample.uuid in dtype_gen_map:
                src = dtype_gen_map[sample.uuid]
                new_dtype = DataTypeGeneralizationSource(
                    generalized_graph_uuid=src.generalized_graph_uuid,
                    original_graph_uuid=src.original_graph_uuid,
                    data_type=src.data_type,
                    create_at=src.create_at,
                    deleted=src.deleted,
                    delete_at=src.delete_at,
                )
                main_session.add(new_dtype)
                stats["datatype_generalization_source"] += 1

            if sample.uuid in backward_map:
                src = backward_map[sample.uuid]
                new_back = BackwardGraphSource(
                    backward_graph_uuid=src.backward_graph_uuid,
                    forward_graph_uuid=src.forward_graph_uuid,
                    original_graph_uuid=src.original_graph_uuid,
                    create_at=src.create_at,
                    deleted=src.deleted,
                    delete_at=src.delete_at,
                )
                main_session.add(new_back)
                stats["backward_graph_source"] += 1

            if sample.uuid in sample_op_name_list_map:
                src = sample_op_name_list_map[sample.uuid]
                new_op_name_list = SampleOpNameList(
                    sample_uuid=src.sample_uuid,
                    op_names_json=src.op_names_json,
                    create_at=src.create_at,
                    deleted=src.deleted,
                    delete_at=src.delete_at,
                )
                main_session.add(new_op_name_list)
                stats["sample_op_name_list"] += 1

                for op_idx in range(sample.num_ops):
                    key = (sample.uuid, op_idx)
                    if key in sample_op_name_map:
                        src = sample_op_name_map[key]
                        new_op_name = SampleOpName(
                            sample_uuid=src.sample_uuid,
                            op_name=src.op_name,
                            op_idx=src.op_idx,
                            op_size=src.op_size,
                            create_at=src.create_at,
                            deleted=src.deleted,
                            delete_at=src.delete_at,
                        )
                        main_session.add(new_op_name)
                        stats["sample_op_name"] += 1

            if sample.uuid in sample_input_tensor_meta_list_map:
                for src in sample_input_tensor_meta_list_map[sample.uuid]:
                    new_input_tensor_meta = SampleInputTensorMeta(
                        sample_uuid=src.sample_uuid,
                        input_name=src.input_name,
                        input_idx=src.input_idx,
                        shape=src.shape,
                        dtype=src.dtype,
                        create_at=src.create_at,
                        deleted=src.deleted,
                        delete_at=src.delete_at,
                    )
                    main_session.add(new_input_tensor_meta)
                    stats["sample_input_tensor_meta"] += 1

        main_session.commit()

        total_merged = sum(stats.values())
        print(
            f"Total merged: {total_merged} records from {new_db_path} into {main_db_path}"
        )
        print(f"  Breakdown: {stats}")

        return stats

    except IntegrityError as e:
        main_session.rollback()
        print(f"IntegrityError during merge: {e}")
        return stats
    except Exception as e:
        main_session.rollback()
        print(f"Error during merge: {e}")
        return stats
    finally:
        new_session.close()
        main_session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Merge new database into main GraphNet database"
    )
    parser.add_argument(
        "--new_db_path",
        type=str,
        required=True,
        help="Path to new database file (e.g., /path/to/new.db)",
    )
    parser.add_argument(
        "--main_db_path",
        type=str,
        required=True,
        help="Path to main GraphNet.db (e.g., /path/to/GraphNet.db)",
    )

    args = parser.parse_args()

    stats = merge_databases(args.main_db_path, args.new_db_path)
    print("Merge Summary:")
    total_merged = sum(stats.values())
    print(f"  Total records merged: {total_merged}")
    for table, count in stats.items():
        print(f"  - {table}: {count}")


if __name__ == "__main__":
    main()
