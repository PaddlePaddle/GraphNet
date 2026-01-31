from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
    text,
)
import argparse

Base = declarative_base()


class Repo(Base):
    __tablename__ = "repo"

    repo_uid = Column(String(255), primary_key=True)
    repo_type = Column(String(50), nullable=False)
    repo_name = Column(String(255), nullable=False)
    repo_url = Column(String(255), nullable=False)
    graph_samples = relationship("GraphSample", back_populates="repo")


class GraphSample(Base):
    __tablename__ = "graph_sample"

    uuid = Column(String(255), primary_key=True)
    repo_uid = Column(String(255), ForeignKey("repo.repo_uid"), nullable=False)
    relative_model_path = Column(String, nullable=False)
    sample_type = Column(String(50), nullable=False)
    is_subgraph = Column(Boolean, default=False)
    num_ops = Column(Integer, default=-1)
    graph_hash = Column(String(255), nullable=False)
    order_value = Column(Integer)
    create_at = Column(DateTime, default=datetime.now)
    deleted = Column(Boolean, default=False)
    delete_at = Column(DateTime)

    __table_args__ = (
        Index("idx_relative_model_path", "relative_model_path"),
        Index("idx_graph_hash", "graph_hash"),
        Index("idx_order_value", "order_value"),
        UniqueConstraint(
            "relative_model_path", "repo_uid", name="uq_relative_model_path_repo_uid"
        ),
    )

    repo = relationship("Repo", back_populates="graph_samples")
    subgraph_sources = relationship(
        "SubgraphSource",
        foreign_keys="SubgraphSource.subgraph_uuid",
        back_populates="subgraph",
    )
    subgraph_as_full_graph = relationship(
        "SubgraphSource",
        foreign_keys="SubgraphSource.full_graph_uuid",
        back_populates="full_graph",
    )
    dimension_sources_as_generalized = relationship(
        "DimensionGeneralizationSource",
        foreign_keys="DimensionGeneralizationSource.generalized_graph_uuid",
        back_populates="generalized_graph",
    )
    dimension_sources_as_original = relationship(
        "DimensionGeneralizationSource",
        foreign_keys="DimensionGeneralizationSource.original_graph_uuid",
        back_populates="original_graph",
    )
    data_type_sources_as_original = relationship(
        "DataTypeGeneralizationSource",
        foreign_keys="DataTypeGeneralizationSource.original_graph_uuid",
        back_populates="original_graph",
    )
    data_type_sources_as_generalized = relationship(
        "DataTypeGeneralizationSource",
        foreign_keys="DataTypeGeneralizationSource.generalized_graph_uuid",
        back_populates="generalized_graph",
    )
    backward_graph_sources_as_forward = relationship(
        "BackwardGraphSource",
        foreign_keys="BackwardGraphSource.forward_graph_uuid",
        back_populates="forward_graph",
    )
    backward_graph_as_backward = relationship(
        "BackwardGraphSource",
        foreign_keys="BackwardGraphSource.backward_graph_uuid",
        back_populates="backward_graph",
    )
    backward_graph_as_original = relationship(
        "BackwardGraphSource",
        foreign_keys="BackwardGraphSource.original_graph_uuid",
        back_populates="original_graph",
    )


class SubgraphSource(Base):
    __tablename__ = "subgraph_source"

    subgraph_uuid = Column(
        String(255), ForeignKey("graph_sample.uuid"), nullable=False, primary_key=True
    )
    full_graph_uuid = Column(
        String(255), ForeignKey("graph_sample.uuid"), nullable=False
    )
    range_start = Column(Integer, nullable=False)
    range_end = Column(Integer, nullable=False)
    create_at = Column(DateTime, default=datetime.now)
    deleted = Column(Boolean, default=False)
    delete_at = Column(DateTime)

    __table_args__ = (
        Index("idx_subgraph_uuid", "subgraph_uuid"),
        Index("idx_full_graph_uuid", "full_graph_uuid"),
    )

    subgraph = relationship(
        "GraphSample", foreign_keys=[subgraph_uuid], back_populates="subgraph_sources"
    )
    full_graph = relationship(
        "GraphSample",
        foreign_keys=[full_graph_uuid],
        back_populates="subgraph_as_full_graph",
    )


class DimensionGeneralizationSource(Base):
    __tablename__ = "dimension_generalization_source"

    generalized_graph_uuid = Column(
        String(255), ForeignKey("graph_sample.uuid"), nullable=False, primary_key=True
    )
    original_graph_uuid = Column(
        String(255), ForeignKey("graph_sample.uuid"), nullable=False
    )
    total_element_size = Column(Integer, nullable=False)
    create_at = Column(DateTime, default=datetime.now)
    deleted = Column(Boolean, default=False)
    delete_at = Column(DateTime)

    __table_args__ = (
        Index("idx_dimension_generalized_graph_uuid", "generalized_graph_uuid"),
        Index("idx_dimension_original_graph_uuid", "original_graph_uuid"),
        Index("idx_total_element_size", "total_element_size"),
    )

    generalized_graph = relationship(
        "GraphSample",
        foreign_keys=[generalized_graph_uuid],
        back_populates="dimension_sources_as_generalized",
    )
    original_graph = relationship(
        "GraphSample",
        foreign_keys=[original_graph_uuid],
        back_populates="dimension_sources_as_original",
    )


class DataTypeGeneralizationSource(Base):
    __tablename__ = "datatype_generalization_source"

    generalized_graph_uuid = Column(
        String(255), ForeignKey("graph_sample.uuid"), nullable=False, primary_key=True
    )
    original_graph_uuid = Column(
        String(255), ForeignKey("graph_sample.uuid"), nullable=False
    )
    data_type = Column(String(50), nullable=False)
    create_at = Column(DateTime, default=datetime.now)
    deleted = Column(Boolean, default=False)
    delete_at = Column(DateTime)

    __table_args__ = (
        Index("idx_datatype_generalized_graph_uuid", "generalized_graph_uuid"),
        Index("idx_datatype_original_graph_uuid", "original_graph_uuid"),
    )

    generalized_graph = relationship(
        "GraphSample",
        foreign_keys=[generalized_graph_uuid],
        back_populates="data_type_sources_as_generalized",
    )
    original_graph = relationship(
        "GraphSample",
        foreign_keys=[original_graph_uuid],
        back_populates="data_type_sources_as_original",
    )


class BackwardGraphSource(Base):
    __tablename__ = "backward_graph_source"

    forward_graph_uuid = Column(
        String(255), ForeignKey("graph_sample.uuid"), nullable=False, primary_key=True
    )
    backward_graph_uuid = Column(
        String(255), ForeignKey("graph_sample.uuid"), nullable=False
    )
    original_graph_uuid = Column(
        String(255), ForeignKey("graph_sample.uuid"), nullable=False
    )
    create_at = Column(DateTime, default=datetime.now)
    deleted = Column(Boolean, default=False)
    delete_at = Column(DateTime)

    __table_args__ = (
        Index("idx_forward_graph_uuid", "forward_graph_uuid"),
        Index("idx_backward_graph_uuid", "backward_graph_uuid"),
        Index("idx_backward_original_graph_uuid", "original_graph_uuid"),
    )

    forward_graph = relationship(
        "GraphSample",
        foreign_keys=[forward_graph_uuid],
        back_populates="backward_graph_sources_as_forward",
    )
    backward_graph = relationship(
        "GraphSample",
        foreign_keys=[backward_graph_uuid],
        back_populates="backward_graph_as_backward",
    )
    original_graph = relationship(
        "GraphSample",
        foreign_keys=[original_graph_uuid],
        back_populates="backward_graph_as_original",
    )


def get_session(db_path: str, echo: bool = False):
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    _insert_initial_data(db_path, engine)


def drop_all_tables(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = OFF;"))

        drop_order = [
            "subgraph_source",
            "dimension_generalization_source",
            "datatype_generalization_source",
            "backward_graph_source",
            "graph_sample",
            "repo",
        ]

        for table in drop_order:
            conn.execute(text(f"DROP TABLE IF EXISTS {table};"))

    print(f"Dropped all tables: {db_path}")


def _insert_initial_data(db_path: str, engine=None):
    if engine is None:
        engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        repos = [
            Repo(
                repo_uid="github_torch_samples",
                repo_type="github",
                repo_name="GraphNet",
                repo_url="https://github.com/PaddlePaddle/GraphNet",
            ),
            Repo(
                repo_uid="github_paddle_samples",
                repo_type="github",
                repo_name="GraphNet",
                repo_url="https://github.com/PaddlePaddle/GraphNet",
            ),
        ]
        for repo in repos:
            existing = session.query(Repo).filter_by(repo_uid=repo.repo_uid).first()
            if not existing:
                session.add(repo)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphNet database management tool")
    parser.add_argument(
        "--db_path", type=str, default="sqlite/GraphNet.db", help="database path"
    )
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    init_parser = subparsers.add_parser("init", help="initialize database")
    subparsers.add_parser("drop", help="drop all tables")

    args = parser.parse_args()

    if args.command == "init":
        print(f"Initializing database: {args.db_path}")
        init_db(args.db_path)
        print("Database initialized successfully.")
    elif args.command == "drop":
        print(f"Dropping all tables: {args.db_path}")
        drop_all_tables(args.db_path)
        print("Tables dropped successfully.")
    else:
        parser.print_help()
