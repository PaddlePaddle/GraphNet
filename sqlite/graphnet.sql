-- SQLite
-- create Repo table
DROP TABLE IF EXISTS Repo;
CREATE TABLE Repo (
    repo_uid VARCHAR(255) NOT NULL PRIMARY KEY,
    repo_type VARCHAR(50) NOT NULL,
    repo_name VARCHAR(255) NOT NULL,
    repo_url TEXT
);

INSERT INTO Repo (repo_uid, repo_type, repo_name, repo_url) VALUES
('github_torch_samples', 'github', 'GraphNet', 'https://github.com/PaddlePaddle/GraphNet'),
('github_paddle_samples', 'github', 'GraphNet', 'https://github.com/PaddlePaddle/GraphNet');



-- create GraphSample table
DROP TABLE IF EXISTS GraphSample;
CREATE TABLE GraphSample (
    uuid VARCHAR(255) NOT NULL PRIMARY KEY,
    repo_uid VARCHAR(255) NOT NULL,
    relative_model_path TEXT NOT NULL UNIQUE,
    sample_type VARCHAR(50) NOT NULL,
    is_subgraph BOOLEAN DEFAULT FALSE,
    num_ops INTEGER DEFAULT -1,
    graph_hash VARCHAR(255) NOT NULL,
    order_value INTEGER,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE,
    delete_at DATETIME,
    FOREIGN KEY (repo_uid) REFERENCES Repo(repo_uid)
);
CREATE INDEX idx_relative_model_path ON GraphSample (relative_model_path);
CREATE INDEX idx_graph_hash ON GraphSample (graph_hash);
CREATE INDEX idx_order_value ON GraphSample (order_value);

-- create SubgraphSource table
DROP TABLE IF EXISTS SubgraphSource;
CREATE TABLE SubgraphSource (
    subgraph_uuid VARCHAR(255) NOT NULL PRIMARY KEY,
    full_graph_uuid VARCHAR(255) NOT NULL,
    range_start INTEGER NOT NULL,
    range_end INTEGER NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE,
    delete_at DATETIME,
    FOREIGN KEY (full_graph_uuid) REFERENCES GraphSample(uuid)
);
CREATE INDEX idx_subgraph_uuid ON SubgraphSource (subgraph_uuid);
CREATE INDEX idx_full_graph_uuid ON SubgraphSource (full_graph_uuid);

-- create DimensionGeneralizationSource table
DROP TABLE IF EXISTS DimensionGeneralizationSource;
CREATE TABLE DimensionGeneralizationSource (
    generalized_graph_uuid VARCHAR(255) NOT NULL,
    original_graph_uuid VARCHAR(255) NOT NULL,
    total_element_size INTEGER NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE,
    delete_at DATETIME,
    FOREIGN KEY (generalized_graph_uuid) REFERENCES GraphSample(uuid),
    FOREIGN KEY (original_graph_uuid) REFERENCES GraphSample(uuid)
);
CREATE INDEX idx_dimension_generalized_graph_uuid ON DimensionGeneralizationSource (generalized_graph_uuid);
CREATE INDEX idx_dimension_original_graph_uuid ON DimensionGeneralizationSource (original_graph_uuid);
CREATE INDEX idx_total_element_size ON DimensionGeneralizationSource (total_element_size);

-- create DataTypeGeneralizationSource table
DROP TABLE IF EXISTS DataTypeGeneralizationSource;
CREATE TABLE DataTypeGeneralizationSource (
    generalized_graph_uuid VARCHAR(255) NOT NULL,
    original_graph_uuid VARCHAR(255) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE,
    delete_at DATETIME,
    FOREIGN KEY (generalized_graph_uuid) REFERENCES GraphSample(uuid),
    FOREIGN KEY (original_graph_uuid) REFERENCES GraphSample(uuid)
);
CREATE INDEX idx_datatype_generalized_graph_uuid ON DataTypeGeneralizationSource (generalized_graph_uuid);
CREATE INDEX idx_datatype_original_graph_uuid ON DataTypeGeneralizationSource (original_graph_uuid);
