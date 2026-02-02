-- SQLite
-- create repo table
CREATE TABLE IF NOT EXISTS repo (
    repo_uid VARCHAR(255) NOT NULL PRIMARY KEY,
    repo_type VARCHAR(50) NOT NULL,
    repo_name VARCHAR(255) NOT NULL,
    repo_url TEXT
);
INSERT OR IGNORE INTO repo (repo_uid, repo_type, repo_name, repo_url) VALUES
('github_torch_samples', 'github', 'GraphNet', 'https://github.com/PaddlePaddle/GraphNet'),
('github_paddle_samples', 'github', 'GraphNet', 'https://github.com/PaddlePaddle/GraphNet');


-- create graph_sample table
CREATE TABLE IF NOT EXISTS graph_sample (
    uuid VARCHAR(255) NOT NULL PRIMARY KEY,
    repo_uid VARCHAR(255) NOT NULL,
    relative_model_path TEXT NOT NULL,
    sample_type VARCHAR(50) NOT NULL,
    is_subgraph BOOLEAN DEFAULT FALSE,
    num_ops INTEGER DEFAULT -1,
    graph_hash VARCHAR(255) NOT NULL,
    order_value INTEGER,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE,
    delete_at DATETIME,
    FOREIGN KEY (repo_uid) REFERENCES repo(repo_uid)
);
CREATE INDEX IF NOT EXISTS idx_relative_model_path ON graph_sample (relative_model_path);
CREATE INDEX IF NOT EXISTS idx_graph_hash ON graph_sample (graph_hash);
CREATE INDEX IF NOT EXISTS idx_order_value ON graph_sample (order_value);
CREATE UNIQUE INDEX IF NOT EXISTS uq_relative_model_path_repo_uid ON graph_sample (relative_model_path, repo_uid);

-- create subgraph_source table
CREATE TABLE IF NOT EXISTS subgraph_source (
    subgraph_uuid VARCHAR(255) NOT NULL PRIMARY KEY,
    full_graph_uuid VARCHAR(255) NOT NULL,
    range_start INTEGER NOT NULL,
    range_end INTEGER NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE,
    delete_at DATETIME,
    FOREIGN KEY (subgraph_uuid) REFERENCES graph_sample(uuid),
    FOREIGN KEY (full_graph_uuid) REFERENCES graph_sample(uuid)
);
CREATE INDEX IF NOT EXISTS idx_subgraph_uuid ON subgraph_source (subgraph_uuid);
CREATE INDEX IF NOT EXISTS idx_full_graph_uuid ON subgraph_source (full_graph_uuid);

-- create dimension_generalization_source table
CREATE TABLE IF NOT EXISTS dimension_generalization_source (
    generalized_graph_uuid VARCHAR(255) NOT NULL PRIMARY KEY,
    original_graph_uuid VARCHAR(255) NOT NULL,
    total_element_size INTEGER NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE,
    delete_at DATETIME,
    FOREIGN KEY (generalized_graph_uuid) REFERENCES graph_sample(uuid),
    FOREIGN KEY (original_graph_uuid) REFERENCES graph_sample(uuid)
);
CREATE INDEX IF NOT EXISTS idx_dimension_generalized_graph_uuid ON dimension_generalization_source (generalized_graph_uuid);
CREATE INDEX IF NOT EXISTS idx_dimension_original_graph_uuid ON dimension_generalization_source (original_graph_uuid);
CREATE INDEX IF NOT EXISTS idx_total_element_size ON dimension_generalization_source (total_element_size);

-- create datatype_generalization_source table
CREATE TABLE IF NOT EXISTS datatype_generalization_source (
    generalized_graph_uuid VARCHAR(255) NOT NULL PRIMARY KEY,
    original_graph_uuid VARCHAR(255) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE,
    delete_at DATETIME,
    FOREIGN KEY (generalized_graph_uuid) REFERENCES graph_sample(uuid),
    FOREIGN KEY (original_graph_uuid) REFERENCES graph_sample(uuid)
);
CREATE INDEX IF NOT EXISTS idx_datatype_generalized_graph_uuid ON datatype_generalization_source (generalized_graph_uuid);
CREATE INDEX IF NOT EXISTS idx_datatype_original_graph_uuid ON datatype_generalization_source (original_graph_uuid);

-- create backward_graph_source table
CREATE TABLE IF NOT EXISTS backward_graph_source (
    forward_graph_uuid VARCHAR(255) NOT NULL PRIMARY KEY,
    backward_graph_uuid VARCHAR(255) NOT NULL,
    original_graph_uuid VARCHAR(255) NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE,
    delete_at DATETIME,
    FOREIGN KEY (forward_graph_uuid) REFERENCES graph_sample(uuid),
    FOREIGN KEY (backward_graph_uuid) REFERENCES graph_sample(uuid),
    FOREIGN KEY (original_graph_uuid) REFERENCES graph_sample(uuid)
);
CREATE INDEX IF NOT EXISTS idx_forward_graph_uuid ON backward_graph_source (forward_graph_uuid);
CREATE INDEX IF NOT EXISTS idx_backward_graph_uuid ON backward_graph_source (backward_graph_uuid);
CREATE INDEX IF NOT EXISTS idx_backward_original_graph_uuid ON backward_graph_source (original_graph_uuid);
