-- create graph_net_sample_buckets table
CREATE TABLE IF NOT EXISTS graph_net_sample_buckets (
    sample_uid VARCHAR(255) NOT NULL PRIMARY KEY,
    op_seq_bucket_id TEXT NOT NULL,
    input_shapes_bucket_id TEXT NOT NULL,
    input_dtypes_bucket_id TEXT NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    delete_at DATETIME,
    deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (sample_uid) REFERENCES graph_sample(uuid)
);
CREATE INDEX IF NOT EXISTS idx_buckets_sample_uid ON graph_net_sample_buckets (sample_uid);
CREATE INDEX IF NOT EXISTS idx_buckets_op_seq_bucket_id ON graph_net_sample_buckets (op_seq_bucket_id);
CREATE INDEX IF NOT EXISTS idx_buckets_input_shapes_bucket_id ON graph_net_sample_buckets (input_shapes_bucket_id);
CREATE INDEX IF NOT EXISTS idx_buckets_input_dtypes_bucket_id ON graph_net_sample_buckets (input_dtypes_bucket_id);

-- create graph_net_sample_groups table
CREATE TABLE IF NOT EXISTS graph_net_sample_groups (
    sample_uid VARCHAR(255) NOT NULL,
    group_uid VARCHAR(255) NOT NULL,
    group_type VARCHAR(50) NOT NULL,
    group_policy VARCHAR(50) NOT NULL,
    policy_version VARCHAR(20) NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    delete_at DATETIME,
    deleted BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (sample_uid, group_uid),
    FOREIGN KEY (sample_uid) REFERENCES graph_sample(uuid)
);
CREATE INDEX IF NOT EXISTS idx_groups_sample_uid ON graph_net_sample_groups (sample_uid);
CREATE INDEX IF NOT EXISTS idx_groups_group_type ON graph_net_sample_groups (group_type);
CREATE INDEX IF NOT EXISTS idx_groups_group_policy ON graph_net_sample_groups (group_policy);
CREATE INDEX IF NOT EXISTS idx_groups_group_uid ON graph_net_sample_groups (group_uid);
CREATE INDEX IF NOT EXISTS idx_groups_policy_version ON graph_net_sample_groups (policy_version);
