--create table sample_op_name
CREATE TABLE IF NOT EXISTS sample_op_name (
    sample_uuid VARCHAR(255) NOT NULL,
    op_name VARCHAR(255) NOT NULL,
    op_idx INTEGER NOT NULL,
    op_size INTEGER NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    delete_at DATETIME,
    deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (sample_uuid) REFERENCES graph_sample(uuid),
    PRIMARY KEY (sample_uuid, op_idx)
);
CREATE INDEX IF NOT EXISTS idx_sample_op_name_sample_uuid ON sample_op_name (sample_uuid);
CREATE INDEX IF NOT EXISTS idx_sample_op_name_op_name ON sample_op_name (op_name);


--create table sample_op_name_list
CREATE TABLE IF NOT EXISTS sample_op_name_list (
    sample_uuid VARCHAR(255) NOT NULL PRIMARY KEY,
    op_names_json TEXT NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    delete_at DATETIME,
    deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (sample_uuid) REFERENCES graph_sample(uuid)
);
CREATE INDEX IF NOT EXISTS idx_sample_op_name_list_op_names ON sample_op_name_list (op_names_json);
