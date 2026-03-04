CREATE TABLE IF NOT EXISTS sample_input_tensor_meta (
    sample_uuid VARCHAR(255) NOT NULL,
    input_name VARCHAR(255) NOT NULL,
    input_idx INTEGER NOT NULL,
    shape TEXT NOT NULL,
    dtype VARCHAR(50) NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    delete_at DATETIME,
    deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (sample_uuid) REFERENCES graph_sample(uuid),
    PRIMARY KEY (sample_uuid, input_name, shape, dtype)
);

CREATE INDEX IF NOT EXISTS idx_sample_input_tensor_meta_sample_uid ON sample_input_tensor_meta(sample_uuid);
CREATE INDEX IF NOT EXISTS idx_sample_input_tensor_meta_shape ON sample_input_tensor_meta(shape);
CREATE INDEX IF NOT EXISTS idx_sample_input_tensor_meta_dtype ON sample_input_tensor_meta(dtype);
