CREATE TEMP VIEW v_sample_buckets_groups AS
SELECT
    g.sample_type,
    g.relative_model_path,
    b.sample_uid,
    b.op_seq_bucket_id,
    grp.group_uid
FROM graph_sample g
JOIN graph_net_sample_buckets b ON g.uuid = b.sample_uid
JOIN graph_net_sample_groups grp ON g.uuid = grp.sample_uid
WHERE g.deleted = 0 AND b.deleted = 0 AND grp.deleted = 0;

select distinct group_concat(sample_uid) as sample_uids, op_seq_bucket_id
from (
    select *
    from v_sample_buckets_groups
    order by sample_uid asc
)
where sample_type = 'fusible_graph'
group by group_uid
order by op_seq_bucket_id asc
;
