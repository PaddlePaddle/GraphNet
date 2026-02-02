#!/bin/bash
set -x

MIN_SEQ_OPS=${1:-16}
MAX_SEQ_OPS=${2:-64}
GPU_ID=${3:-0}
LIMITS_HANDLED_MODELS=${4:-40}

OP_RANGE=$MIN_SEQ_OPS-$MAX_SEQ_OPS

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
RESUME="true"

DECOMPOSE_WORKSPACE=/tmp/subgraph_dataset_workspace
DEVICE_REWRITED_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/device_rewrited
DIMENSION_GENERALIZER_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/dimension_generalized_samples
OP_NAMES_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/sample_op_names
SPLIT_POINTS_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/split_points
RANGE_DECOMPOSE_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/range_decompose
GRAPH_VAR_RENAME_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/graph_var_renamed
DEDUPLICATED_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/deduplicated
CUMSUM_NUM_KERNELS_DIR=$DECOMPOSE_WORKSPACE/cumsum_num_kernels
FUSIBLE_SUBGRAPH_RANGES_DIR=$DECOMPOSE_WORKSPACE/fusible_subgraph_ranges
GROUPED_FUSIBLE_SUBGRAPH_RANGES_DIR=$DECOMPOSE_WORKSPACE/grouped_fusible_subgraph_ranges
FUSIBLE_SUBGRAPH_SAMPLES_DIR=$DECOMPOSE_WORKSPACE/fusible_subgraph_samples
RENAMED_FUSIBLE_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/renamed_fusible_subgraphs
DEDUPLICATED_FUSIBLE_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/deduplicated_fusible_subgraphs
DIMENSION_SUBGRAPH_SAMPLES_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/workspace_dimension_subgraph_samples
UNITTESTS_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/unittests

mkdir -p "$DECOMPOSE_WORKSPACE"

model_list="$GRAPH_NET_ROOT/graph_net/config/small100_torch_samples_list.txt" 
range_decomposed_subgraph_list=${DECOMPOSE_WORKSPACE}/range_decomposed_subgraph_sample_list.txt
deduplicated_subgraph_list=${DECOMPOSE_WORKSPACE}/deduplicated_subgraph_sample_list.txt
device_rewrited_subgraph_list=${DECOMPOSE_WORKSPACE}/device_rewrited_subgraph_sample_list.txt
fusible_subgraph_list=${DECOMPOSE_WORKSPACE}/fusible_subgraph_sample_list.txt
deduplicated_fusible_subgraphs_list=${DECOMPOSE_WORKSPACE}/deduplicated_fusible_subgraph_sample_list.txt
group_subgraph_sources_list=${DECOMPOSE_WORKSPACE}/group_subgraph_sources_sample_list.txt
dimension_generalizer_samples_list=dimension_generalizer_sample_list.txt
model_runnable_predicator=ModelRunnablePredicator

function generate_subgraph_list_by_index() {
    local target_dir="$1"
    local sample_list_name="$2"  
    local max_index="$3" 
    echo ">>> Generate subgraph_sample_list for samples under ${target_dir} with index 0-${max_index}."
    echo ">>>"
    for index in $(seq 0 $max_index); do
        local sample_list="${target_dir}/${index}/${sample_list_name}"
        echo ">>> Generating list for index ${index}"
        
        find ${target_dir}/${index} -name "model.py" \
            | xargs dirname \
            | xargs realpath --relative-to=${target_dir}/${index} \
            | tee $sample_list
            
        echo "Generated: $sample_list"
    done
}

function generate_subgraph_list() {
    local target_dir="$1"
    local sample_list="$2"
    echo ">>> Generate subgraph_sample_list for samples under ${target_dir}."
    echo ">>>"
    cat $model_list \
        | grep -v '# ' \
        | xargs -I {} find ${target_dir}/{} -name "model.py" \
        | xargs dirname \
        | xargs realpath --relative-to=$target_dir \
        | tee $sample_list
}

function rewrite_device() {
    echo ">>> [1] Rewrite devices for subgraph samples under ${GRAPH_NET_ROOT}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list ${model_list} \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/device_rewrite_sample_pass.py",
    "handler_class_name": "DeviceRewriteSamplePass",
    "handler_config": {
        "device": "cuda",
        "resume": ${RESUME},
        "model_path_prefix": "${GRAPH_NET_ROOT}",
        "output_dir": "${DEVICE_REWRITED_OUTPUT_DIR}"
    }
}
EOF
)
}

function dimension_generalizer(){
    echo ">>> [2] dimension generalize for samoles under ${device_rewrited_subgraph_list}."
    echo ">>>"
    python3 -m graph_net.apply_sample_pass \
        --model-path-list $device_rewrited_subgraph_list \
        --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/dimension_generalizer.py" \
        --sample-pass-class-name "ApplyDimGenPasses" \
        --sample-pass-config $(base64 -w 0 <<EOF
{
    "output_dir": "${DIMENSION_GENERALIZER_OUTPUT_DIR}",
    "model_path_prefix": "$DEVICE_REWRITED_OUTPUT_DIR",
    "dimension_generalizer_filepath": "$GRAPH_NET_ROOT/graph_net/torch/static_to_dynamic.py",
    "dimension_generalizer_class_name": "StaticToDynamic",
    "limits_handled_models": ${LIMITS_HANDLED_MODELS},
    "last_model_log_file": "/tmp/a.py"
}
EOF
)
}

function generate_op_names() {
    echo ">>> [3] Generate op_names.txt for samples in ${device_rewrited_subgraph_list}."
    echo ">>>"    
    python3 -m graph_net.model_path_handler \
        --model-path-list $device_rewrited_subgraph_list \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/op_names_extractor.py",
    "handler_class_name": "OpNamesExtractor",
    "handler_config": {
        "resume": ${RESUME},
        "model_path_prefix": "$DEVICE_REWRITED_OUTPUT_DIR",
        "output_dir": "${OP_NAMES_OUTPUT_DIR}"
    }
}
EOF
)
}

function generate_split_point() {
    echo ">>> [4] Generate split points for samples in ${device_rewrited_subgraph_list}."
    echo ">>>   MIN_SEQ_OPS: ${MIN_SEQ_OPS}, MAX_SEQ_OPS: ${MAX_SEQ_OPS}"
    echo ">>>"
    python3 -m graph_net.apply_sample_pass \
        --model-path-list $device_rewrited_subgraph_list \
        --sample-pass-file-path $GRAPH_NET_ROOT/graph_net/torch/sample_pass/typical_sequence_split_points.py \
        --sample-pass-class-name TypicalSequenceSplitPointsGenerator \
        --sample-pass-config=$(base64 -w 0 <<EOF
{
        "model_path_prefix": "$DEVICE_REWRITED_OUTPUT_DIR",
        "output_dir": "$DECOMPOSE_WORKSPACE", 
        "op_names_path_prefix": "${OP_NAMES_OUTPUT_DIR}",
        "device": "cuda",
        "window_size": 64,
        "fold_policy": "default",
        "fold_times": 16,
        "min_seq_ops": ${MIN_SEQ_OPS},
        "max_seq_ops": ${MAX_SEQ_OPS},
        "subgraph_ranges_file_name": "typical_subgraph_ranges.json",
        "subgraph_ranges_json": "$DECOMPOSE_WORKSPACE/subgraph_ranges_${OP_RANGE}ops.json",
        "output_json": "$DECOMPOSE_WORKSPACE/split_results_${OP_RANGE}ops.json"
}
EOF
)
}

function range_decompose() {
    echo ">>> [5] Decompose according to split_results.json for samples in ${device_rewrited_subgraph_list}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list "$device_rewrited_subgraph_list" \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py",
    "handler_class_name": "SubgraphGenerator",
    "handler_config": {
        "resume": ${RESUME},
        "model_path_prefix": "$DEVICE_REWRITED_OUTPUT_DIR",
        "output_dir": "${RANGE_DECOMPOSE_OUTPUT_DIR}",
        "subgraph_ranges_json_root": "$SPLIT_POINTS_OUTPUT_DIR",
        "subgraph_ranges_json_file_name": "typical_subgraph_ranges.json",
        "group_head_and_tail": false,
        "chain_style": false
    }
}
EOF
)
}

function rename_decomposed_subgraph() {
    echo ">>> [6] Rename subgraph samples under ${RANGE_DECOMPOSE_OUTPUT_DIR}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list ${range_decomposed_subgraph_list} \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/ast_graph_variable_renamer.py",
    "handler_class_name": "AstGraphVariableRenamer",
    "handler_config": {
        "device": "cuda",
        "try_run": false,
        "resume": ${RESUME},
        "model_path_prefix": "${RANGE_DECOMPOSE_OUTPUT_DIR}",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "NaiveDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "$GRAPH_VAR_RENAME_OUTPUT_DIR"
    }
}
EOF
)
}

function remove_duplicate_renamed_graphs() {
    echo ">>> [7] Remove duplicated subgraph samples under ${GRAPH_VAR_RENAME_OUTPUT_DIR}."
    echo ">>>"
    python3 -m graph_net.tools.deduplicated \
        --samples-dir ${GRAPH_VAR_RENAME_OUTPUT_DIR} \
        --target-dir ${DEDUPLICATED_OUTPUT_DIR}
}

function gen_fusible_subgraphs() {
    echo ">>> [8] Generate fusible subgraphs for subgraph samples under ${DEVICE_REWRITED_OUTPUT_DIR}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --use-subprocess    \
        --model-path-list "$deduplicated_subgraph_list" \
        --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/cumsum_num_kernels_generator.py",
    "handler_class_name": "CumSumNumKernelsGenerator",
    "handler_config": {
        "output_json_file_name": "cumsum_num_kernels.json",
        "model_path_prefix": "${DEDUPLICATED_OUTPUT_DIR}",
        "output_dir": "$CUMSUM_NUM_KERNELS_DIR",
        "device": "cuda",
        "resume": ${RESUME}
    }
}
EOF
)

    python3 -m graph_net.model_path_handler \
        --model-path-list "$deduplicated_subgraph_list" \
        --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/fusible_subgraph_ranges_generator.py",
    "handler_class_name": "FusibleSubgraphRangesGenerator",
    "handler_config": {
        "model_path_prefix": "$CUMSUM_NUM_KERNELS_DIR",
        "input_json_file_name": "cumsum_num_kernels.json",
        "output_json_file_name": "fusible_subgraph_ranges.json",
        "output_dir": "$FUSIBLE_SUBGRAPH_RANGES_DIR",
        "resume": ${RESUME}
    }
}
EOF
)

    python3 -m graph_net.apply_sample_pass \
        --model-path-list "$deduplicated_subgraph_list" \
        --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/sample_pass/group_fusible_subgraph_ranges.py" \
        --sample-pass-class-name "GroupFusibleSubgraphRanges" \
        --sample-pass-config $(base64 -w 0 <<EOF
{
    "subgraph_model_path_prefix": "$FUSIBLE_SUBGRAPH_RANGES_DIR",
    "output_dir": "$GROUPED_FUSIBLE_SUBGRAPH_RANGES_DIR",
    "input_json_file_name": "fusible_subgraph_ranges.json",
    "output_json_file_name": "grouped_fusible_subgraph_ranges.json",
    "output_json_key": "subgraph_ranges"
}
EOF
)

    python3 -m graph_net.model_path_handler \
        --model-path-list "$model_list" \
        --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py",
    "handler_class_name": "SubgraphGenerator",
    "handler_config": {
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$FUSIBLE_SUBGRAPH_SAMPLES_DIR",
        "subgraph_ranges_json_root": "$GROUPED_FUSIBLE_SUBGRAPH_RANGES_DIR",
        "subgraph_ranges_json_file_name": "grouped_fusible_subgraph_ranges.json",
        "device": "cuda",
        "resume": ${RESUME}
    }
}
EOF
)
}


function rename_fusible_subgraph() {
    echo ">>> [9] Rename subgraph samples under ${FUSIBLE_SUBGRAPH_SAMPLES_DIR}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list ${fusible_subgraph_list} \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/ast_graph_variable_renamer.py",
    "handler_class_name": "AstGraphVariableRenamer",
    "handler_config": {
        "device": "cuda",
        "try_run": false,
        "resume": ${RESUME},
        "model_path_prefix": "${FUSIBLE_SUBGRAPH_SAMPLES_DIR}",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "RenamedDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "$RENAMED_FUSIBLE_SUBGRAPH_DIR"
    }
}
EOF
)
}

function remove_duplicate_fusible_graphs() {
    echo ">>> [10] Remove duplicated subgraph samples under ${RENAMED_FUSIBLE_SUBGRAPH_DIR}."
    echo ">>>"
    python3 -m graph_net.tools.deduplicated \
        --samples-dir ${RENAMED_FUSIBLE_SUBGRAPH_DIR} \
        --target-dir ${DEDUPLICATED_FUSIBLE_SUBGRAPH_DIR}
}

function get_dimension_subgraph(){
    echo ">>> [11] Generating dimension_subgraph samples under ${DIMENSION_GENERALIZER_OUTPUT_DIR}."
    for index in {0..8}; do
        echo ">>> Generating dimension_subgraph variant index: ${index}"
        python3 -m graph_net.model_path_handler \
        --model-path-list "${DIMENSION_GENERALIZER_OUTPUT_DIR}/${index}/${dimension_generalizer_samples_list}" \
        --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py",
    "handler_class_name": "SubgraphGenerator",
    "handler_config": {
        "model_path_prefix": "${DIMENSION_GENERALIZER_OUTPUT_DIR}/${index}",
        "output_dir": "${DIMENSION_SUBGRAPH_SAMPLES_OUTPUT_DIR}/${index}",
        "subgraph_ranges_json_root": "$GROUPED_FUSIBLE_SUBGRAPH_RANGES_DIR",
        "subgraph_ranges_json_file_name": "grouped_fusible_subgraph_ranges.json",
        "device": "cuda",
        "resume": ${RESUME}
    }
}
EOF
)
    done
}

function generate_unittests() {
    echo ">>> [12] Generate unittests for subgraph samples under ${DIMENSION_SUBGRAPH_SAMPLES_OUTPUT_DIR}. "
    echo ">>>"
    for index in {0..8}; do
        echo ">>> Generating unittests variant index: ${index}"
        python3 -m graph_net.model_path_handler \
            --model-path-list "${DIMENSION_SUBGRAPH_SAMPLES_OUTPUT_DIR}/${index}/dimension_subgraph_list.txt" \
            --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "${GRAPH_NET_ROOT}/graph_net/sample_pass/agent_unittest_generator.py",
    "handler_class_name": "AgentUnittestGeneratorPass",
    "handler_config": {
        "framework": "torch",
        "model_path_prefix": "${DIMENSION_SUBGRAPH_SAMPLES_OUTPUT_DIR}/${index}",
        "output_dir": "${UNITTESTS_OUTPUT_DIR}/${index}",
        "device": "cuda",
        "generate_main": false,
        "try_run": true,
        "resume": ${RESUME},
        "data_input_predicator_filepath": "${GRAPH_NET_ROOT}/graph_net/torch/constraint_util.py",                                                                   
        "data_input_predicator_class_name": "RenamedDataInputPredicator"
    }
}
EOF
)
    done
}

main() {
    timestamp=`date +%Y%m%d_%H%M`
    suffix="${OP_RANGE}ops_${timestamp}"

    rewrite_device 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_rewrite_device_${suffix}.txt
    generate_subgraph_list ${DEVICE_REWRITED_OUTPUT_DIR} ${device_rewrited_subgraph_list}
    
    dimension_generalizer 2>&1 | tee ${DIMENSION_GENERALIZER_OUTPUT_DIR}/log_dimension_generalizer_${suffix}.txt
    generate_subgraph_list_by_index ${DIMENSION_GENERALIZER_OUTPUT_DIR} ${dimension_generalizer_samples_list} 8
    
    generate_op_names 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_op_names_${suffix}.txt
    generate_split_point 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_split_point_${suffix}.txt
    range_decompose 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_range_decompose_${suffix}.txt
    generate_subgraph_list ${RANGE_DECOMPOSE_OUTPUT_DIR} ${range_decomposed_subgraph_list}

    rename_decomposed_subgraph 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_rename_decomposed_subgraph_${suffix}.txt
    remove_duplicate_renamed_graphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_remove_duplicate_renamed_graphs_${suffix}.txt
    generate_subgraph_list ${DEDUPLICATED_OUTPUT_DIR} ${deduplicated_subgraph_list}

    gen_fusible_subgraphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_fusible_subgraphs_${suffix}.txt
    generate_subgraph_list ${FUSIBLE_SUBGRAPH_SAMPLES_DIR} ${fusible_subgraph_list}
    
    rename_fusible_subgraph 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_rename_fusible_subgraph_${suffix}.txt
    remove_duplicate_fusible_graphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_remove_duplicate_fusible_graphs_${suffix}.txt
    generate_subgraph_list ${DEDUPLICATED_FUSIBLE_SUBGRAPH_DIR} ${deduplicated_fusible_subgraphs_list}
    
    get_dimension_subgraph 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_get_dimension_subgraph_${suffix}.txt
    generate_subgraph_list_by_index ${DIMENSION_SUBGRAPH_SAMPLES_OUTPUT_DIR} dimension_subgraph_list.txt 8
    
    generate_unittests 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_unittests_${suffix}.txt
}

main
