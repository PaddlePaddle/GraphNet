#!/bin/bash
set -x

MIN_SEQ_OPS=${1:-4}
MAX_SEQ_OPS=${2:-64}
GPU_ID=${3:-5}

OP_RANGE=$MIN_SEQ_OPS-$MAX_SEQ_OPS

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
RESUME="true"

DECOMPOSE_WORKSPACE=/work/graphnet_test_workspace/subgraph_dataset_20260203
DEVICE_REWRITED_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/01_device_rewrited_samples
DIMENSION_GENERALIZED_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/02_dimension_generalized_samples
OP_NAMES_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/03_sample_op_names
SUBGRAPH_RANGES_JSON_ROOT=$DECOMPOSE_WORKSPACE/04_subgraph_ranges
RANGE_DECOMPOSE_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/05_range_decompose_subgraphs
GRAPH_VAR_RENAME_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/06_renamed_subgraphs
DEDUPLICATED_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/07_deduplicated_subgraphs

# fusible_subgraphs
CUMSUM_NUM_KERNELS_DIR=$DECOMPOSE_WORKSPACE/08_cumsum_num_kernels
FUSIBLE_SUBGRAPH_RANGES_DIR=$DECOMPOSE_WORKSPACE/09_fusible_subgraph_ranges
GROUPED_FUSIBLE_SUBGRAPH_RANGES_DIR=$DECOMPOSE_WORKSPACE/10_grouped_fusible_subgraph_ranges
SUBGRAPH_DIMENSION_GENERALIZED_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/11_dimension_generalized_fusible_subgraphs
RENAMED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/12_renamed_dimension_generalized_fusible_subgraphs
DEDUPLICATED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/13_deduplicated_dimension_generalized_fusible_subgraphs
DTYPE_GENERALIZED_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/14_dtype_generalized_fusible_subgraphs
UNITTESTS_OUTPUT_DIR=$DECOMPOSE_WORKSPACE/15_kernelbench_unittests

# typical_subgraphs
DIMENSION_GENERALIZED_TIPICAL_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/2-08_dimension_generalized_typical_subgraphs
RENAMED_DIMENSION_GENERALIZED_TYPICAL_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/2-09_renamed_dimension_generalized_typical_subgraphs
DEDUPLICATED_DIMENSION_GENERALIZED_TYPICAL_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/2-10_deduplicated_dimension_generalized_typical_subgraphs
DTYPE_GENERALIZED_TYPICAL_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/2-11_dtype_generalized_typical_subgraphs
TYPICAL_SUBGRAPH_UNITTESTS_DIR=$DECOMPOSE_WORKSPACE/2-12_typical_kernelbench_unittests

mkdir -p "$DECOMPOSE_WORKSPACE"

model_list="$GRAPH_NET_ROOT/graph_net/config/torch_samples_list.txt"
DB_PATH=$DECOMPOSE_WORKSPACE/small100_torch_samples.db

device_rewrited_sample_list=${DECOMPOSE_WORKSPACE}/device_rewrited_sample_list.txt
range_decomposed_subgraph_list=${DECOMPOSE_WORKSPACE}/range_decomposed_subgraph_sample_list.txt
deduplicated_subgraph_list=${DECOMPOSE_WORKSPACE}/deduplicated_subgraph_sample_list.txt

# fusible_subgraphs
dimension_generalized_subgraph_list=${DECOMPOSE_WORKSPACE}/dimension_generalized_subgraph_sample_list.txt
deduplicated_fusible_subgraphs_list=${DECOMPOSE_WORKSPACE}/deduplicated_dimension_generalized_subgraph_sample_list.txt
dtype_generalized_subgraphs_list=${DECOMPOSE_WORKSPACE}/dtype_generalized_subgraphs_sample_list.txt

# typical_subgraphs
dimension_generalized_typical_subgraph_list=${DECOMPOSE_WORKSPACE}/dimension_generalized_typical_subgraph_list.txt
deduplicated_typical_subgraph_list=${DECOMPOSE_WORKSPACE}/deduplicated_dimension_generalized_typical_subgraph_list.txt
dtype_generalized_typical_subgraph_list=${DECOMPOSE_WORKSPACE}/dtype_generalized_typical_subgraph_list.txt


if [[ "$model_list" == *"/torch_samples_list.txt" ]]; then
    USE_SUBPROCESS_ARGS="--use-subprocess"
else
    USE_SUBPROCESS_ARGS=""
fi

function generate_generalized_subgraph_list() {
    local target_dir="$1"
    local sample_list="$2"
    echo ">>> Generate subgraph_sample_list for samples under ${target_dir}."
    echo ">>>"
    find ${target_dir} -name "model.py" \
        | xargs dirname \
        | xargs realpath --relative-to=${target_dir} \
        | tee $sample_list
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

function insert_graph_sample(){
    local target_dir="$1"
    local repo_uid="$2"
    local sample_type="$3"
    local sample_list="$4"
    echo ">>> [0] Inserting samples into database: ${DB_PATH}."
    echo ">>>"

    if [ ! -f "$DB_PATH" ]; then
        echo "Fail ! No Database ! : $DB_PATH"
        exit 1
    fi

    local order_value=0
    while IFS= read -r model_rel_path; do
        echo "insert : $model_rel_path"
        python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_insert.py" \
            --model_path_prefix "${target_dir}" \
            --relative_model_path "$model_rel_path" \
            --repo_uid "${repo_uid}" \
            --sample_type "${sample_type}" \
            --order_value "$order_value" \
            --db_path "$DB_PATH"

        ((order_value++))

    done < "$sample_list"
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
    echo ">>> [2] Apply dimension generalization for samples under ${device_rewrited_sample_list}."
    echo ">>>"
    python3 -m graph_net.apply_sample_pass ${USE_SUBPROCESS_ARGS} \
        --model-path-list $device_rewrited_sample_list \
        --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/dimension_generalizer.py" \
        --sample-pass-class-name "ApplyDimGenPasses" \
        --sample-pass-config $(base64 -w 0 <<EOF
{
    "output_dir": "${DIMENSION_GENERALIZED_OUTPUT_DIR}",
    "model_path_prefix": "$DEVICE_REWRITED_OUTPUT_DIR",
    "dimension_generalizer_filepath": "$GRAPH_NET_ROOT/graph_net/torch/static_to_dynamic.py",
    "dimension_generalizer_class_name": "StaticToDynamic",
    "resume": ${RESUME},
    "last_model_log_file": "/tmp/a.py"
}
EOF
)
}

function generate_op_names() {
    echo ">>> [3] Generate op_names.txt for samples in ${model_list}."
    echo ">>>"
    python3 -m graph_net.model_path_handler ${USE_SUBPROCESS_ARGS} \
        --model-path-list $model_list \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/op_names_extractor.py",
    "handler_class_name": "OpNamesExtractor",
    "handler_config": {
        "resume": ${RESUME},
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "${OP_NAMES_OUTPUT_DIR}"
    }
}
EOF
)
}

function generate_split_point() {
    echo ">>> [4] Generate subgraph_ranges.json for samples in ${model_list}."
    echo ">>>   MIN_SEQ_OPS: ${MIN_SEQ_OPS}, MAX_SEQ_OPS: ${MAX_SEQ_OPS}"
    echo ">>>"
    python3 -m graph_net.apply_sample_pass \
        --model-path-list $model_list \
        --sample-pass-file-path $GRAPH_NET_ROOT/graph_net/torch/sample_pass/typical_sequence_split_points.py \
        --sample-pass-class-name TypicalSequenceSplitPointsGenerator \
        --sample-pass-config=$(base64 -w 0 <<EOF
{
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$SUBGRAPH_RANGES_JSON_ROOT", 
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
    echo ">>> [5] Decompose according to subgraph_ranges.json for samples in ${device_rewrited_sample_list}."
    echo ">>>"
    python3 -m graph_net.model_path_handler ${USE_SUBPROCESS_ARGS} \
        --model-path-list "$device_rewrited_sample_list" \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py",
    "handler_class_name": "SubgraphGenerator",
    "handler_config": {
        "resume": ${RESUME},
        "model_path_prefix": "$DEVICE_REWRITED_OUTPUT_DIR",
        "output_dir": "${RANGE_DECOMPOSE_OUTPUT_DIR}",
        "subgraph_ranges_json_root": "${SUBGRAPH_RANGES_JSON_ROOT}",
        "subgraph_ranges_json_file_name": "typical_subgraph_ranges.json",
        "group_head_and_tail": false,
        "chain_style": false,
        "device": "cuda"
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

###############################################################################
# fusible subgraphs generating pipeline
###############################################################################

function gen_fusible_subgraph_ranges() {
    echo ">>> [8] Generate fusible subgraphs for subgraph samples under ${DEDUPLICATED_OUTPUT_DIR}."
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
}

function subgraph_dimension_generalizer(){
    echo ">>> [9] Generate dimension generalized subgraph samples under ${DIMENSION_GENERALIZED_OUTPUT_DIR}."
    for index in {0..8}; do
        echo ">>> Generating dimension generalized subgraph variant index: ${index}"
        dimension_generalized_sample_list="${DIMENSION_GENERALIZED_OUTPUT_DIR}/${index}/dimension_generalized_sample_list.txt"
        generate_subgraph_list ${DIMENSION_GENERALIZED_OUTPUT_DIR}/${index} ${dimension_generalized_sample_list}
        python3 -m graph_net.model_path_handler ${USE_SUBPROCESS_ARGS} \
            --model-path-list "${dimension_generalized_sample_list}" \
            --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py",
    "handler_class_name": "SubgraphGenerator",
    "handler_config": {
        "model_path_prefix": "${DIMENSION_GENERALIZED_OUTPUT_DIR}/${index}",
        "output_dir": "${SUBGRAPH_DIMENSION_GENERALIZED_OUTPUT_DIR}/${index}",
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

function rename_dimension_generalized_fusible_subgraph() {
    echo ">>> [10] Rename subgraph samples under ${SUBGRAPH_DIMENSION_GENERALIZED_OUTPUT_DIR}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list ${dimension_generalized_subgraph_list} \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/ast_graph_variable_renamer.py",
    "handler_class_name": "AstGraphVariableRenamer",
    "handler_config": {
        "device": "cuda",
        "try_run": false,
        "resume": ${RESUME},
        "model_path_prefix": "${SUBGRAPH_DIMENSION_GENERALIZED_OUTPUT_DIR}",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "RenamedDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "${RENAMED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}"
    }
}
EOF
)
}

function remove_duplicate_dimension_generalized_fusible_graphs() {
    echo ">>> [11] Remove duplicated subgraph samples under ${RENAMED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}."
    echo ">>>"
    for index in {0..8}; do
        python3 -m graph_net.tools.deduplicated \
            --samples-dir ${RENAMED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}/${index} \
            --target-dir ${DEDUPLICATED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}/${index}
    done
}

function dtype_generalizer() {
    echo ">>> [12] Data type generalizer for samples under ${DEDUPLICATED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}."
    echo ">>>"
    python3 -m graph_net.apply_sample_pass \
        --use-subprocess \
        --model-path-list $deduplicated_fusible_subgraphs_list \
        --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/dtype_generalizer.py" \
        --sample-pass-class-name ApplyDataTypeGeneralizationPasses \
        --sample-pass-config $(base64 -w 0 <<EOF
{
    "output_dir": "$DTYPE_GENERALIZED_OUTPUT_DIR",
    "model_path_prefix": "$DEDUPLICATED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR",
    "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
    "try_run": false,
    "device": "cuda",
    "resume": ${RESUME}
}
EOF
)
}

function generate_unittests() {
    echo ">>> [13] Generate unittests for subgraph samples under ${DTYPE_GENERALIZED_OUTPUT_DIR}. "
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list ${dtype_generalized_subgraphs_list} \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "${GRAPH_NET_ROOT}/graph_net/sample_pass/agent_unittest_generator.py",
    "handler_class_name": "AgentUnittestGeneratorPass",
    "handler_config": {
        "framework": "torch",
        "model_path_prefix": "${DTYPE_GENERALIZED_OUTPUT_DIR}",
        "output_dir": "${UNITTESTS_OUTPUT_DIR}",
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
}

###############################################################################
# typical subgraphs generating pipeline
###############################################################################

function dimension_generalizer_for_typical_subgraphs() {
    echo ">>> [2-9] Generate dimension generalized subgraph samples under ${DIMENSION_GENERALIZED_OUTPUT_DIR}."
    for index in {0..8}; do
        echo ">>> Generating dimension generalized subgraph variant index: ${index}"
        dimension_generalized_sample_list="${DIMENSION_GENERALIZED_OUTPUT_DIR}/${index}/dimension_generalized_sample_list.txt"
        dimension_generalized_subgraph_list="${DIMENSION_GENERALIZED_OUTPUT_DIR}/${index}/dimension_generalized_subgraph_list.txt"
        generate_subgraph_list ${DIMENSION_GENERALIZED_OUTPUT_DIR}/${index} ${dimension_generalized_sample_list}
        sed 's|/_decomposed/.*||' ${deduplicated_subgraph_list} | uniq | grep -Fxf "$dimension_generalized_sample_list" > ${dimension_generalized_subgraph_list}

        python3 -m graph_net.model_path_handler ${USE_SUBPROCESS_ARGS} \
            --model-path-list "${dimension_generalized_subgraph_list}" \
            --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py",
    "handler_class_name": "SubgraphGenerator",
    "handler_config": {
        "model_path_prefix": "${DIMENSION_GENERALIZED_OUTPUT_DIR}/${index}",
        "output_dir": "${DIMENSION_GENERALIZED_TIPICAL_SUBGRAPH_DIR}/${index}",
        "subgraph_ranges_json_root": "$SUBGRAPH_RANGES_JSON_ROOT",
        "subgraph_ranges_json_file_name": "typical_subgraph_ranges.json",
        "device": "cuda",
        "resume": ${RESUME}
    }
}
EOF
)
    done
}

function rename_dimension_generalized_typical_subgraphs() {
    echo ">>> [2-10] Rename subgraph samples under ${DIMENSION_GENERALIZED_TIPICAL_SUBGRAPH_DIR}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list ${dimension_generalized_typical_subgraph_list} \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/ast_graph_variable_renamer.py",
    "handler_class_name": "AstGraphVariableRenamer",
    "handler_config": {
        "device": "cuda",
        "try_run": false,
        "resume": ${RESUME},
        "model_path_prefix": "${DIMENSION_GENERALIZED_TIPICAL_SUBGRAPH_DIR}",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "RenamedDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "${RENAMED_DIMENSION_GENERALIZED_TYPICAL_SUBGRAPH_DIR}"
    }
}
EOF
)
}

function remove_duplicate_dimension_generalized_typical_subgraphs() {
    echo ">>> [2-11] Remove duplicated subgraph samples under ${RENAMED_DIMENSION_GENERALIZED_TYPICAL_SUBGRAPH_DIR}."
    echo ">>>"
    for index in {0..8}; do
        python3 -m graph_net.tools.deduplicated \
            --samples-dir ${RENAMED_DIMENSION_GENERALIZED_TYPICAL_SUBGRAPH_DIR}/${index} \
            --target-dir ${DEDUPLICATED_DIMENSION_GENERALIZED_TYPICAL_SUBGRAPH_DIR}/${index}
    done
}

function dtype_generalizer_for_typical_subgraphs() {
    echo ">>> [2-12] Data type generalizer for samples under ${DEDUPLICATED_DIMENSION_GENERALIZED_TYPICAL_SUBGRAPH_DIR}."
    echo ">>>"
    python3 -m graph_net.apply_sample_pass \
        --use-subprocess \
        --model-path-list $deduplicated_typical_subgraph_list \
        --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/dtype_generalizer.py" \
        --sample-pass-class-name ApplyDataTypeGeneralizationPasses \
        --sample-pass-config $(base64 -w 0 <<EOF
{
    "output_dir": "$DTYPE_GENERALIZED_TYPICAL_SUBGRAPH_DIR",
    "model_path_prefix": "$DEDUPLICATED_DIMENSION_GENERALIZED_TYPICAL_SUBGRAPH_DIR",
    "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
    "try_run": false,
    "device": "cuda",
    "resume": ${RESUME}
}
EOF
)
}

function generate_unittest_for_typical_subgraphs() {
    echo ">>> [2-13] Generate unittests for subgraph samples under ${DTYPE_GENERALIZED_TYPICAL_SUBGRAPH_DIR}. "
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list ${dtype_generalized_typical_subgraph_list} \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "${GRAPH_NET_ROOT}/graph_net/sample_pass/agent_unittest_generator.py",
    "handler_class_name": "AgentUnittestGeneratorPass",
    "handler_config": {
        "framework": "torch",
        "model_path_prefix": "${DTYPE_GENERALIZED_TYPICAL_SUBGRAPH_DIR}",
        "output_dir": "${TYPICAL_SUBGRAPH_UNITTESTS_DIR}",
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
}

function do_common_generalzation_and_decompose() {
    timestamp=`date +%Y%m%d_%H%M`
    suffix="${OP_RANGE}ops_${timestamp}"

    # rewrite the device in model to cuda
    rewrite_device 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_rewrite_device_${suffix}.txt
    generate_subgraph_list ${DEVICE_REWRITED_OUTPUT_DIR} ${device_rewrited_sample_list}

    # whole-graph dimension generalization
    dimension_generalizer 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_dimension_generalizer_${suffix}.txt

    # typical subgraph decomposition
    generate_op_names 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_op_names_${suffix}.txt
    generate_split_point 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_split_point_${suffix}.txt
    range_decompose 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_range_decompose_${suffix}.txt
    generate_subgraph_list ${RANGE_DECOMPOSE_OUTPUT_DIR} ${range_decomposed_subgraph_list}
    
    rename_decomposed_subgraph 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_rename_decomposed_subgraph_${suffix}.txt
    remove_duplicate_renamed_graphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_remove_duplicate_renamed_graphs_${suffix}.txt
    generate_subgraph_list ${DEDUPLICATED_OUTPUT_DIR} ${deduplicated_subgraph_list}
}

function generate_fusible_subgraphs() {
    timestamp=`date +%Y%m%d_%H%M`
    suffix="${OP_RANGE}ops_${timestamp}"

    # generate fusible subgraph ranges
    gen_fusible_subgraph_ranges 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_fusible_subgraphs_${suffix}.txt

    # subgraph dimension generalization
    subgraph_dimension_generalizer 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_subgraph_dimension_generalizer_${suffix}.txt
    generate_generalized_subgraph_list ${SUBGRAPH_DIMENSION_GENERALIZED_OUTPUT_DIR} ${dimension_generalized_subgraph_list}
    
    rename_dimension_generalized_fusible_subgraph 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_rename_dimension_generalized_subgraph_${suffix}.txt
    remove_duplicate_dimension_generalized_fusible_graphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_remove_duplicate_dimension_generalized_subgraphs_${suffix}.txt
    generate_generalized_subgraph_list ${DEDUPLICATED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR} ${deduplicated_fusible_subgraphs_list}

    # dtype generalization
    dtype_generalizer 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_dtype_generalizer_${suffix}.txt
    generate_generalized_subgraph_list ${DTYPE_GENERALIZED_OUTPUT_DIR} ${dtype_generalized_subgraphs_list}

    # generate kernelbench format unittest
    generate_unittests 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_unittests_${suffix}.txt
}

function generate_typical_subgraphs() {
    timestamp=`date +%Y%m%d_%H%M`
    suffix="${OP_RANGE}ops_${timestamp}"

    # subgraph dimension generalization
    dimension_generalizer_for_typical_subgraphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_dimension_generalizer_typical_subgraphs_${suffix}.txt
    generate_generalized_subgraph_list ${DIMENSION_GENERALIZED_TIPICAL_SUBGRAPH_DIR} ${dimension_generalized_typical_subgraph_list}

    rename_dimension_generalized_typical_subgraphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_rename_dimension_generalized_typical_subgraph_${suffix}.txt
    remove_duplicate_dimension_generalized_typical_subgraphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_remove_duplicate_dimension_generalized_subgraphs_${suffix}.txt
    generate_generalized_subgraph_list ${DEDUPLICATED_DIMENSION_GENERALIZED_TYPICAL_SUBGRAPH_DIR} ${deduplicated_typical_subgraph_list}

    # dtype generalization
    dtype_generalizer_for_typical_subgraphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_dtype_generalizer_typical_subgraphs_${suffix}.txt
    generate_generalized_subgraph_list ${DTYPE_GENERALIZED_TYPICAL_SUBGRAPH_DIR} ${dtype_generalized_typical_subgraph_list}

    # generate kernelbench format unittest
    generate_unittest_for_typical_subgraphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_unittests_typical_subgraphs_${suffix}.txt
}

function main() {
    timestamp=`date +%Y%m%d_%H%M`

    # init database
    #python ${GRAPH_NET_ROOT}/sqlite/init_db.py --db_path ${DB_PATH} 2>&1 | tee sqlite/logs/init_db_${timestamp}.log
    #insert_graph_sample ${GRAPH_NET_ROOT} "github_torch_samples" "full_graph" ${model_list}

    #generate_fusible_subgraphs
    #insert_graph_sample ${DEDUPLICATED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR} "github_torch_samples" "fusible_graph" ${dtype_generalized_subgraphs_list}

    generate_typical_subgraphs
    #insert_graph_sample ${DEDUPLICATED_OUTPUT_DIR} "github_torch_samples" "typical_graph" ${deduplicated_subgraph_list}    
}

summary() {
    num_original_samples=`cat $model_list | grep "^samples/" | wc -l`
    echo "Number of original graphnet samples: $num_original_samples"

    num_device_rewrited_samples=`find ${DEVICE_REWRITED_OUTPUT_DIR} -name "model.py" | wc -l`
    device_rewrited_successed_precent=$(( num_device_rewrited_samples * 100 / num_original_samples ))
    echo "- [Step  1] device rewrite: successed=${num_device_rewrited_samples}, percent=$device_rewrited_successed_precent%"

    num_successed_dimension_generalized_samples=`find ${DIMENSION_GENERALIZED_OUTPUT_DIR} -name "model.py" | wc -l`
    dimension_generalized_samples_successed_percent=$((num_successed_dimension_generalized_samples * 100 / (num_original_samples * 9)))
    echo "- [Step  2] dimension generalization: successed=${num_successed_dimension_generalized_samples}, percent=${dimension_generalized_samples_successed_percent}%"
    for index in {0..8}; do
        num_successed_dimension_generalized_samples=`find ${DIMENSION_GENERALIZED_OUTPUT_DIR}/${index} -name "model.py" | wc -l`
        dimension_generalized_samples_successed_percent=$(( num_successed_dimension_generalized_samples * 100 / num_original_samples ))
        echo "    ${index}, successed=${num_successed_dimension_generalized_samples}, percent=${dimension_generalized_samples_successed_percent}%"
    done
    echo ""

    num_successed_op_names=`find ${OP_NAMES_OUTPUT_DIR} -name op_names.txt | wc -l`
    op_names_successed_percent=$(( num_successed_op_names * 100 / num_original_samples ))
    echo "- [Step  3] generate op names: successed=${num_successed_op_names}, percent=${op_names_successed_percent}%"

    num_typical_subgraph_ranges=`find ${SUBGRAPH_RANGES_JSON_ROOT} -name typical_subgraph_ranges.json | wc -l`
    typical_subgraph_ranges_successed_percent=$(( num_typical_subgraph_ranges * 100 / num_original_samples ))
    echo "- [Step  4] generate typical subgraph ranges: successed=${num_typical_subgraph_ranges}, percent=${typical_subgraph_ranges_successed_percent}%"

    num_successed_range_decomposed_subgraphs=`find ${RANGE_DECOMPOSE_OUTPUT_DIR} -name "model.py" | wc -l`
    echo "- [Step  5] range decompose: successed=${num_successed_range_decomposed_subgraphs}"
    
    num_renamed_subgraphs=`find ${GRAPH_VAR_RENAME_OUTPUT_DIR} -name "model.py" | wc -l`
    echo "- [Step  6] rename: successed=${num_renamed_subgraphs}"
    
    num_deduplicated_subgraphs=`find ${DEDUPLICATED_OUTPUT_DIR} -name "model.py" | wc -l`
    echo "- [Step  7] remove duplicated: successed=${num_deduplicated_subgraphs}"

    num_successed_cumsum_kernels_subgraphs=`find ${CUMSUM_NUM_KERNELS_DIR} -name "cumsum_num_kernels.json" | wc -l`
    cumsum_kernels_successed_percent=$((num_successed_cumsum_kernels_subgraphs * 100 / num_deduplicated_subgraphs))
    echo "- [Step  8] cumsum kernels: successed=${num_successed_cumsum_kernels_subgraphs}, percent=${cumsum_kernels_successed_percent}%"

    num_fusible_subgraph_ranges=`find ${FUSIBLE_SUBGRAPH_RANGES_DIR} -name "fusible_subgraph_ranges.json" | wc -l`
    num_grouped_fusible_subgraph_ranges=`find ${GROUPED_FUSIBLE_SUBGRAPH_RANGES_DIR} -name "grouped_fusible_subgraph_ranges.json" | wc -l`
    echo "    fusible subgraph ranges: successed=${num_fusible_subgraph_ranges}"
    echo "    grouped fusible subgraph ranges: successed=${num_grouped_fusible_subgraph_ranges}"
    echo ""

    num_successed_dimension_generalized_subgraphs=`find ${SUBGRAPH_DIMENSION_GENERALIZED_OUTPUT_DIR} -name "model.py" | wc -l`
    echo "- [Step  9] subgraph dimension generalization: successed=${num_successed_dimension_generalized_subgraphs}"
    for index in {0..8}; do
        num_successed_dimension_generalized_subgraphs=`find ${SUBGRAPH_DIMENSION_GENERALIZED_OUTPUT_DIR}/${index} -name "model.py" | wc -l`
        echo "    ${index}, successed=${num_successed_dimension_generalized_subgraphs}"
    done
    echo ""

    num_renamed_fusible_subgraphs=`find ${RENAMED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Step 10] rename: successed=${num_renamed_fusible_subgraphs}"
    for index in {0..8}; do
        num_renamed_fusible_subgraphs_index=`find ${RENAMED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}/${index} -name "model.py" | wc -l`
        echo "    ${index}, successed=${num_renamed_fusible_subgraphs_index}"
    done
    echo ""

    num_deduplicated_fusible_subgraphs=`find ${DEDUPLICATED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Step 11] remove duplicated: successed=${num_deduplicated_fusible_subgraphs}"
    for index in {0..8}; do
        num_deduplicated_fusible_subgraphs_index=`find ${DEDUPLICATED_DIMENSION_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}/${index} -name "model.py" | wc -l`
        echo "    ${index}, successed=${num_deduplicated_fusible_subgraphs_index}"
    done
    echo ""

    num_dtype_generalized_subgraphs=`find ${DTYPE_GENERALIZED_OUTPUT_DIR} -name "model.py" | wc -l`
    echo "- [Step 12] dtype generalization: successed=${num_dtype_generalized_subgraphs}"
    for dtype in float32 float16 bfloat16
    do
        num_dtype_generalized_subgraphs_index=`find ${DTYPE_GENERALIZED_OUTPUT_DIR}/${dtype} -name "model.py" | wc -l`
        echo "    ${dtype}, successed=${num_dtype_generalized_subgraphs_index}"
    done
    echo ""

    num_successed_unittests=`find ${UNITTESTS_OUTPUT_DIR} -name "*_test.py" | wc -l`
    unittest_successed_percent=$((num_successed_unittests * 100 / num_dtype_generalized_subgraphs))
    echo "- [Step 13] generate unittest: successed=${num_successed_unittests}, percent=${unittest_successed_percent}%"
    for dtype in float32 float16 bfloat16
    do
        num_successed_unittests=`find ${UNITTESTS_OUTPUT_DIR}/${dtype} -name "*_test.py" | wc -l`
        echo "    ${dtype}, successed=${num_successed_unittests}"
    done
}

main

set +x
#summary 2>&1 | tee ${DECOMPOSE_WORKSPACE}/summary.txt
