#!/bin/bash
set -x

MIN_SEQ_OPS=${1:-4}
MAX_SEQ_OPS=${2:-64}
GPU_ID=${3:-0}

OP_RANGE=$MIN_SEQ_OPS-$MAX_SEQ_OPS

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
RESUME="true"

DECOMPOSE_WORKSPACE=/tmp/subgraph_dataset_workspace
OUTPUT_DIR=$DECOMPOSE_WORKSPACE/outputs
DB_PATH=$OUTPUT_DIR/GraphNet.db

DEVICE_REWRITED_SAMPLE_DIR=$DECOMPOSE_WORKSPACE/01_device_rewrited_samples
DIM_GENERALIZED_SAMPLE_DIR=$DECOMPOSE_WORKSPACE/02_dimension_generalized_samples
SAMPLE_OP_NAMES_DIR=$DECOMPOSE_WORKSPACE/03_sample_op_names
TYPICAL_SUBGRAPH_RANGE_DIR=$DECOMPOSE_WORKSPACE/04_typical_subgraph_ranges
TYPICAL_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/05_typical_subgraphs
RENAMED_TYPICAL_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/06_renamed_typical_subgraphs
DEDUP_TYPICAL_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/07_deduplicated_typical_subgraphs

# fusible_subgraphs
CUMSUM_NUM_KERNELS_DIR=$DECOMPOSE_WORKSPACE/08_cumsum_num_kernels
FUSIBLE_SUBGRAPH_RANGE_DIR=$DECOMPOSE_WORKSPACE/09_fusible_subgraph_ranges
GROUPED_FUSIBLE_SUBGRAPH_RANGE_DIR=$DECOMPOSE_WORKSPACE/10_grouped_fusible_subgraph_ranges
DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/11_dimension_generalized_fusible_subgraphs
RENAMED_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/12_renamed_dimension_generalized_fusible_subgraphs
DEDUP_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/13_deduplicated_dimension_generalized_fusible_subgraphs
DTYPE_GENERALIZED_FUSIBLE_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/14_dtype_generalized_fusible_subgraphs
# FUSIBLE_SUBGRAPH_UNITTEST_DIR=$DECOMPOSE_WORKSPACE/15_fusible_subgraphs_unittests

# typical_subgraphs
DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/2-08_dimension_generalized_typical_subgraphs
RENAMED_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/2-09_renamed_dimension_generalized_typical_subgraphs
DEDUP_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/2-10_deduplicated_dimension_generalized_typical_subgraphs
DTYPE_GENERALIZED_TYPICAL_SUBGRAPH_DIR=$DECOMPOSE_WORKSPACE/2-11_dtype_generalized_typical_subgraphs
# TYPICAL_SUBGRAPH_UNITTEST_DIR=$DECOMPOSE_WORKSPACE/2-12_typical_kernelbench_unittests

mkdir -p $DECOMPOSE_WORKSPACE
mkdir -p $OUTPUT_DIR

model_list="$GRAPH_NET_ROOT/graph_net/config/small100_torch_samples_list.txt"

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
        "output_dir": "${DEVICE_REWRITED_SAMPLE_DIR}"
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
    "output_dir": "${DIM_GENERALIZED_SAMPLE_DIR}",
    "model_path_prefix": "$DEVICE_REWRITED_SAMPLE_DIR",
    "dimension_generalizer_filepath": "$GRAPH_NET_ROOT/graph_net/torch/static_to_dynamic.py",
    "dimension_generalizer_class_name": "StaticToDynamic",
    "resume": ${RESUME},
    "last_model_log_file": "/tmp/a.py"
}
EOF
)
    cp -rf $DEVICE_REWRITED_SAMPLE_DIR $DIM_GENERALIZED_SAMPLE_DIR/9
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
        "output_dir": "${SAMPLE_OP_NAMES_DIR}"
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
        "output_dir": "$TYPICAL_SUBGRAPH_RANGE_DIR", 
        "op_names_path_prefix": "${SAMPLE_OP_NAMES_DIR}",
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
        "model_path_prefix": "$DEVICE_REWRITED_SAMPLE_DIR",
        "output_dir": "${TYPICAL_SUBGRAPH_DIR}",
        "subgraph_ranges_json_root": "${TYPICAL_SUBGRAPH_RANGE_DIR}",
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
    echo ">>> [6] Rename subgraph samples under ${TYPICAL_SUBGRAPH_DIR}."
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
        "model_path_prefix": "${TYPICAL_SUBGRAPH_DIR}",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "NaiveDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "$RENAMED_TYPICAL_SUBGRAPH_DIR"
    }
}
EOF
)
}

function remove_duplicate_renamed_graphs() {
    echo ">>> [7] Remove duplicated subgraph samples under ${RENAMED_TYPICAL_SUBGRAPH_DIR}."
    echo ">>>"
    python3 -m graph_net.tools.deduplicated \
        --samples-dir ${RENAMED_TYPICAL_SUBGRAPH_DIR} \
        --target-dir ${DEDUP_TYPICAL_SUBGRAPH_DIR}
}

###############################################################################
# fusible subgraphs generating pipeline
###############################################################################

function gen_fusible_subgraph_ranges() {
    echo ">>> [8] Generate fusible subgraphs for subgraph samples under ${DEDUP_TYPICAL_SUBGRAPH_DIR}."
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
        "model_path_prefix": "${DEDUP_TYPICAL_SUBGRAPH_DIR}",
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
        "output_dir": "$FUSIBLE_SUBGRAPH_RANGE_DIR",
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
    "subgraph_model_path_prefix": "$FUSIBLE_SUBGRAPH_RANGE_DIR",
    "output_dir": "$GROUPED_FUSIBLE_SUBGRAPH_RANGE_DIR",
    "input_json_file_name": "fusible_subgraph_ranges.json",
    "output_json_file_name": "grouped_fusible_subgraph_ranges.json",
    "output_json_key": "subgraph_ranges"
}
EOF
)
}

function subgraph_dimension_generalizer(){
    echo ">>> [9] Generate dimension generalized subgraph samples under ${DIM_GENERALIZED_SAMPLE_DIR}."
    for index in {0..9}; do
        echo ">>> Generating dimension generalized subgraph variant index: ${index}"
        dimension_generalized_sample_list="${DIM_GENERALIZED_SAMPLE_DIR}/${index}/dimension_generalized_sample_list.txt"
        generate_subgraph_list ${DIM_GENERALIZED_SAMPLE_DIR}/${index} ${dimension_generalized_sample_list}
        python3 -m graph_net.model_path_handler ${USE_SUBPROCESS_ARGS} \
            --model-path-list "${dimension_generalized_sample_list}" \
            --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py",
    "handler_class_name": "SubgraphGenerator",
    "handler_config": {
        "model_path_prefix": "${DIM_GENERALIZED_SAMPLE_DIR}/${index}",
        "output_dir": "${DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}/${index}",
        "subgraph_ranges_json_root": "$GROUPED_FUSIBLE_SUBGRAPH_RANGE_DIR",
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
    echo ">>> [10] Rename subgraph samples under ${DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}."
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
        "model_path_prefix": "${DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "RenamedDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "${RENAMED_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}"
    }
}
EOF
)
}

function remove_duplicate_dimension_generalized_fusible_graphs() {
    echo ">>> [11] Remove duplicated subgraph samples under ${RENAMED_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}."
    echo ">>>"
    for index in {0..9}; do
        python3 -m graph_net.tools.deduplicated \
            --samples-dir ${RENAMED_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}/${index} \
            --target-dir ${DEDUP_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}/${index}
    done
}

function dtype_generalizer() {
    echo ">>> [12] Data type generalizer for samples under ${DEDUP_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}."
    echo ">>>"
    python3 -m graph_net.apply_sample_pass \
        --use-subprocess \
        --model-path-list $deduplicated_fusible_subgraphs_list \
        --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/dtype_generalizer.py" \
        --sample-pass-class-name ApplyDataTypeGeneralizationPasses \
        --sample-pass-config $(base64 -w 0 <<EOF
{
    "output_dir": "$DTYPE_GENERALIZED_FUSIBLE_SUBGRAPH_DIR",
    "model_path_prefix": "$DEDUP_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR",
    "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
    "try_run": false,
    "device": "cuda",
    "resume": ${RESUME}
}
EOF
)
}

function generate_unittests() {
    echo ">>> [13] Generate unittests for subgraph samples under ${DTYPE_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}. "
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list ${dtype_generalized_subgraphs_list} \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "${GRAPH_NET_ROOT}/graph_net/sample_pass/agent_unittest_generator.py",
    "handler_class_name": "AgentUnittestGeneratorPass",
    "handler_config": {
        "framework": "torch",
        "model_path_prefix": "${DTYPE_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}",
        "output_dir": "${FUSIBLE_SUBGRAPH_UNITTEST_DIR}",
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
    echo ">>> [2-9] Generate dimension generalized subgraph samples under ${DIM_GENERALIZED_SAMPLE_DIR}."
    for index in {0..8}; do
        echo ">>> Generating dimension generalized subgraph variant index: ${index}"
        dimension_generalized_sample_list="${DIM_GENERALIZED_SAMPLE_DIR}/${index}/dimension_generalized_sample_list.txt"
        # An optimized method for backup
        # dimension_generalized_subgraph_list="${DIM_GENERALIZED_SAMPLE_DIR}/${index}/dimension_generalized_subgraph_list.txt"
        # generate_subgraph_list ${DIM_GENERALIZED_SAMPLE_DIR}/${index} ${dimension_generalized_sample_list}
        # sed 's|/_decomposed/.*||' ${deduplicated_subgraph_list} | uniq | grep -Fxf "$dimension_generalized_sample_list" > ${dimension_generalized_subgraph_list}
        python3 -m graph_net.model_path_handler ${USE_SUBPROCESS_ARGS} \
            --model-path-list "${dimension_generalized_sample_list}" \
            --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py",
    "handler_class_name": "SubgraphGenerator",
    "handler_config": {
        "model_path_prefix": "${DIM_GENERALIZED_SAMPLE_DIR}/${index}",
        "output_dir": "${DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR}/${index}",
        "subgraph_ranges_json_root": "$TYPICAL_SUBGRAPH_RANGE_DIR",
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
    echo ">>> [2-10] Rename subgraph samples under ${DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR}."
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
        "model_path_prefix": "${DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR}",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "RenamedDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "${RENAMED_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR}"
    }
}
EOF
)
}

function remove_duplicate_dimension_generalized_typical_subgraphs() {
    echo ">>> [2-11] Remove duplicated subgraph samples under ${RENAMED_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR}."
    echo ">>>"
    for index in {0..8}; do
        python3 -m graph_net.tools.deduplicated \
            --samples-dir ${RENAMED_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR}/${index} \
            --target-dir ${DEDUP_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR}/${index}
    done
    cp -rf ${DEDUP_TYPICAL_SUBGRAPH_DIR} ${DEDUP_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR}/9
}

function dtype_generalizer_for_typical_subgraphs() {
    echo ">>> [2-12] Data type generalizer for samples under ${DEDUP_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR}."
    echo ">>>"
    python3 -m graph_net.apply_sample_pass \
        --use-subprocess \
        --model-path-list $deduplicated_typical_subgraph_list \
        --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/dtype_generalizer.py" \
        --sample-pass-class-name ApplyDataTypeGeneralizationPasses \
        --sample-pass-config $(base64 -w 0 <<EOF
{
    "output_dir": "$DTYPE_GENERALIZED_TYPICAL_SUBGRAPH_DIR",
    "model_path_prefix": "$DEDUP_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR",
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
        "output_dir": "${TYPICAL_SUBGRAPH_UNITTEST_DIR}",
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
    generate_subgraph_list ${DEVICE_REWRITED_SAMPLE_DIR} ${device_rewrited_sample_list}

    # whole-graph dimension generalization
    dimension_generalizer 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_dimension_generalizer_${suffix}.txt

    # typical subgraph decomposition
    generate_op_names 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_op_names_${suffix}.txt
    generate_split_point 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_split_point_${suffix}.txt
    range_decompose 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_range_decompose_${suffix}.txt
    generate_subgraph_list ${TYPICAL_SUBGRAPH_DIR} ${range_decomposed_subgraph_list}
    
    rename_decomposed_subgraph 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_rename_decomposed_subgraph_${suffix}.txt
    remove_duplicate_renamed_graphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_remove_duplicate_renamed_graphs_${suffix}.txt
    generate_subgraph_list ${DEDUP_TYPICAL_SUBGRAPH_DIR} ${deduplicated_subgraph_list}
}

function generate_fusible_subgraphs() {
    timestamp=`date +%Y%m%d_%H%M`
    suffix="${OP_RANGE}ops_${timestamp}"

    # generate fusible subgraph ranges
    gen_fusible_subgraph_ranges 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_fusible_subgraphs_${suffix}.txt

    # subgraph dimension generalization
    subgraph_dimension_generalizer 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_subgraph_dimension_generalizer_${suffix}.txt
    generate_generalized_subgraph_list ${DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR} ${dimension_generalized_subgraph_list}
    
    rename_dimension_generalized_fusible_subgraph 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_rename_dimension_generalized_subgraph_${suffix}.txt
    remove_duplicate_dimension_generalized_fusible_graphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_remove_duplicate_dimension_generalized_subgraphs_${suffix}.txt
    generate_generalized_subgraph_list ${DEDUP_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR} ${deduplicated_fusible_subgraphs_list}

    # dtype generalization
    dtype_generalizer 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_dtype_generalizer_fusible_subgraphs_${suffix}.txt
    generate_generalized_subgraph_list ${DTYPE_GENERALIZED_FUSIBLE_SUBGRAPH_DIR} ${dtype_generalized_subgraphs_list}

    # generate kernelbench format unittest
    # generate_unittests 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_unittests_${suffix}.txt
}

function generate_typical_subgraphs() {
    timestamp=`date +%Y%m%d_%H%M`
    suffix="${OP_RANGE}ops_${timestamp}"

    # subgraph dimension generalization
    dimension_generalizer_for_typical_subgraphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_dimension_generalizer_typical_subgraphs_${suffix}.txt
    generate_generalized_subgraph_list ${DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR} ${dimension_generalized_typical_subgraph_list}

    rename_dimension_generalized_typical_subgraphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_rename_dimension_generalized_typical_subgraph_${suffix}.txt
    remove_duplicate_dimension_generalized_typical_subgraphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_remove_duplicate_dimension_generalized_subgraphs_${suffix}.txt
    generate_generalized_subgraph_list ${DEDUP_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR} ${deduplicated_typical_subgraph_list}

    # dtype generalization
    dtype_generalizer_for_typical_subgraphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_dtype_generalizer_typical_subgraphs_${suffix}.txt
    generate_generalized_subgraph_list ${DTYPE_GENERALIZED_TYPICAL_SUBGRAPH_DIR} ${dtype_generalized_typical_subgraph_list}

    # generate kernelbench format unittest
    # generate_unittest_for_typical_subgraphs 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_unittests_typical_subgraphs_${suffix}.txt
}

function generate_database() {
    timestamp=`date +%Y%m%d_%H%M`

    # init database
    if [ ! -f ${DB_PATH} ]; then
        python ${GRAPH_NET_ROOT}/sqlite/init_db.py --db_path ${DB_PATH} 2>&1 | tee ${DECOMPOSE_WORKSPACE}/log_init_db_${timestamp}.txt
    fi
    
    # full_graph
    insert_graph_sample ${GRAPH_NET_ROOT} "hf_torch_samples" "full_graph" ${model_list}

    # fusible_graph, typical_graph
    for sample_type in fusible_graph typical_graph; do
        insert_graph_sample $OUTPUT_DIR/$sample_type "hf_torch_samples" $sample_type $OUTPUT_DIR/${sample_type}/sample_list.txt
    done

    # insert buckets
    python ${GRAPH_NET_ROOT}/sqlite/graph_net_sample_bucket_generator.py --db_path ${DB_PATH}

    # insert groups
    python ${GRAPH_NET_ROOT}/sqlite/graph_net_sample_groups_insert.py --db_path ${DB_PATH}
}

function main() {
    do_common_generalzation_and_decompose

    sample_type="fusible_graph"
    generate_fusible_subgraphs
    cp -rf $DTYPE_GENERALIZED_FUSIBLE_SUBGRAPH_DIR $OUTPUT_DIR/$sample_type
    cp -rf $dtype_generalized_subgraphs_list $OUTPUT_DIR/$sample_type/sample_list.txt

    sample_type="typical_graph"
    generate_typical_subgraphs
    cp -rf $DTYPE_GENERALIZED_TYPICAL_SUBGRAPH_DIR $OUTPUT_DIR/$sample_type
    cp -rf $dtype_generalized_typical_subgraph_list $OUTPUT_DIR/$sample_type/sample_list.txt

    generate_database
}

function summary() {
    num_original_samples=`cat $model_list | grep "^samples/" | wc -l`
    echo "Number of original graphnet samples: $num_original_samples"

    num_device_rewrited_samples=`find ${DEVICE_REWRITED_SAMPLE_DIR} -name "model.py" | wc -l`
    device_rewrited_successed_precent=$(( num_device_rewrited_samples * 100 / num_original_samples ))
    echo "- [Common - 1] device rewrite: successed=${num_device_rewrited_samples}, percent=$device_rewrited_successed_precent%"

    num_dim_generalized_samples=`find ${DIM_GENERALIZED_SAMPLE_DIR} -name "model.py" | wc -l`
    dim_generalization_successed_percent=$((num_dim_generalized_samples * 100 / (num_original_samples * 9)))
    echo "- [Common - 2] dimension generalization: successed=${num_dim_generalized_samples}, percent=${dim_generalization_successed_percent}%"
    for index in {0..9}; do
        num_dim_generalized_samples_index=`find ${DIM_GENERALIZED_SAMPLE_DIR}/${index} -name "model.py" | wc -l`
        dim_generalization_successed_percent=$((num_dim_generalized_samples_index * 100 / num_original_samples))
        echo "    ${index}, successed=${num_dim_generalized_samples_index}, percent=${dim_generalization_successed_percent}%"
    done
    echo ""

    num_op_names=`find ${SAMPLE_OP_NAMES_DIR} -name op_names.txt | wc -l`
    op_names_successed_percent=$((num_op_names * 100 / num_original_samples))
    echo "- [Common - 3] generate op names: successed=${num_op_names}, percent=${op_names_successed_percent}%"

    num_typical_subgraph_ranges=`find ${TYPICAL_SUBGRAPH_RANGE_DIR} -name typical_subgraph_ranges.json | wc -l`
    typical_subgraph_ranges_successed_percent=$((num_typical_subgraph_ranges * 100 / num_original_samples))
    echo "- [Common - 4] generate typical subgraph ranges: successed=${num_typical_subgraph_ranges}, percent=${typical_subgraph_ranges_successed_percent}%"

    num_typical_subgraphs=`find ${TYPICAL_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Common - 5] range decompose: successed=${num_typical_subgraphs}"

    num_renamed_typical_subgraphs=`find ${RENAMED_TYPICAL_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Common - 6] rename: successed=${num_renamed_typical_subgraphs}"
    
    num_dedup_typical_subgraphs=`find ${DEDUP_TYPICAL_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Common - 7] remove duplicated: successed=${num_dedup_typical_subgraphs}"
    echo ""

    # fusible subgraphs
    num_cumsum_kernels_subgraphs=`find ${CUMSUM_NUM_KERNELS_DIR} -name "cumsum_num_kernels.json" | wc -l`
    cumsum_kernels_successed_percent=$((num_cumsum_kernels_subgraphs * 100 / num_dedup_typical_subgraphs))
    echo "- [Fusible - 1] cumsum kernels: successed=${num_cumsum_kernels_subgraphs}, percent=${cumsum_kernels_successed_percent}%"

    num_fusible_subgraph_ranges=`find ${FUSIBLE_SUBGRAPH_RANGE_DIR} -name "fusible_subgraph_ranges.json" | wc -l`
    num_grouped_fusible_subgraph_ranges=`find ${GROUPED_FUSIBLE_SUBGRAPH_RANGE_DIR} -name "grouped_fusible_subgraph_ranges.json" | wc -l`
    echo "    fusible subgraph ranges: successed=${num_fusible_subgraph_ranges}"
    echo "    grouped fusible subgraph ranges: successed=${num_grouped_fusible_subgraph_ranges}"
    echo ""

    num_dim_generalized_subgraphs=`find ${DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Fusible - 2] subgraph dimension generalization: successed=${num_dim_generalized_subgraphs}"
    for index in {0..9}; do
        num_dim_generalized_subgraphs=`find ${DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}/${index} -name "model.py" | wc -l`
        echo "    ${index}, successed=${num_dim_generalized_subgraphs}"
    done
    echo ""

    num_renamed_fusible_subgraphs=`find ${RENAMED_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Fusible - 3] rename: successed=${num_renamed_fusible_subgraphs}"
    for index in {0..9}; do
        num_renamed_fusible_subgraphs_index=`find ${RENAMED_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}/${index} -name "model.py" | wc -l`
        echo "    ${index}, successed=${num_renamed_fusible_subgraphs_index}"
    done
    echo ""

    num_deduplicated_fusible_subgraphs=`find ${DEDUP_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Fusible - 4] remove duplicated: successed=${num_deduplicated_fusible_subgraphs}"
    for index in {0..9}; do
        num_deduplicated_fusible_subgraphs_index=`find ${DEDUP_DIM_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}/${index} -name "model.py" | wc -l`
        echo "    ${index}, successed=${num_deduplicated_fusible_subgraphs_index}"
    done
    echo ""

    num_dtype_generalized_subgraphs=`find ${DTYPE_GENERALIZED_FUSIBLE_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Fusible - 5] dtype generalization: successed=${num_dtype_generalized_subgraphs}"
    for dtype in float32 float16 bfloat16
    do
        num_dtype_generalized_subgraphs_index=`find ${DTYPE_GENERALIZED_FUSIBLE_SUBGRAPH_DIR}/${dtype} -name "model.py" | wc -l`
        echo "    ${dtype}, successed=${num_dtype_generalized_subgraphs_index}"
    done
    echo ""

    # typical subgraphs
    num_dim_generalized_typical_subgraphs=`find ${DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Typical - 1] subgraph dimension generalization: successed=${num_dim_generalized_typical_subgraphs}"
    for index in {0..8}; do
        num_dim_generalized_typical_index=`find ${DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR}/${index} -name "model.py" | wc -l`
        echo "    ${index}, successed=${num_dim_generalized_typical_index}"
    done
    echo ""

    num_renamed_dim_generalized_typical_subgraphs=`find ${RENAMED_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Typical - 2] rename: successed=${num_renamed_dim_generalized_typical_subgraphs}"
    for index in {0..8}; do
        num_renamed_dim_generalized_typical_subgraphs_index=`find ${RENAMED_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR}/${index} -name "model.py" | wc -l`
        echo "    ${index}, successed=${num_renamed_dim_generalized_typical_subgraphs_index}"
    done
    echo ""

    num_dedup_typical_subgraphs=`find ${DEDUP_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Typical - 3] remove duplicated: successed=${num_dedup_typical_subgraphs}"
    for index in {0..9}; do
        num_deduplicated_typical_subgraphs_index=`find ${DEDUP_DIM_GENERALIZED_TYPICAL_SUBGRAPH_DIR}/${index} -name "model.py" | wc -l`
        echo "    ${index}, successed=${num_deduplicated_typical_subgraphs_index}"
    done
    echo ""

    num_dtype_generalized_typical_subgraphs=`find ${DTYPE_GENERALIZED_TYPICAL_SUBGRAPH_DIR} -name "model.py" | wc -l`
    echo "- [Typical - 4] dtype generalization: successed=${num_dtype_generalized_typical_subgraphs}"
    for dtype in float32 float16 bfloat16
    do
        num_dtype_generalized_typical_subgraphs_index=`find ${DTYPE_GENERALIZED_TYPICAL_SUBGRAPH_DIR}/${dtype} -name "model.py" | wc -l`
        echo "    ${dtype}, successed=${num_dtype_generalized_typical_subgraphs_index}"
    done
    echo ""
}

main

set +x
summary 2>&1 | tee ${DECOMPOSE_WORKSPACE}/summary.txt
