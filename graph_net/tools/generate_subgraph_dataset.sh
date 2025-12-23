#!/bin/bash
set -x

MIN_SEQ_OPS=${1:-16}
MAX_SEQ_OPS=${2:-64}
GPU_ID=${3:-0}
OP_RANGE="${MIN_SEQ_OPS}-${MAX_SEQ_OPS}"
RESUME="true"

export CUDA_VISIBLE_DEVICES=$GPU_ID

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
DECOMPOSE_WORKSPACE=/tmp/subgraph_dataset_workspace
LEVEL_WORKSPACE=$DECOMPOSE_WORKSPACE/decomposed_${OP_RANGE}ops
model_list="$GRAPH_NET_ROOT/graph_net/config/small100_torch_samples_list.txt"

mkdir -p "$LEVEL_WORKSPACE"

declare -A DIRS=(
    ["OP_NAMES"]="$DECOMPOSE_WORKSPACE/sample_op_names"
    ["RANGE_DECOMPOSE_SAMPLES"]="$LEVEL_WORKSPACE/decomposed_subgraphs"
    ["RENAMED_SAMPLES"]="$LEVEL_WORKSPACE/decomposed_subgraphs_renamed"
    ["DEDUPED_SAMPLES"]="$LEVEL_WORKSPACE/decomposed_subgraphs_deduped"
    ["CUMSUM_NUM_KERNELS"]="$LEVEL_WORKSPACE/cumsum_num_kernels"
    ["FUSIBLE_RANGES"]="$LEVEL_WORKSPACE/fusible_subgraph_ranges"
    ["FUSIBLE_SAMPLES"]="$LEVEL_WORKSPACE/fusible_subgraphs"
    ["RENAMED_FUSIBLE_SAMPLES"]="$LEVEL_WORKSPACE/fusible_subgraphs_renamed"
    ["DEDUPED_FUSIBLE_SAMPLES"]="$LEVEL_WORKSPACE/fusible_subgraphs_deduped"
    ["DEVICE_REWRITTEN_SAMPLES"]="$LEVEL_WORKSPACE/fusible_subgraphs_device_rewritten"
    ["UNITTESTS"]="$LEVEL_WORKSPACE/fusible_subgraphs_unittests"
)

function RUN_PIPELINE_STEP() {
    local step="$1"
    local command="$2"
    shift 2
    
    echo ">>> [$step] Performing: $command"
    echo ">>>"
    eval "$command \"\$@\"" 2>&1 | tee "${LEVEL_WORKSPACE}/log_${command}_${OP_RANGE}ops_$(date +%Y%m%d_%H%M).txt"
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

function generate_op_names() {
    python3 -m graph_net.model_path_handler \
        --model-path-list "$1" \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/typical_sequence_split_points.py",
    "handler_class_name": "OpNamesExtractor",
    "handler_config": {
        "resume": ${RESUME},
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$2"
    }
}
EOF
)
}

function generate_split_point() {
    python3 -m graph_net.torch.typical_sequence_split_points \
        --model-list "$1" \
        --op-names-path-prefix "$2" \
        --device "cuda" \
        --window-size 64 \
        --fold-policy default \
        --fold-times 16 \
        --min-seq-ops ${MIN_SEQ_OPS} \
        --max-seq-ops ${MAX_SEQ_OPS} \
        --subgraph-ranges-json "$3/subgraph_ranges_${OP_RANGE}ops.json" \
        --output-json "$3/split_results_${OP_RANGE}ops.json"
}

function range_decompose() {
    python3 -m graph_net.model_path_handler \
        --model-path-list "$1" \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/graph_decomposer.py",
    "handler_class_name": "RangeDecomposerExtractor",
    "handler_config": {
        "resume": ${RESUME},
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$3",
        "split_results_path": "$2/split_results_${OP_RANGE}ops.json",
        "subgraph_ranges_path": "$2/subgraph_ranges_${OP_RANGE}ops.json",
        "group_head_and_tail": true,
        "chain_style": false
    }
}
EOF
)
}

function rename_subgraph() {
    python3 -m graph_net.model_path_handler \
        --model-path-list $1 \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/graph_variable_renamer.py",
    "handler_class_name": "GraphVariableRenamer",
    "handler_config": {
        "device": "cuda",
        "resume": ${RESUME},
        "model_path_prefix": "$2",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "NaiveDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "$3"
    }
}
EOF
)
}

function remove_duplicates() {
    python3 -m graph_net.tools.deduplicated \
        --samples-dir "$1" \
        --target-dir "$2"
}

function rewrite_device() {
    python3 -m graph_net.model_path_handler \
        --model-path-list "$1" \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_passes/device_rewrite_sample_pass.py",
    "handler_class_name": "DeviceRewriteSamplePass",
    "handler_config": {
        "device": "cuda",
        "resume": ${RESUME},
        "model_path_prefix": "$2",
        "output_dir": "$3"
    }
}
EOF
)
}

function cumsum_num_kernels() {
    python3 -m graph_net.model_path_handler \
        --use-subprocess    \
        --model-path-list "$1" \
        --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_passes/cumsum_num_kernels_generator.py",
    "handler_class_name": "CumSumNumKernelsGenerator",
    "handler_config": {
        "output_json_file_name": "cumsum_num_kernels.json",
        "model_path_prefix": "$2",
        "output_dir": "$3",
        "resume": ${RESUME}
    }
}
EOF
)
}
function gen_fusible_subgraphs_ranges(){
    python3 -m graph_net.model_path_handler \
        --model-path-list "$1" \
        --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/fusible_subgraph_ranges_generator.py",
    "handler_class_name": "FusibleSubgraphRangesGenerator",
    "handler_config": {
        "model_path_prefix": "$2",
        "input_json_file_name": "cumsum_num_kernels.json",
        "output_json_file_name": "fusible_subgraph_ranges.json",
        "output_dir": "$3",
        "resume": ${RESUME}
    }
}
EOF
)
}
    
function gen_fusible_subgraphs() {
    python3 -m graph_net.model_path_handler \
        --model-path-list "$1" \
        --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_passes/subgraph_generator.py",
    "handler_class_name": "SubgraphGenerator",
    "handler_config": {
        "model_path_prefix": "$2",
        "output_dir": "$4",
        "subgraph_ranges_json_root": "$3",
        "resume": ${RESUME}
    }
}
EOF
)
}

function generate_unittests() {
    python3 -m graph_net.model_path_handler \
        --model-path-list $1 \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/agent_unittest_generator.py",
    "handler_class_name": "AgentUnittestGeneratorPass",
    "handler_config": {
        "framework": "torch",
        "model_path_prefix": "$2",
        "output_dir": "$3",
        "device": "cuda",
        "generate_main": false,
        "try_run": true,
        "resume": ${RESUME},
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",                                                                                     
        "data_input_predicator_class_name": "RenamedDataInputPredicator"
    }
}
EOF
)
}

main() {
    RUN_PIPELINE_STEP "1" generate_op_names "$model_list" ${DIRS["OP_NAMES"]}
    RUN_PIPELINE_STEP "2" generate_split_point "$model_list" ${DIRS["OP_NAMES"]} ${LEVEL_WORKSPACE}
    RUN_PIPELINE_STEP "3" range_decompose "$model_list" ${LEVEL_WORKSPACE} ${DIRS["RANGE_DECOMPOSE_SAMPLES"]}

    range_decomposed_subgraphs_list="$LEVEL_WORKSPACE/range_decomposed_subgraph_sample_list.txt"
    generate_subgraph_list ${DIRS["RANGE_DECOMPOSE_SAMPLES"]} $range_decomposed_subgraphs_list
    RUN_PIPELINE_STEP "4" rename_subgraph "$range_decomposed_subgraphs_list" ${DIRS["RANGE_DECOMPOSE_SAMPLES"]} ${DIRS["RENAMED_SAMPLES"]}
    RUN_PIPELINE_STEP "5" remove_duplicates ${DIRS["RENAMED_SAMPLES"]} ${DIRS["DEDUPED_SAMPLES"]}

    deduped_subgraphs_list="$LEVEL_WORKSPACE/deduplicated_subgraph_sample_list.txt"
    generate_subgraph_list ${DIRS["DEDUPED_SAMPLES"]} $deduped_subgraphs_list
    RUN_PIPELINE_STEP "6" cumsum_num_kernels "$deduped_subgraphs_list" ${DIRS["DEDUPED_SAMPLES"]} ${DIRS["CUMSUM_NUM_KERNELS"]}
    RUN_PIPELINE_STEP "7" gen_fusible_subgraphs_ranges "$deduped_subgraphs_list" ${DIRS["CUMSUM_NUM_KERNELS"]} ${DIRS["FUSIBLE_RANGES"]}
    RUN_PIPELINE_STEP "8" gen_fusible_subgraphs "$deduped_subgraphs_list" ${DIRS["DEDUPED_SAMPLES"]} ${DIRS["FUSIBLE_RANGES"]} ${DIRS["FUSIBLE_SAMPLES"]}

    fusible_subgraphs_list="$LEVEL_WORKSPACE/fusible_subgraphs_list.txt"
    generate_subgraph_list ${DIRS["FUSIBLE_SAMPLES"]} $fusible_subgraphs_list
    RUN_PIPELINE_STEP "9" rename_subgraph "$fusible_subgraphs_list" ${DIRS["FUSIBLE_SAMPLES"]} ${DIRS["RENAMED_FUSIBLE_SAMPLES"]}
    RUN_PIPELINE_STEP "10" remove_duplicates ${DIRS["RENAMED_FUSIBLE_SAMPLES"]} ${DIRS["DEDUPED_FUSIBLE_SAMPLES"]}

    deduped_fusible_subgraphs_list="$LEVEL_WORKSPACE/deduplicated_fusible_subgraph_sample_list.txt"
    generate_subgraph_list ${DIRS["DEDUPED_FUSIBLE_SAMPLES"]} $deduped_fusible_subgraphs_list
    RUN_PIPELINE_STEP "11" rewrite_device "$deduped_fusible_subgraphs_list" ${DIRS["DEDUPED_FUSIBLE_SAMPLES"]} ${DIRS["DEVICE_REWRITTEN_SAMPLES"]}

    device_rewrited_subgraphs_list="$LEVEL_WORKSPACE/device_rewrited_fusible_subgraph_sample_list.txt"
    generate_subgraph_list ${DIRS["DEVICE_REWRITTEN_SAMPLES"]} $device_rewrited_subgraphs_list
    RUN_PIPELINE_STEP "12" generate_unittests "$device_rewrited_subgraphs_list" ${DIRS["DEVICE_REWRITTEN_SAMPLES"]} ${DIRS["UNITTESTS"]}
}

main
