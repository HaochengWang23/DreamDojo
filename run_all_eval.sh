#!/bin/bash
set -euo pipefail

cd /home/featurize/work/DreamDojo

export PATH=/home/featurize/bin:$PATH

CKPT="/home/featurize/models/DreamDojo_2B_GR1_converted/model_ema_bf16.pt"
EXP="cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes_release_oss"
DATA_ROOT="datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1"
RESULT_ROOT="/home/featurize/results/eval_runs"

COMMON_ARGS=(
  --experiment "$EXP"
  --checkpoint-path "$CKPT"
  --num-frames 13
  --num-samples 100
  --data-split full
  --offload-diffusion-model
  --offload-text-encoder
  --offload-tokenizer
  --disable-guardrails
)

run_group () {
  local group_name="$1"
  local group_path="$2"

  echo "======================================"
  echo "Running group: $group_name"
  echo "Dataset root: $group_path"
  echo "======================================"

  mkdir -p "${RESULT_ROOT}/${group_name}"

  for subdir in "${group_path}"/*; do
    [ -d "$subdir" ] || continue

    local dataset_name
    dataset_name=$(basename "$subdir")

    echo "--------------------------------------"
    echo "Running dataset: ${group_name}/${dataset_name}"
    echo "--------------------------------------"

    mkdir -p "${RESULT_ROOT}/${group_name}/${dataset_name}"

    python examples/action_conditioned.py \
      --output-dir "${RESULT_ROOT}/${group_name}/${dataset_name}/runtime" \
      --save-dir "${RESULT_ROOT}/${group_name}/${dataset_name}" \
      --dataset-path "$subdir" \
      "${COMMON_ARGS[@]}" \
      2>&1 | tee "${RESULT_ROOT}/${group_name}/${dataset_name}/run.log"
  done
}

# 1) In-lab_Eval：逐个子任务跑
run_group "In-lab_Eval" "${DATA_ROOT}/In-lab_Eval"

# 2) EgoDex_Eval：如果里面有多个子目录，也逐个跑
run_group "EgoDex_Eval" "${DATA_ROOT}/EgoDex_Eval"

# 3) DreamDojo-HV_Eval：如果里面有多个子目录，也逐个跑
run_group "DreamDojo-HV_Eval" "${DATA_ROOT}/DreamDojo-HV_Eval"

echo "======================================"
echo "All evaluation jobs finished."
echo "Results saved to: ${RESULT_ROOT}"
echo "======================================"
