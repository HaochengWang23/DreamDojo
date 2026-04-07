#!/bin/bash
set -euo pipefail

cd /home/featurize/work/DreamDojo
export PATH=/home/featurize/bin:$PATH

CKPT="/home/featurize/models/DreamDojo_2B_GR1_converted/model_ema_bf16.pt"
EXP="cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes_release_oss"
DATA_ROOT="datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1"
RESULT_ROOT="/home/featurize/results/eval_runs_49"

COMMON_ARGS=(
  --experiment "$EXP"
  --checkpoint-path "$CKPT"
  --num-frames 49
  --num-samples 100
  --data-split full
  --offload-diffusion-model
  --offload-text-encoder
  --offload-tokenizer
  --disable-guardrails
)

run_single_dataset () {
  local group_name="$1"
  local dataset_name="$2"
  local dataset_path="$3"

  local out_dir="${RESULT_ROOT}/${group_name}/${dataset_name}"

  echo "======================================"
  echo "Running dataset: ${group_name}/${dataset_name}"
  echo "Dataset path: ${dataset_path}"
  echo "Output dir: ${out_dir}"
  echo "======================================"

  # 清空旧结果，避免因为已有 pred.mp4 被自动 skip
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  # python examples/action_conditioned.py \
  #   --save-dir "${out_dir}" \
  #   --dataset-path "${dataset_path}" \
  #   "${COMMON_ARGS[@]}" \
  #   2>&1 | tee "${out_dir}/run.log"

  if python examples/action_conditioned.py \
    --output-dir "${out_dir}" \
    --dataset-path "${dataset_path}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${out_dir}/run.log"; then
    echo "[OK] ${group_name}/${dataset_name}" | tee -a "${RESULT_ROOT}/master_status.log"
  else
    echo "[FAILED] ${group_name}/${dataset_name}" | tee -a "${RESULT_ROOT}/master_status.log"
  fi
}

run_inlab_group () {
  local group_name="In-lab_Eval"
  local group_path="${DATA_ROOT}/In-lab_Eval"

  mkdir -p "${RESULT_ROOT}/${group_name}"

  for subdir in "${group_path}"/*; do
    [ -d "$subdir" ] || continue
    local dataset_name
    dataset_name=$(basename "$subdir")
    run_single_dataset "${group_name}" "${dataset_name}" "$subdir"
  done
}

run_whole_group () {
  local group_name="$1"
  local group_path="$2"
  run_single_dataset "${group_name}" "${group_name}" "${group_path}"
}

echo "Starting full evaluation..."

# 1) In-lab：逐个子数据集
run_inlab_group

# 2) EgoDex：整个目录作为一个 dataset
run_whole_group "EgoDex_Eval" "${DATA_ROOT}/EgoDex_Eval"

# 3) DreamDojo-HV：整个目录作为一个 dataset
run_whole_group "DreamDojo-HV_Eval" "${DATA_ROOT}/DreamDojo-HV_Eval"

echo "======================================"
echo "All evaluation jobs finished."
echo "Results saved to: ${RESULT_ROOT}"
echo "======================================"

# 自动汇总
python summarize_eval.py | tee "${RESULT_ROOT}/summary.log"