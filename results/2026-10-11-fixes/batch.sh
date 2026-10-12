set -x
source /workspace/env.sh
for name in r0_outcome_seed7 r2_gated_seed7 r0_outcome_seed8 r1_full_seed8 r2_gated_seed8; do
  uv run --no-sync python scripts/train_grpo.py --config configs/training/$name.json --model-path "$MODEL" --train-source /workspace/data/hotpot_train_v1.1.json --output-dir /workspace/runs/$name && \
  uv run --no-sync python scripts/evaluate_checkpoint.py --model-path "$MODEL" --adapter-path /workspace/runs/$name/checkpoint-final/adapter --source /workspace/data/hotpot_dev_distractor_v1.json --output-dir /workspace/results/$name --name $name
  echo "RUN_DONE $name exit=$?"
done
echo BATCH_DONE
