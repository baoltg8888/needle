set -x
source /workspace/env.sh
for name in r3_coverage2_seed7 r4_drgrpo_seed7 r3_coverage2_seed8 r4_drgrpo_seed8 r0_outcome_seed9 r3_coverage2_seed9 r4_drgrpo_seed9 r0_dynsample_seed8; do
  uv run --no-sync python scripts/train_grpo.py --config configs/training/$name.json --model-path "$MODEL" --train-source /workspace/data/hotpot_train_v1.1.json --output-dir /workspace/runs/$name && \
  uv run --no-sync python scripts/evaluate_checkpoint.py --model-path "$MODEL" --adapter-path /workspace/runs/$name/checkpoint-final/adapter --source /workspace/data/hotpot_dev_distractor_v1.json --output-dir /workspace/results/$name --name $name
  echo "RUN_DONE $name exit=$?"
done
echo BATCH_DONE
