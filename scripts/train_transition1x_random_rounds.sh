rounds="0 4 8 12 16 20"
valid_file="data/transition1x/val.xyz"
test_file="data/transition1x/test.xyz"
base_work_root="rets/transition1x-random-rounds-lr1e-2-stage2"

for round in $rounds; do
  training_file="workflow/random/train_round${round}.xyz"

  if [ ! -f "$training_file" ]; then
    echo "Missing training file: $training_file" >&2
    exit 1
  fi

  work_dir="${base_work_root}/round${round}"
  log_dir="${work_dir}/logs"
  model_dir="${work_dir}/model"
  checkpoints_dir="${work_dir}/checkpoints"
  results_dir="${work_dir}/results"
  downloads_dir="${work_dir}/downloads"

  mkdir -p "$log_dir" "$model_dir" "$checkpoints_dir" "$results_dir" "$downloads_dir"

  mace_run_train \
    --name="transition1x_round${round}" \
    --train_file=$training_file \
    --valid_file=$valid_file \
    --test_file=$test_file \
    --work_dir=$work_dir \
    --log_dir=$log_dir \
    --model_dir=$model_dir \
    --checkpoints_dir=$checkpoints_dir \
    --results_dir=$results_dir \
    --downloads_dir=$downloads_dir \
    --forces_weight=10.0 \
    --E0s='average' \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=6.0 \
    --batch_size=64 \
    --valid_batch_size=64 \
    --max_num_epochs=100 \
    --lr=1e-2 \
    --weight_decay=5e-7 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --device=cuda \
    --default_dtype='float32' \
    --scheduler_patience=5 \
    --seed=0 \
    --enable_cueq=True \
    --stage_two \
    --pair_repulsion
done
