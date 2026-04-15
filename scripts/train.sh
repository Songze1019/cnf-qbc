training_file="src/nacl/data/train512.xyz"
test_file="src/nacl/data/test512.xyz"
checkpoints_dir="src/nacl/checkpoints"

mace_run_train \
  --name="alldata" \
  --train_file=$training_file \
  --valid_file=$test_file \
  --test_file=$test_file \
  --forces_weight=10.0 \
  --E0s='average' \
  --hidden_irreps='128x0e + 128x1o' \
  --r_max=6.0 \
  --batch_size=16 \
  --max_num_epochs=50 \
  --ema \
  --ema_decay=0.99 \
  --amsgrad \
  --device=cuda \
  --default_dtype='float32' \
  --scheduler_patience=5 \
  --checkpoints_dir=$checkpoints_dir \
  --seed=0 \
  --enable_cueq=True \
  --pair_repulsion \
  # --model="MACELES" \
