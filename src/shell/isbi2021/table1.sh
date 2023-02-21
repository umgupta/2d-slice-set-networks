#!/usr/bin/env sh

# DEFINE
DATA_ROOT_PATH=""
DEVICE="cuda"

################ Table 1 #######################################################
#### commands to train the model with full training data
################################################################################

# 3D-CNN
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 3d_cnn \
  -r result/3d_cnn/ \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_3d.py \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "default"

# 2D-slice model with lstm for aggregation
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 2d_slice_lstm \
  -r result/2d_slice_lstm/ \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_slice_lstm.py \
  --data.root_path "$DATA_ROOT_PATH" \
  --train.gradient_norm_clip 1 --model.arch.initialization "default"

# 2D-slice  model with attention for aggregation
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 2d_slice_attention \
  -r result/2d_slice_attention \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "attention" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom"

# 2D-slice model with mean aggregation
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 2d_slice_mean \
  -r result/2d_slice_mean \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom"

# 2D-slice model with max for aggregation
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 2d_slice_max \
  -r result/2d_slice_max \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "max" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom"
