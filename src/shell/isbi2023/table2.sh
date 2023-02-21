#!/usr/bin/env sh

# DEFINE
DATA_ROOT_PATH=""
DEVICE="cuda"

################ Table 1 #######################################################
#### commands to train the model with full training data on UKBB  for brain age prediction
################################################################################

# 3D-CNN
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 3d_cnn \
  -r result/3d_cnn/ \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_3d.py \
  --train.optimizer sgd --train.lr 1e-4 \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom"


# 2D-slice model with attention aggregation (no pos encoding)
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 2d_slice_attention_no_pos_encoding \
  -r result/2d_slice_attention_no_pos_encoding \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "attention" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom" \
  --model.arch.use_position_encoding False --model.arch.encoder_2d "encoder1" \
  --model.arch.load_pretrained_encoder False

# 2D-slice model with attention aggregation (pos encoding)
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 2d_slice_attention_pos_encoding \
  -r result/2d_slice_attention_pos_encoding \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "attention" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom" \
  --model.arch.use_position_encoding True --model.arch.encoder_2d "encoder1" \
  --model.arch.load_pretrained_encoder False


# 2D-slice model with mean aggregation (no pos encoding)
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 2d_slice_mean_no_pos_encoding \
  -r result/2d_slice_mean_no_pos_encoding \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom" \
  --model.arch.use_position_encoding False --model.arch.encoder_2d "encoder1" \
  --model.arch.load_pretrained_encoder False

# 2D-slice model with mean aggregation (pos encoding)
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 2d_slice_mean_pos_encoding \
  -r result/2d_slice_mean_pos_encoding \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom" \
  --model.arch.use_position_encoding True --model.arch.encoder_2d "encoder1" \
  --model.arch.load_pretrained_encoder False

# 2D-slice model with resnet encoder (no pos encoding & no pretraining)
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 2d_slice_mean_resnet_no_pos_encoding_no_pretraining \
  -r result/2d_slice_mean_resnet_no_pos_encoding_no_pretraining \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom" \
  --model.arch.use_position_encoding False --model.arch.encoder_2d "resnet18" \
  --model.arch.load_pretrained_encoder False

# 2D-slice model with resnet encoder (no pos encoding, but with pretraining)
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 2d_slice_mean_resnet_no_pos_encoding_pretraining \
  -r result/2d_slice_mean_resnet_no_pos_encoding_pretraining \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom" \
  --model.arch.use_position_encoding False --model.arch.encoder_2d "resnet18" \
  --model.arch.load_pretrained_encoder True

# 2D-slice model with resnet encoder (pos encoding & pretraining)
python3 -m src.scripts.main -c config/config_ukbb_brain_age.py \
  --exp_name 2d_slice_mean_resnet_pos_encoding_pretraining \
  -r result/2d_slice_mean_resnet_pos_encoding_pretraining \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/ukbb/brain_age_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom" \
  --model.arch.use_position_encoding True --model.arch.encoder_2d "resnet18" \
  --model.arch.load_pretrained_encoder True