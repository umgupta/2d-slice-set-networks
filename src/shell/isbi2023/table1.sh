#!/usr/bin/env sh

# DEFINE
DATA_ROOT_PATH=""
DEVICE="cuda"

################ Table 1 #######################################################
#### commands to train the model with full training data on ADNI for AD prediction
#### we show the command for 2d-slice set network with mean operation and along
#### sagital axis.
#### for other models just use use the correct data.frame_dim parameter,
#  1 for sagittal (default)
#  2 for coronal
#  3 for axial
################################################################################

################################################################################

# 3D-CNN
python3 -m src.scripts.main -c config/config_adni_ad.py \
  --exp_name 3d_cnn \
  -r result/3d_cnn/ \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/adni/adni_3d.py \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom"

# 2D-slice model with mean aggregation (no pos encoding)
python3 -m src.scripts.main -c config/config_adni_ad.py \
  --exp_name 2d_slice_mean_no_pos_encoding_frame_dim_1 \
  -r result/2d_slice_mean_no_pos_encoding_frame_dim_1 \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/adni/adni_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom" \
  --model.arch.use_position_encoding False --model.arch.encoder_2d "encoder1" \
  --model.arch.load_pretrained_encoder False --data.frame_dim 1

# 2D-slice model with mean aggregation (pos encoding)
python3 -m src.scripts.main -c config/config_adni_ad.py \
  --exp_name 2d_slice_mean_pos_encoding_frame_dim_1 \
  -r result/2d_slice_mean_pos_encoding_frame_dim_1 \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/adni/adni_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom" \
  --model.arch.use_position_encoding True --model.arch.encoder_2d "encoder1" \
  --model.arch.load_pretrained_encoder False --data.frame_dim 1

# 2D-slice model with resnet encoder (no pos encoding & no pretraining)
python3 -m src.scripts.main -c config/config_adni_ad.py \
  --exp_name 2d_slice_mean_resnet_no_pos_encoding_no_pretraining_frame_dim_1 \
  -r result/2d_slice_mean_resnet_no_pos_encoding_no_pretraining_frame_dim_1 \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/adni/adni_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom" \
  --model.arch.use_position_encoding False --model.arch.encoder_2d "resnet18" \
  --model.arch.load_pretrained_encoder False --data.frame_dim 1

# 2D-slice model with resnet encoder (pos encoding & pretraining)
python3 -m src.scripts.main -c config/config_adni_ad.py \
  --exp_name 2d_slice_mean_resnet_pos_encoding_pretraining_frame_dim_1 \
  -r result/2d_slice_mean_resnet_pos_encoding_pretraining_frame_dim_1 \
  --device $DEVICE --wandb.use 0 \
  --model.arch.file src/arch/adni/adni_slice_set.py \
  --model.arch.attn_dim 32 --model.arch.attn_num_heads 1 \
  --model.arch.attn_drop 1 --model.arch.agg_fn "mean" \
  --data.root_path "$DATA_ROOT_PATH" --model.arch.initialization "custom" \
  --model.arch.use_position_encoding True --model.arch.encoder_2d "resnet18" \
  --model.arch.load_pretrained_encoder True --data.frame_dim 1
