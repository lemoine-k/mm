# ds_dagm
#sh tools/dist_train.sh configs/bisenetv2/bisenetv2_fcn_2xb4-20k_ds_dagm-512x512.py 2
#sh tools/dist_train.sh configs/unet/unet-s5-d16_fcn_2xb4-20k_ds_dagm-512x512.py 2
#sh tools/dist_train.sh configs/segformer/segformer_mit-b0_2xb4-20k_ds_dagm-512x512.py 2
sh tools/dist_train.sh configs/bisenetv2/bisenetv2_fcn_2xb4-ohem-20k_ds_dagm-512x512.py 2

# ksdd2
#sh tools/dist_train.sh configs/bisenetv2/bisenetv2_fcn_2xb4-20k_ksdd2-512x512.py 2
#sh tools/dist_train.sh configs/unet/unet-s5-d16_fcn_2xb4-20k_ksdd2-512x512.py 2
#sh tools/dist_train.sh configs/segformer/segformer_mit-b0_2xb4-20k_ksdd2-512x512.py 2
#sh tools/dist_train.sh configs/bisenetv2/bisenetv2_fcn_2xb4-ohem-20k_ksdd2-512x512.py 2