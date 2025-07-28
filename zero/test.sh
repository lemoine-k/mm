#sh tools/dist_train.sh configs/unet/unet-s5-d16_fcn_2xb4-20k_neu_seg-224x224.py 2
#sh tools/dist_train.sh configs/pspnet/pspnet_r50-d8_2xb4-20k_neu_seg-224x224.py 2
#sh tools/dist_train.sh configs/deeplabv3/deeplabv3_r50-d8_2xb4-20k_neu_seg-224x224.py 2
#sh tools/dist_train.sh configs/deeplabv3plus/deeplabv3plus_r101-d8_2xb4-20k_neu_seg-224x224.py 2
#sh tools/dist_train.sh configs/segformer/segformer_mit-b0_2xb4-20k_neu_seg-224x224.py 2
#sh tools/dist_train.sh configs/bisenetv2/bisenetv2_fcn_2xb4-20k_neu_seg-224x224.py 2
#sh tools/dist_train.sh configs/kynet/kynet_upernet_2xb4-20k_neu_seg-224x224.py 2

sh tools/dist_test.sh \
    configs/kynet/kynet_upernet_2xb4-20k_neu_seg-224x224.py\
    work_dirs/kynet_upernet_2xb4-20k_neu_seg-224x224/best_mIoU_iter_11200.pth \
    2 \
#    --out outputs/kynet_upernet_2xb4-20k_neu_seg-224x224 \
    --work-dir work_dirs/kynet_upernet_2xb4-20k_neu_seg-224x224/test
