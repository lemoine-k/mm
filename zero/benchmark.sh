# 测试模型的推理速度（FPS）和内存占用

#CONFIG_FILE=%1
#CHECKPOINT_FILE=%2
#
#python tools/analysis_tools/benchmark.py \
#    %CONFIG_FILE \
#    %CHECKPOINT_FILE \

python tools/analysis_tools/benchmark.py \
    configs/bisenetv2/bisenetv2_fcn_2xb4-20k_ds_dagm-512x512.py \
    work_dirs/bisenetv2_fcn_2xb4-20k_ds_dagm-512x512/best_mIoU_iter_16000.pth