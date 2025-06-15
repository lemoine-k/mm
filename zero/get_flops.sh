# 计算模型的计算复杂度（FLOPs）和参数量（Params）

#CONFIG_FILE=$1
#INPUT_SHAPE=$2
#
#python tools/analysis_tools/get_flops.py \
#    %CONFIG_FILE \
#    --shape INPUT_SHAPE

python tools/analysis_tools/get_flops.py \
        configs/bisenetv2/bisenetv2_fcn_2xb4-20k_ds_dagm-512x512.py \
        --shape 512 512