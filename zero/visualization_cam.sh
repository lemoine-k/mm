# 用于可视化分割模型对特定类别的关注区域

#IMG_FILE=$1
#CONFIG_FILE=$2
#CHECKPOINT_FILE=$3
#TARGET_LAYERS=$4
#CATEGORY_INDEX=$5
#
#python tools/analysis_tools/visualization_cam.py \
#    %IMG_FILE \
#    %CONFIG_FILE \
#    %CHECKPOINT_FILE \
#    --target-layers %TARGET_LAYERS \
#    --category-index CATEGORY_INDEX

python tools/analysis_tools/visualization_cam.py \
        data/ds_dagm/images/test/226.png \
        configs/aaaaa/segformer-defect-mitb0-2xb4-20k_ds_dagm-512x512.py \
        work_dirs/segformer-defect-mitb0-2xb4-20k_ds_dagm-512x512/best_mIoU_iter_10000.pth \
        --target-layers backbone.detail.detail_branch[1][-1].activate \
        --category-index 3

