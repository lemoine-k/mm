# 分析训练日志文件，绘制损失（loss）或评估指标（如 mIoU、mAcc、aAcc）的曲线图

#LOG_JSON=$1
#
#python tools/analysis_tools/analyze_logs.py \
#        %LOG_JSON \
#        --key mIoU \
#        --legend mIoU

python tools/analysis_tools/analyze_logs.py \
        work_dirs/mixbisenetv2_fcn_2xb4-20k_ds_dagm-512x512/20250617_040427/vis_data/20250617_040427.json \
        --key mIoU \
        --legend mIoU