# 分析训练日志文件，绘制损失（loss）或评估指标（如 mIoU、mAcc、aAcc）的曲线图

LOG_JSON=$1

python tools/analysis_tools/analyze_logs.py \
        %LOG_JSON \
        --key mIoU \
        --legend mIoU