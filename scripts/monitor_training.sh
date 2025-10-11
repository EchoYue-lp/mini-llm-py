#!/bin/bash
# 训练监控脚本 - 实时显示训练进度

LOG_FILE=${1:-training.log}

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 日志文件不存在: $LOG_FILE"
    echo ""
    echo "用法: ./scripts/monitor_training.sh [log_file]"
    echo ""
    echo "请先启动训练并记录日志:"
    echo "  python -m scripts.train_encoder_decoder 2>&1 | tee training.log"
    exit 1
fi

clear

echo "=========================================="
echo "   训练监控 - 实时更新中..."
echo "=========================================="
echo ""

# 监控循环
while true; do
    # 移动光标到顶部
    tput cup 4 0

    echo "📊 最新训练状态:"
    echo "----------------------------------------"

    # 获取最后的训练损失
    LAST_TRAIN=$(grep "Train Loss" "$LOG_FILE" | tail -1)
    if [ -n "$LAST_TRAIN" ]; then
        echo "  $LAST_TRAIN"
    fi

    # 获取最后的验证损失
    LAST_VAL=$(grep "Val Loss" "$LOG_FILE" | tail -1)
    if [ -n "$LAST_VAL" ]; then
        echo "  $LAST_VAL"
    fi

    # 获取最佳模型信息
    BEST_MODEL=$(grep "最佳模型已保存" "$LOG_FILE" | tail -1)
    if [ -n "$BEST_MODEL" ]; then
        echo "  ✓ $BEST_MODEL"
    fi

    echo "----------------------------------------"
    echo ""

    # 显示最近5行日志
    echo "📝 最近日志 (最后5行):"
    echo "----------------------------------------"
    tail -5 "$LOG_FILE" | sed 's/^/  /'
    echo "----------------------------------------"
    echo ""

    # 显示当前时间和GPU状态
    echo "⏰ 更新时间: $(date '+%Y-%m-%d %H:%M:%S')"

    # 尝试显示GPU使用情况（如果有nvidia-smi）
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "🎮 GPU 状态:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %s | 使用率: %s%% | 内存: %s/%s MB\n", $1, $2, $3, $4, $5}'
    fi

    echo ""
    echo "按 Ctrl+C 退出监控"

    # 等待5秒更新
    sleep 5
done
