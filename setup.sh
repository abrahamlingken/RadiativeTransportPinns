#!/bin/bash
# ================================
# Radiative Transport PINNs 环境设置脚本 (Linux/Mac)
# ================================

echo "正在创建 conda 环境..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo "环境创建失败，请检查错误信息"
    exit 1
fi

echo ""
echo "环境创建成功！"
echo ""
echo "激活环境:"
echo "  conda activate radiative-transport-pinns"
echo ""
echo "运行项目:"
echo "  python PINNS2.py"
echo ""
