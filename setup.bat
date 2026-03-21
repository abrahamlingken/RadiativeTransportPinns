@echo off
REM ================================
REM Radiative Transport PINNs 环境设置脚本
REM ================================

echo 正在创建 conda 环境...
call conda env create -f environment.yml

if %errorlevel% neq 0 (
    echo 环境创建失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo 环境创建成功！
echo.
echo 激活环境:
echo   conda activate radiative-transport-pinns
echo.
echo 运行项目:
echo   python PINNS2.py
echo.
pause
