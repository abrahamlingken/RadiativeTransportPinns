#!/usr/bin/env powershell
# 恢复脚本 - 如果整理后出现问题，从备份恢复

param(
    [Parameter(Mandatory=$true)]
    [string]$BackupDir
)

if (!(Test-Path $BackupDir)) {
    Write-Host "错误: 备份目录 $BackupDir 不存在！" -ForegroundColor Red
    exit 1
}

Write-Host "从 $BackupDir 恢复文件..." -ForegroundColor Yellow

# 恢复所有Python文件
Get-ChildItem -Path $BackupDir -Filter "*.py" | ForEach-Object {
    Copy-Item $_.FullName ".\" -Force
    Write-Host "  恢复: $($_.Name)" -ForegroundColor Gray
}

Write-Host "`n恢复完成！请重新运行整理脚本或手动修复。" -ForegroundColor Green
