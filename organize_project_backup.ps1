#!/usr/bin/env powershell
# 备份脚本 - 在执行organize_project.ps1之前运行

$BackupDir = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Write-Host "创建备份目录: $BackupDir" -ForegroundColor Yellow

# 创建备份
Copy-Item -Path ".\*.py" -Destination $BackupDir -Recurse -Force
Copy-Item -Path ".\Scripts\*.py" -Destination $BackupDir -Recurse -Force

Write-Host "备份完成！如果整理后出现问题，可以从 $BackupDir 恢复文件。" -ForegroundColor Green
