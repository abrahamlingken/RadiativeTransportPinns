#!/usr/bin/env powershell
# Backup script - Run this before organize_project.ps1

$BackupDir = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

Write-Host "Creating backup in: $BackupDir" -ForegroundColor Yellow

# Backup Python files
Copy-Item -Path "*.py" -Destination $BackupDir -Force
Copy-Item -Path "Scripts\*.py" -Destination $BackupDir -Force -ErrorAction SilentlyContinue

Write-Host "Backup completed! If organization fails, restore from $BackupDir" -ForegroundColor Green
