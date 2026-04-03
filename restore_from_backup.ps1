#!/usr/bin/env powershell
# Restore script - Restore files from backup if needed

param(
    [Parameter(Mandatory=$true)]
    [string]$BackupDir
)

if (!(Test-Path $BackupDir)) {
    Write-Host "Error: Backup directory $BackupDir does not exist!" -ForegroundColor Red
    exit 1
}

Write-Host "Restoring from $BackupDir..." -ForegroundColor Yellow

Get-ChildItem -Path $BackupDir -Filter "*.py" | ForEach-Object {
    Copy-Item $_.FullName ".\" -Force
    Write-Host "  Restored: $($_.Name)" -ForegroundColor Gray
}

Write-Host "`nRestore completed!" -ForegroundColor Green
