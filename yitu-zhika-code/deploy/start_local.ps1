# Windows 本地启动一图知卡后端（上线前的本地演练；与云端同代码）
# 用法：双击运行，或在此目录下执行  powershell -ExecutionPolicy Bypass -File start_local.ps1
# 停止：Ctrl+C
$ErrorActionPreference = 'Stop'
$PY = "C:\Users\user\miniconda3\envs\yitu\python.exe"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:Path = "C:\Users\user\miniconda3\envs\yitu;C:\Users\user\miniconda3\envs\yitu\Library\bin;" + $env:Path
Set-Location "$PSScriptRoot\.."

Write-Host "启动本地服务 -> http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "浏览器/小程序也请访问本机局域网IP（如 http://10.21.202.139:8000）。Ctrl+C 停止。"
& $PY app\inference_service.py --port 8000
