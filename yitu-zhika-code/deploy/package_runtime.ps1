# 打包运行时（Windows 本机执行）：把可运行/可上传的内容打成 zip，不含 venv/日志/.env
# 输出：桌面\yitu-zhika-code-runtime.zip（约 1.5GB，含模型权重+审计文件+营养库）
$ErrorActionPreference = 'Stop'
$src = Join-Path $PSScriptRoot '..'   # yitu-zhika-code
$src = [System.IO.Path]::GetFullPath($src)
$out = Join-Path ([Environment]::GetFolderPath('Desktop')) 'yitu-zhika-code-runtime.zip'
if (Test-Path $out) { Remove-Item $out -Force }

$exclude = @('venv','__pycache__','logs','.git','.venv')
$items = Get-ChildItem $src -Force | Where-Object { $exclude -notcontains $_.Name }
Compress-Archive -Path ($items | ForEach-Object { $_.FullName }) -DestinationPath $out -CompressionLevel Optimal
Write-Host "打包完成: $out ($([math]::Round((Get-Item $out).Length/1MB,1)) MB)"
Write-Host "上传到服务器: scp -r \"$out\" user@服务器IP:/home/user/"
