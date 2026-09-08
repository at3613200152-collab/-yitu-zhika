# Overnight training batch: S0 -> S1 -> S3
$ErrorActionPreference = "Continue"
$python = "C:\Users\user\miniconda3\envs\yitu\python.exe"
$scriptDir = $PSScriptRoot
$script = Join-Path $scriptDir "train_meal_seed.py"
$outRoot = Join-Path $scriptDir "experiments"
$backbone = "C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code\checkpoints\food101_pretrained\backbone_only.pt"
$logFile = Join-Path $outRoot "overnight_log.txt"

# Ensure output dir exists
New-Item -ItemType Directory -Path $outRoot -Force | Out-Null

$runs = @(
    @("s0", "123", "", ""),
    @("s0", "7",   "", ""),
    @("s1", "42",  $backbone, ""),
    @("s1", "123", $backbone, ""),
    @("s1", "7",   $backbone, ""),
    @("s3", "42",  "", "--augment --mixup 0.2"),
    @("s3", "123", "", "--augment --mixup 0.2"),
    @("s3", "7",   "", "--augment --mixup 0.2")
)

$startMsg = "=== Overnight training started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host $startMsg
Add-Content -Path $logFile -Value $startMsg -Encoding UTF8

foreach ($run in $runs) {
    $tag = $run[0]
    $seed = $run[1]
    $bb = $run[2]
    $extra = $run[3]

    $startRun = "`n--- Starting $tag seed=$seed at $(Get-Date -Format 'HH:mm:ss') ---"
    Write-Host $startRun
    Add-Content -Path $logFile -Value $startRun -Encoding UTF8

    $argList = @($script, "--tag", $tag, "--seed", $seed, "--epochs", "30", "--out_root", $outRoot)
    if ($bb) { $argList += @("--backbone", $bb) }
    if ($extra) { $argList += $extra -split ' ' }

    & $python @argList 2>&1 | ForEach-Object {
        Write-Host $_
        Add-Content -Path $logFile -Value $_ -Encoding UTF8
    }

    $endRun = "--- Finished $tag seed=$seed at $(Get-Date -Format 'HH:mm:ss') (exit=$LASTEXITCODE) ---"
    Write-Host $endRun
    Add-Content -Path $logFile -Value $endRun -Encoding UTF8
}

$endMsg = "`n=== Overnight training finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host $endMsg
Add-Content -Path $logFile -Value $endMsg -Encoding UTF8
