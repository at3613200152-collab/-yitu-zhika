# S3 re-run: augmentation + mixup (after bug fix)
$ErrorActionPreference = "Continue"
$python = "C:\Users\user\miniconda3\envs\yitu\python.exe"
$scriptDir = $PSScriptRoot
$script = Join-Path $scriptDir "train_meal_seed.py"
$outRoot = Join-Path $scriptDir "experiments"
$logFile = Join-Path $outRoot "s3_rerun_log.txt"

New-Item -ItemType Directory -Path $outRoot -Force | Out-Null

$runs = @(
    @("s3", "42",  "--augment --mixup 0.2"),
    @("s3", "123", "--augment --mixup 0.2"),
    @("s3", "7",   "--augment --mixup 0.2")
)

$startMsg = "=== S3 re-run started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host $startMsg
Add-Content -Path $logFile -Value $startMsg -Encoding UTF8

foreach ($run in $runs) {
    $tag = $run[0]
    $seed = $run[1]
    $extra = $run[2]

    $startRun = "`n--- Starting $tag seed=$seed at $(Get-Date -Format 'HH:mm:ss') ---"
    Write-Host $startRun
    Add-Content -Path $logFile -Value $startRun -Encoding UTF8

    $argList = @($script, "--tag", $tag, "--seed", $seed, "--epochs", "30", "--out_root", $outRoot)
    if ($extra) { $argList += $extra -split ' ' }

    & $python @argList 2>&1 | ForEach-Object {
        Write-Host $_
        Add-Content -Path $logFile -Value $_ -Encoding UTF8
    }

    $endRun = "--- Finished $tag seed=$seed at $(Get-Date -Format 'HH:mm:ss') (exit=$LASTEXITCODE) ---"
    Write-Host $endRun
    Add-Content -Path $logFile -Value $endRun -Encoding UTF8
}

$endMsg = "`n=== S3 re-run finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host $endMsg
Add-Content -Path $logFile -Value $endMsg -Encoding UTF8
