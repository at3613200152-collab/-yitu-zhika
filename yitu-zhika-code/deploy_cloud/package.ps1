$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stage = Join-Path ([IO.Path]::GetTempPath()) ('yitu-cloud-' + $stamp)
$out = Join-Path ([Environment]::GetFolderPath('Desktop')) ('yitu-cloud-lite-' + $stamp + '.zip')
New-Item -ItemType Directory -Path $stage | Out-Null
function Copy-RuntimeFile($source, $relative) {
    if (!(Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing runtime file: $relative" }
    $dest = Join-Path $stage $relative
    New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $dest
}
foreach ($folder in @('app','src','scripts','recipe')) {
    Get-ChildItem -LiteralPath (Join-Path $repo $folder) -Recurse -File -Filter '*.py' | ForEach-Object {
        Copy-RuntimeFile $_.FullName ([IO.Path]::GetRelativePath($repo, $_.FullName))
    }
}
foreach ($model in @()) {
    Copy-RuntimeFile (Join-Path $repo "checkpoints/$model/best.pt") "checkpoints/$model/best.pt"
    foreach ($file in @('test_metrics.json','protocol.json','test_predictions.csv')) {
        Copy-RuntimeFile (Join-Path $repo "results/$model/$file") "results/$model/$file"
    }
    $audit = if ($model -eq 'calorieclip_official_v1') {'completion_audit.json'} else {'paired_completion_audit.json'}
    Copy-RuntimeFile (Join-Path $repo "results/$model/$audit") "results/$model/$audit"
}
foreach ($tag in @('v1_expanded','v3_corrected_seed42')) {
    Copy-RuntimeFile (Join-Path $repo "checkpoints/meal_macros_v1/$tag/best.pt") "checkpoints/meal_macros_v1/$tag/best.pt"
}
foreach ($name in @('meal_official_v1','meal_macros_expanded_v1','meal_macros_corrected_v2')) {
    Copy-RuntimeFile (Join-Path $repo "results/$name/manifest.json") "results/$name/manifest.json"
}
Copy-RuntimeFile (Join-Path $repo 'data/nutrition5k/dishes_verified.csv') 'data/nutrition5k/dishes_verified.csv'
Copy-RuntimeFile (Join-Path $env:USERPROFILE '.cache/torch/hub/checkpoints/resnet50-0676ba61.pth') 'torch_cache/hub/checkpoints/resnet50-0676ba61.pth'
Copy-RuntimeFile (Join-Path $PSScriptRoot 'Dockerfile') 'Dockerfile'
Copy-RuntimeFile (Join-Path $PSScriptRoot 'cloud_entry.py') 'cloud_entry.py'
& 'C:\Users\user\miniconda3\envs\yitu\python.exe' (Join-Path $PSScriptRoot 'write_runtime_hashes.py') $stage
if ($LASTEXITCODE -ne 0) { throw 'Runtime manifest generation failed' }
New-Item -ItemType Directory -Path (Join-Path $stage 'logs') -Force | Out-Null
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $out -CompressionLevel Fastest
Write-Output "STAGE=$stage"
Write-Output "ZIP=$out"
Write-Output "BYTES=$((Get-Item -LiteralPath $out).Length)"
