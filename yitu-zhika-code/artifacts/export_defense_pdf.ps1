param(
  [Parameter(Mandatory=$true)][string]$Pptx,
  [Parameter(Mandatory=$true)][string]$PdfOut,
  [Parameter(Mandatory=$true)][string]$BackupDir
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $BackupDir)) { New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null }

if (Test-Path -LiteralPath $PdfOut) {
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $bak = Join-Path $BackupDir ("_desktop_pdf_backup_{0}.pdf" -f $stamp)
  Copy-Item -LiteralPath $PdfOut -Destination $bak -Force
  Write-Output ("backed up old pdf -> {0} ({1} bytes)" -f $bak, (Get-Item -LiteralPath $bak).Length)
}

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$ppt.DisplayAlerts = 1
$pres = $ppt.Presentations.Open($Pptx, $true, $false, $true)
Write-Output ("opened slides={0}" -f $pres.Slides.Count)
# ppSaveAsPDF = 32
$pres.SaveAs($PdfOut, 32)
$pres.Close()
$ppt.Quit()

$fi = Get-Item -LiteralPath $PdfOut
$head = [System.IO.File]::ReadAllBytes($PdfOut)[0..4]
$magic = -join ($head | ForEach-Object { [char]$_ })
Write-Output ("PDF size={0} magic={1}" -f $fi.Length, $magic)
