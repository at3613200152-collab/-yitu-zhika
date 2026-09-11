param([Parameter(Mandatory=$true)][string]$Pptx)
$ErrorActionPreference = 'Stop'
$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$ppt.DisplayAlerts = 1
$pres = $ppt.Presentations.Open($Pptx, $true, $false, $true)
Write-Host ("opened slides={0}" -f $pres.Slides.Count)
for ($i = 1; $i -le $pres.Slides.Count; $i++) {
  $s = $pres.Slides.Item($i)
  $worst = 0.0
  $worstName = ''
  foreach ($sh in $s.Shapes) {
    if ($sh.HasTextFrame -ne -1) { continue }
    if ($sh.TextFrame.HasText -ne -1) { continue }
    if ($sh.Top -ge 500) { continue }   # footer / page-number boxes live at top=508 by design
    try {
      $r = $sh.TextFrame.TextRange
      $bottom = $r.BoundTop + $r.BoundHeight
      if ($bottom -gt $worst) { $worst = $bottom; $worstName = $sh.Name }
    } catch { }
  }
  $flag = ''
  if ($worst -gt 505) { $flag = '  <== OVERLAPS FOOTER' }
  if ($worst -gt 540) { $flag = '  <== OFF SLIDE' }
  Write-Host ("slide {0,2}: text bottom max = {1,7:F1} ({2}){3}" -f $i, $worst, $worstName, $flag)
}
$pres.Close()
$ppt.Quit()
Write-Host "LINT-OK"
