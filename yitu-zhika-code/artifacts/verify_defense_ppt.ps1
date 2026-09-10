param(
  [Parameter(Mandatory=$true)][string]$Pptx
)
# 注意：Presentations.Open 的第四个参数 WithWindow 必须为 msoTrue(-1)。
# 传 msoFalse 时 PowerPoint 会以 "PowerPoint could not open the file." 拒绝打开
# 任何带备注页的演示文稿（与文件本身是否有效无关）。
$ErrorActionPreference = 'Stop'
$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$ppt.DisplayAlerts = 1
$pres = $ppt.Presentations.Open($Pptx, $true, $false, $true)
Write-Host ("OPENED slides={0} w={1} h={2}" -f $pres.Slides.Count, $pres.PageSetup.SlideWidth, $pres.PageSetup.SlideHeight)
for ($i = 1; $i -le $pres.Slides.Count; $i++) {
  $s = $pres.Slides.Item($i)
  $pics = 0
  $tbls = 0
  foreach ($sh in $s.Shapes) {
    if ($sh.Type -eq 13) { $pics++ }
    if ($sh.HasTable -eq -1) { $tbls++ }
  }
  $notesLen = 0
  try { $notesLen = $s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text.Length } catch { $notesLen = -1 }
  Write-Host ("slide {0}: shapes={1} pictures={2} tables={3} notes_chars={4}" -f $i, $s.Shapes.Count, $pics, $tbls, $notesLen)
}
$pres.Close()
$ppt.Quit()
Write-Host "VERIFY-OK"
