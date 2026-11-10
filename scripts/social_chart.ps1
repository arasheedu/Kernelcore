<#
.SYNOPSIS
  Draws a simple portrait chart for social media (1080 x 1350) from bench_chart\results.json:
  Kernelcore's speed relative to llama.cpp's CPU backend, four headline numbers on one axis.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\social_chart.ps1
#>
param(
    [string]$Results = (Join-Path (Split-Path $PSScriptRoot -Parent) 'bench_chart\results.json'),
    [string]$Out = (Join-Path (Split-Path $PSScriptRoot -Parent) 'bench_chart\social_chart.png')
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$doc = Get-Content -Raw $Results | ConvertFrom-Json

function Best([string]$backend, [string]$test) {
    $v = @($doc.records | Where-Object { $_.backend -eq $backend -and $_.test -eq $test } | ForEach-Object { [double]$_.avg_ts })
    if (-not $v.Count) { throw "no result for $backend / $test" }
    ($v | Measure-Object -Maximum).Maximum
}
$x = [string][char]0x00D7   # multiplication sign, kept out of the file's bytes (Windows PowerShell reads ANSI)
$dot = "  $([char]0x00B7)  "
$chatName = if ($doc.chat -and $doc.chat.name) { $doc.chat.name -replace '^qwen3:8b$', 'Qwen3 8B' } else { 'chat model' }
$pp = @($doc.records | Where-Object { $_.backend -eq 'cpu' -and $_.test -like 'pp*' } | ForEach-Object { [int]$_.test.Substring(2) } | Sort-Object -Unique)[-1]
$rows = @(
    @{ model = 'Llama 3.2 3B'; task = 'writing a reply'; test = 'tg64' },
    @{ model = 'Llama 3.2 3B'; task = "reading a $pp-token prompt"; test = "pp$pp" },
    @{ model = "$chatName chat"; task = 'writing the answer'; test = 'chat reply' },
    @{ model = "$chatName chat"; task = 'reading the question'; test = 'chat prompt' }
)
foreach ($r in $rows) { $r.cpu = Best 'cpu' $r.test; $r.kc = Best 'Kernelcore' $r.test; $r.ratio = $r.kc / $r.cpu }
$gpuTg = $null
if (@($doc.records | Where-Object { $_.backend -eq 'gpu' -and $_.test -eq 'tg64' }).Count) { $gpuTg = Best 'gpu' 'tg64' }

# ---- canvas ----
$W = 1080; $H = 1350; $M = 64
$bmp = New-Object System.Drawing.Bitmap($W, $H)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = 'AntiAlias'; $g.TextRenderingHint = 'ClearTypeGridFit'; $g.PixelOffsetMode = 'HighQuality'
function C([string]$hex) { [System.Drawing.ColorTranslator]::FromHtml($hex) }
function B([string]$hex) { New-Object System.Drawing.SolidBrush((C $hex)) }
function F([float]$size, [string]$face = 'Segoe UI') { New-Object System.Drawing.Font($face, $size, [System.Drawing.GraphicsUnit]::Pixel) }
$g.Clear((C '#fbfbfa'))
$ink = B '#1d1d1f'; $ink2 = B '#5f6368'; $ink3 = B '#8a8f98'
$cCpu = C '#2a78d6'; $cKc = C '#eb6834'          # the same colors as bench_chart.png
$fTitle = F 50 'Segoe UI Semibold'; $fSub = F 25; $fModel = F 29 'Segoe UI Semibold'; $fTask = F 26
$fVal = F 27 'Segoe UI Semibold'; $fVal2 = F 23; $fLeg = F 25; $fFoot = F 20; $fTick = F 21

# bar with a 4 px rounded data end, square at the baseline
function Bar([float]$bx, [float]$by, [float]$bw, [float]$bh, $color) {
    $r = 4; $p = New-Object System.Drawing.Drawing2D.GraphicsPath
    $p.AddLine($bx, $by, $bx + $bw - $r, $by)
    $p.AddArc($bx + $bw - 2 * $r, $by, 2 * $r, 2 * $r, 270, 90)
    $p.AddArc($bx + $bw - 2 * $r, $by + $bh - 2 * $r, 2 * $r, 2 * $r, 0, 90)
    $p.AddLine($bx + $bw - $r, $by + $bh, $bx, $by + $bh)
    $p.CloseFigure()
    $g.FillPath((New-Object System.Drawing.SolidBrush($color)), $p); $p.Dispose()
}

# title and subtitle: the finding first
$g.DrawString('Same CPU, faster LLMs', $fTitle, $ink, $M - 4, 64)
$g.DrawString("Kernelcore vs llama.cpp's own CPU backend$($dot)Intel i5-13400F, no GPU", $fSub, $ink2, $M, 136)
$g.DrawString("Speed relative to the CPU backend (1.00$x, dashed line); higher is better", $fSub, $ink2, $M, 172)

# legend
$ly = 238
Bar $M ($ly + 6) 22 22 $cCpu
$g.DrawString("llama.cpp CPU backend", $fLeg, $ink, $M + 32, $ly)
$lx2 = $M + 32 + $g.MeasureString("llama.cpp CPU backend", $fLeg).Width + 36
Bar $lx2 ($ly + 6) 22 22 $cKc
$g.DrawString("Kernelcore (CUDA C kernels on the CPU cores)", $fLeg, $ink, $lx2 + 32, $ly)

# rows on one axis: 1x = the CPU backend
$plotX = $M; $maxR = [math]::Max(2.0, (($rows | ForEach-Object { $_.ratio }) | Measure-Object -Maximum).Maximum)
$plotW = 610; $scale = $plotW / $maxR
$top = 316; $rowH = 222; $barH = 46; $gap = 10
$plotTop = $top + 78; $plotBottom = $top + 3 * $rowH + 78 + 2 * $barH + $gap
$grid = New-Object System.Drawing.Pen((C '#3c4043'), 2)
$grid.DashStyle = 'Dash'
$i = 0
foreach ($r in $rows) {
    $y = $top + $i * $rowH
    $g.DrawString($r.model, $fModel, $ink, $M - 2, $y)
    $mw = $g.MeasureString($r.model, $fModel).Width
    $g.DrawString($r.task, $fTask, $ink2, $M + $mw - 4, $y + 3)
    $by = $y + 78
    Bar $plotX $by (1.0 * $scale) $barH $cCpu
    $g.DrawString(("1.00$x"), $fVal, $ink, $plotX + $scale + 14, $by + 5)
    $g.DrawString(("{0:N1} tok/s" -f $r.cpu), $fVal2, $ink2, $plotX + $scale + 14 + $g.MeasureString("1.00$x", $fVal).Width + 4, $by + 8)
    $by2 = $by + $barH + $gap
    $bw = $r.ratio * $scale
    Bar $plotX $by2 $bw $barH $cKc
    # 1x baseline (the CPU backend), drawn over the bars so it shows where Kernelcore passes it
    $g.DrawLine($grid, $plotX + $scale, $by - 6, $plotX + $scale, $by2 + $barH + 6)
    $rl = "{0:N2}$x" -f $r.ratio
    $g.DrawString($rl, $fVal, $ink, $plotX + $bw + 14, $by2 + 5)
    $g.DrawString(("{0:N1} tok/s" -f $r.kc), $fVal2, $ink2, $plotX + $bw + 14 + $g.MeasureString($rl, $fVal).Width + 4, $by2 + 8)
    $i++
}

# footer
$fy = $H - 150
$g.DrawLine((New-Object System.Drawing.Pen((C '#e3e3e0'), 2)), $M, $fy, $W - $M, $fy)
$g.DrawString("Best of 2 benchmark rounds (chat: best of 6 requests).", $fFoot, $ink2, $M, $fy + 16)
if ($gpuTg) { $g.DrawString(("For scale: an RTX 4060 GPU writes {0:N0} tok/s on the same 3B model." -f $gpuTg), $fFoot, $ink2, $M, $fy + 44) }
$g.DrawString("Kernelcore$($dot)github.com/arasheedu/Kernelcore", (F 22 'Segoe UI Semibold'), $ink, $M, $fy + 84)
$g.Dispose()
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
foreach ($r in $rows) { '{0,-14} {1,-28} CPU {2,7:N1}  Kernelcore {3,7:N1}  {4:N2}x' -f $r.model, $r.task, $r.cpu, $r.kc, $r.ratio }
"wrote $Out"
