<#
.SYNOPSIS
  Draws a simple portrait chart for social media (1080 x 1350) from bench_chart\results.json:
  Kernelcore's speed relative to llama.cpp's CPU backend, four headline numbers on one axis,
  with each test's setup, the machine and models, and how Kernelcore works.

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

function Records([string]$backend, [string]$test) { @($doc.records | Where-Object { $_.backend -eq $backend -and $_.test -eq $test }) }
function Best([string]$backend, [string]$test) {
    $v = @(Records $backend $test | ForEach-Object { [double]$_.avg_ts })
    if (-not $v.Count) { throw "no result for $backend / $test" }
    ($v | Measure-Object -Maximum).Maximum
}
# kept out of the file's bytes: Windows PowerShell reads scripts as ANSI
$x = [string][char]0x00D7; $dot = "  $([char]0x00B7)  "; $q1 = [string][char]0x201C; $q2 = [string][char]0x201D

$set = $doc.settings; $sys = $doc.system
$llama = $doc.meta.cpu
$llamaQuant = if ($llama.model_type -match 'Q4_K - Medium') { 'Q4_K_M' } else { ($llama.model_type -split ' ', 3)[2] }
$llamaGiB = [math]::Round([double]$llama.model_size / 1GB, 2)
$chatName = if ($doc.chat.name -eq 'qwen3:8b') { 'Qwen3 8B' } else { $doc.chat.name }
$chatGiB = [math]::Round([double]$doc.chat.size / 1GB, 2)
$chatRec = (Records 'Kernelcore' 'chat reply' | Select-Object -First 1)
$gen = [int]$set.gen
$pp = @($doc.records | Where-Object { $_.backend -eq 'cpu' -and $_.test -like 'pp*' } | ForEach-Object { [int]$_.test.Substring(2) } | Sort-Object -Unique)[-1]
$rows = @(
    @{ model = 'Llama 3.2 3B'; task = 'writing a reply'; test = "tg$gen"
       note = "$gen new tokens, one at a time (llama-bench tg$gen)" },
    @{ model = 'Llama 3.2 3B'; task = "reading a $pp-token prompt"; test = "pp$pp"
       note = "$pp tokens in one batch (llama-bench pp$pp)" },
    @{ model = "$chatName chat"; task = 'writing the answer'; test = 'chat reply'
       note = "$($chatRec.predicted_n) tokens, greedy, no thinking (llama-server)" },
    @{ model = "$chatName chat"; task = 'reading the question'; test = 'chat prompt'
       note = "$q1$($doc.chat.prompt)$q2 in the chat template, $($chatRec.prompt_n) tokens" }
)
foreach ($r in $rows) { $r.cpu = Best 'cpu' $r.test; $r.kc = Best 'Kernelcore' $r.test; $r.ratio = $r.kc / $r.cpu }
$gpuTg = $null
if ((Records 'gpu' "tg$gen").Count) { $gpuTg = Best 'gpu' "tg$gen" }

# ---- canvas ----
$W = 1080; $H = 1350; $M = 64
$bmp = New-Object System.Drawing.Bitmap($W, $H)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = 'AntiAlias'; $g.TextRenderingHint = 'ClearTypeGridFit'; $g.PixelOffsetMode = 'HighQuality'
function C([string]$hex) { [System.Drawing.ColorTranslator]::FromHtml($hex) }
function B([string]$hex) { New-Object System.Drawing.SolidBrush((C $hex)) }
function F([float]$size, [string]$face = 'Segoe UI') { New-Object System.Drawing.Font($face, $size, [System.Drawing.GraphicsUnit]::Pixel) }
$g.Clear((C '#fbfbfa'))
$ink = B '#1d1d1f'; $ink2 = B '#5f6368'; $ink3 = B '#80868b'
$cCpu = C '#2a78d6'; $cKc = C '#eb6834'          # the same colors as bench_chart.png
$fTitle = F 50 'Segoe UI Semibold'; $fSub = F 24; $fModel = F 28 'Segoe UI Semibold'; $fTask = F 25; $fNote = F 20
$fVal = F 26 'Segoe UI Semibold'; $fVal2 = F 22; $fLeg = F 24; $fHead = F 20 'Segoe UI Semibold'; $fFoot = F 19

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

# title: the finding first
$g.DrawString('Same CPU, faster LLMs', $fTitle, $ink, $M - 4, 52)
$g.DrawString("Kernelcore vs llama.cpp's own CPU backend$($dot)Intel i5-13400F, no GPU", $fSub, $ink2, $M, 122)
$g.DrawString("Speed relative to the CPU backend (1.00$x, dashed line); higher is better", $fSub, $ink2, $M, 156)

# legend
$ly = 212
Bar $M ($ly + 5) 22 22 $cCpu
$g.DrawString('llama.cpp CPU backend', $fLeg, $ink, $M + 32, $ly)
$lx2 = $M + 32 + $g.MeasureString('llama.cpp CPU backend', $fLeg).Width + 36
Bar $lx2 ($ly + 5) 22 22 $cKc
$g.DrawString('Kernelcore (CUDA C kernels on the CPU cores)', $fLeg, $ink, $lx2 + 32, $ly)

# rows on one axis: 1x = the CPU backend
$plotX = $M; $maxR = [math]::Max(2.0, (($rows | ForEach-Object { $_.ratio }) | Measure-Object -Maximum).Maximum)
$plotW = 610; $scale = $plotW / $maxR
$top = 276; $rowH = 196; $barH = 40; $gap = 8
$grid = New-Object System.Drawing.Pen((C '#3c4043'), 2)
$grid.DashStyle = 'Dash'
$i = 0
foreach ($r in $rows) {
    $y = $top + $i * $rowH
    $g.DrawString($r.model, $fModel, $ink, $M - 2, $y)
    $mw = $g.MeasureString($r.model, $fModel).Width
    $g.DrawString($r.task, $fTask, $ink2, $M + $mw - 4, $y + 3)
    $g.DrawString($r.note, $fNote, $ink3, $M, $y + 40)
    $by = $y + 78
    Bar $plotX $by (1.0 * $scale) $barH $cCpu
    $g.DrawString("1.00$x", $fVal, $ink, $plotX + $scale + 14, $by + 3)
    $g.DrawString(("{0:N1} tok/s" -f $r.cpu), $fVal2, $ink2, $plotX + $scale + 14 + $g.MeasureString("1.00$x", $fVal).Width + 4, $by + 6)
    $by2 = $by + $barH + $gap
    $bw = $r.ratio * $scale
    Bar $plotX $by2 $bw $barH $cKc
    # 1x baseline (the CPU backend), drawn over the bars so it shows where Kernelcore passes it
    $g.DrawLine($grid, $plotX + $scale, $by - 6, $plotX + $scale, $by2 + $barH + 6)
    $rl = "{0:N2}$x" -f $r.ratio
    $g.DrawString($rl, $fVal, $ink, $plotX + $bw + 14, $by2 + 3)
    $g.DrawString(("{0:N1} tok/s" -f $r.kc), $fVal2, $ink2, $plotX + $bw + 14 + $g.MeasureString($rl, $fVal).Width + 4, $by2 + 6)
    $i++
}

# setup block: how it works, machine and models, method
$fy = $top + 4 * $rowH + 6
$g.DrawLine((New-Object System.Drawing.Pen((C '#e3e3e0'), 2)), $M, $fy, $W - $M, $fy)
$lines = @(
    @('How', "Kernelcore runs llama.cpp's layers as unmodified CUDA C kernels, compiled for the CPU by the open-source Booth compiler. The same .cu files also build with nvcc for NVIDIA GPUs."),
    @('Machine', "$($sys.processor -replace '13th Gen Intel\(R\) Core\(TM\) ', 'Intel ') ($($sys.cores -replace ', 2.5 GHz base clock', '')), $($sys.memory -replace ' \(2 x 16 GB\)', ''), $($sys.os -replace ' \(build \d+\)', '')."),
    @('Models', "Llama 3.2 3B Instruct $llamaQuant ($llamaGiB GiB), $chatName $llamaQuant ($chatGiB GiB, from Ollama)."),
    @('Method', "Best of $($set.rounds) llama-bench rounds of $($set.reps) runs; chat: best of $([int]$set.rounds * [int]$doc.chat.reps) requests. Threads: CPU backend $($set.cpu_threads); Kernelcore $($set.kc_threads) for llama.cpp plus its own pool.")
)
if ($gpuTg) { $lines += , @('For scale', ("An RTX 4060 GPU writes {0:N0} tok/s on the same 3B model." -f $gpuTg)) }
$ty = $fy + 14; $labW = 112
foreach ($l in $lines) {
    $g.DrawString($l[0], $fHead, $ink, $M, $ty)
    $rect = New-Object System.Drawing.RectangleF(($M + $labW), $ty, ($W - 2 * $M - $labW), 200)
    $sz = $g.MeasureString($l[1], $fFoot, [int]($W - 2 * $M - $labW))
    $g.DrawString($l[1], $fFoot, $ink2, $rect)
    $ty += [math]::Max(26, $sz.Height) + 6
}
$g.DrawString("Kernelcore$($dot)github.com/arasheedu/Kernelcore", (F 22 'Segoe UI Semibold'), $ink, $M, [math]::Max($ty + 8, $H - 52))
$g.Dispose()
if ($ty + 40 -gt $H - 10) { Write-Warning "setup block runs to y=$ty; the canvas is $H tall" }
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
foreach ($r in $rows) { '{0,-14} {1,-28} CPU {2,7:N1}  Kernelcore {3,7:N1}  {4:N2}x' -f $r.model, $r.task, $r.cpu, $r.kc, $r.ratio }
"wrote $Out (setup block ends at y=$ty)"
