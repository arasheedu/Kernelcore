<#
.SYNOPSIS
  Benchmarks llama.cpp on the CPU backend, pcuda and the GPU (llama-bench), then draws one PNG with
  this machine's system info and the results.

.DESCRIPTION
  Backends:
    CPU    llama.cpp's CPU backend    pcuda build, GGML_PCUDA_DISABLE=1, -t <CpuThreads>
    pcuda  CUDA C kernels on CPU cores  pcuda build, -t 1
    GPU    llama.cpp's CUDA backend   CUDA build, -ngl 99
  Each round runs every backend once (llama-bench -r <Reps>); rounds alternate the backends so
  that heat and turbo state hit all of them alike. The chart shows each test's best round.
  Writes results.json (raw records + system info), results.csv and bench_chart.png to -OutDir.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\bench_chart.ps1
  powershell -ExecutionPolicy Bypass -File scripts\bench_chart.ps1 -RenderOnly   # redraw from results.json
  powershell -ExecutionPolicy Bypass -File scripts\bench_chart.ps1 -NoGpu -Prompt 32,128 -Rounds 1
#>
param(
    [string]$Model = "$env:USERPROFILE\.ollama\models\blobs\sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff",
    [string]$ModelName = '',                         # shown on the chart; default: llama-bench's model type
    [string]$PcudaBench = 'C:\BitsPlus\VisOllamInference\build-pcuda\bin\llama-bench.exe',
    [string]$GpuBench = 'C:\BitsPlus\VisOllamInference\build-gpu\bin\llama-bench.exe',
    [int]$CpuThreads = 6,
    [string]$Prompt = '32,128,512',
    [int]$Gen = 64,
    [int]$Reps = 3,
    [int]$Rounds = 2,
    [int]$CooldownSeconds = 15,
    [string]$OutDir = (Join-Path (Get-Location) 'bench_chart'),
    [switch]$NoGpu,
    [switch]$RenderOnly
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$jsonPath = Join-Path $OutDir 'results.json'

# ---------------------------------------------------------------- system info
function Get-SystemInfo {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $cs = Get-CimInstance Win32_ComputerSystem
    $os = Get-CimInstance Win32_OperatingSystem
    $mods = @(Get-CimInstance Win32_PhysicalMemory)
    $memType = switch (($mods | Select-Object -First 1).SMBIOSMemoryType) { 34 { 'DDR5' } 26 { 'DDR4' } 24 { 'DDR3' } default { 'RAM' } }
    $speed = ($mods | Measure-Object -Property ConfiguredClockSpeed -Maximum).Maximum
    $totalGB = [math]::Round(($mods | Measure-Object -Property Capacity -Sum).Sum / 1GB)
    $sizes = ($mods | Group-Object { [math]::Round($_.Capacity / 1GB) } | ForEach-Object { "$($_.Count) x $($_.Name) GB" }) -join ', '
    $gpu = ''
    try {
        $smi = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ($smi) {
            $f = $smi -split ',\s*'
            $gpu = "$($f[0]), $([math]::Round([double]$f[1] / 1024)) GB, driver $($f[2])"
        }
    } catch {}
    if (-not $gpu) { $gpu = (Get-CimInstance Win32_VideoController | Select-Object -First 1).Name }
    $pcudaCommit = ''
    try { $pcudaCommit = (& git -C (Split-Path $PSScriptRoot -Parent) rev-parse --short HEAD 2>$null) } catch {}
    [ordered]@{
        processor = ($cpu.Name -replace '\s+', ' ').Trim()
        cores     = "$($cpu.NumberOfCores) cores, $($cpu.NumberOfLogicalProcessors) threads, $([math]::Round($cpu.MaxClockSpeed / 1000, 1)) GHz base clock"
        memory    = "$totalGB GB $memType-$speed ($sizes)"
        graphics  = $gpu
        os        = "$($os.Caption -replace 'Microsoft ', '') (build $($os.BuildNumber))"
        computer  = "$($cs.Manufacturer) $($cs.Model)".Trim()
        pcuda     = $pcudaCommit
        date      = (Get-Date).ToString('yyyy-MM-dd HH:mm')
    }
}

# ---------------------------------------------------------------- benchmark runs
$backends = @(
    [ordered]@{ key = 'cpu'; label = 'CPU'; detail = "llama.cpp CPU backend, $CpuThreads threads"; exe = $PcudaBench; env = @{ GGML_PCUDA_DISABLE = '1' }; args = @('-t', "$CpuThreads") },
    [ordered]@{ key = 'pcuda'; label = 'pcuda'; detail = 'CUDA C kernels on the CPU cores'; exe = $PcudaBench; env = @{}; args = @('-t', '1') },
    [ordered]@{ key = 'gpu'; label = 'GPU'; detail = 'llama.cpp CUDA backend, all layers on the GPU'; exe = $GpuBench; env = @{}; args = @('-ngl', '99') }
)
if ($NoGpu) { $backends = @($backends | Where-Object { $_.key -ne 'gpu' }) }

function Invoke-Bench($b) {
    $saved = @{}
    foreach ($k in $b.env.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $b.env[$k]) }
    try {
        # Windows PowerShell turns a native program's stderr lines into errors under 'Stop'
        $ErrorActionPreference = 'Continue'
        $out = & $b.exe -m $Model -p $Prompt -n "$Gen" -r "$Reps" -o json @($b.args) 2>$null
        if ($LASTEXITCODE -ne 0) { throw "llama-bench exited with $LASTEXITCODE" }
        return @(($out -join "`n") | ConvertFrom-Json)
    } finally {
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
}

if (-not $RenderOnly) {
    if (-not (Test-Path $Model)) { throw "model not found: $Model" }
    foreach ($b in $backends) { if (-not (Test-Path $b.exe)) { throw "$($b.label): llama-bench not found at $($b.exe) (use -NoGpu to skip the GPU)" } }
    $records = New-Object System.Collections.ArrayList
    $meta = @{}
    for ($r = 1; $r -le $Rounds; $r++) {
        foreach ($b in $backends) {
            Write-Host "round $r/${Rounds}: $($b.label) ..."
            foreach ($t in (Invoke-Bench $b)) {
                $test = if ([int]$t.n_gen -gt 0) { "tg$($t.n_gen)" } else { "pp$($t.n_prompt)" }
                [void]$records.Add([ordered]@{ backend = $b.key; round = $r; test = $test; avg_ts = [double]$t.avg_ts; stddev_ts = [double]$t.stddev_ts })
                Write-Host ("  {0,-6} {1,9:N1} tok/s" -f $test, [double]$t.avg_ts)
                if (-not $meta[$b.key]) { $meta[$b.key] = @{ build_commit = $t.build_commit; model_type = $t.model_type; model_size = $t.model_size; model_n_params = $t.model_n_params; cpu_info = $t.cpu_info; gpu_info = $t.gpu_info } }
            }
            Start-Sleep -Seconds $CooldownSeconds
        }
    }
    $doc = [ordered]@{
        system   = Get-SystemInfo
        settings = [ordered]@{ model = $Model; prompt = $Prompt; gen = $Gen; reps = $Reps; rounds = $Rounds; cpu_threads = $CpuThreads }
        backends = @($backends | ForEach-Object { [ordered]@{ key = $_.key; label = $_.label; detail = $_.detail } })
        meta     = $meta
        records  = $records
    }
    $doc | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $jsonPath
}

$doc = Get-Content -Raw $jsonPath | ConvertFrom-Json
$sys = $doc.system
$bks = @($doc.backends)

# best round per (backend, test)
$best = @{}
foreach ($rec in $doc.records) {
    $k = "$($rec.backend)|$($rec.test)"
    if (-not $best.ContainsKey($k) -or $rec.avg_ts -gt $best[$k].avg_ts) { $best[$k] = $rec }
}
$tests = @($doc.records | ForEach-Object { $_.test } | Select-Object -Unique)
$genTests = @($tests | Where-Object { $_ -like 'tg*' })
$ppTests = @($tests | Where-Object { $_ -like 'pp*' } | Sort-Object { [int]($_.Substring(2)) })

# table view next to the image
$rows = foreach ($t in $genTests + $ppTests) {
    $row = [ordered]@{ test = $t }
    foreach ($b in $bks) { $k = "$($b.key)|$t"; $row[$b.label] = if ($best.ContainsKey($k)) { [math]::Round($best[$k].avg_ts, 1) } else { '' } }
    [pscustomobject]$row
}
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $OutDir 'results.csv')

# ---------------------------------------------------------------- drawing
Add-Type -AssemblyName System.Drawing
function C([string]$hex) { [System.Drawing.ColorTranslator]::FromHtml($hex) }
$surface = C '#fcfcfb'; $ink = C '#0b0b0b'; $ink2 = C '#52514e'; $grid = C '#e6e5e1'; $card = C '#f4f3f0'
# categorical slots 1-3 (validated all-pairs, light mode); values are always labeled on the bars
$seriesColor = @{ cpu = C '#2a78d6'; pcuda = C '#eb6834'; gpu = C '#1baf7a' }

function F([float]$px, [string]$face = 'Segoe UI', [System.Drawing.FontStyle]$style = 'Regular') {
    New-Object System.Drawing.Font($face, $px, $style, [System.Drawing.GraphicsUnit]::Pixel)
}
$fTitle = F 34 'Segoe UI Semibold'; $fSub = F 18; $fHead = F 21 'Segoe UI Semibold'; $fLabel = F 15
$fValue = F 18; $fValueB = F 18 'Segoe UI Semibold'; $fCat = F 17; $fTick = F 14; $fFoot = F 14

function Brush($c) { New-Object System.Drawing.SolidBrush($c) }
$bInk = Brush $ink; $bInk2 = Brush $ink2; $bCard = Brush $card
$pGrid = New-Object System.Drawing.Pen($grid, 1)
$pAxis = New-Object System.Drawing.Pen((C '#b9b8b2'), 1.5)

function Bar($g, [float]$x, [float]$y, [float]$w, [float]$h, $color) {
    # anchored on the baseline (square left end), 4 px rounded data end
    $r = 4.0
    $br = Brush $color
    if ($w -le 2 * $r) { $g.FillRectangle($br, $x, $y, [math]::Max($w, 1), $h); return }
    $p = New-Object System.Drawing.Drawing2D.GraphicsPath
    $p.AddLine($x, $y, $x + $w - $r, $y)
    $p.AddArc($x + $w - 2 * $r, $y, 2 * $r, 2 * $r, 270, 90)
    $p.AddArc($x + $w - 2 * $r, $y + $h - 2 * $r, 2 * $r, 2 * $r, 0, 90)
    $p.AddLine($x + $w - $r, $y + $h, $x, $y + $h)
    $p.CloseFigure()
    $g.FillPath($br, $p)
}
function NiceStep([double]$max) {
    $raw = $max / 5
    $mag = [math]::Pow(10, [math]::Floor([math]::Log10($raw)))
    foreach ($m in 1, 2, 2.5, 5, 10) { if ($m * $mag -ge $raw) { return $m * $mag } }
}
function Fmt([double]$v) { if ($v -ge 1000) { $v.ToString('N0') } else { $v.ToString('N1') } }

$W = 1800; $M = 56
$leftW = 520; $chartX = $M + $leftW + 56; $chartW = $W - $M - $chartX
$barH = 30; $barGap = 2; $groupGap = 22; $catW = 92; $labelRoom = 250
$nb = $bks.Count
function ChartHeight($n) { 46 + $n * ($nb * $barH + ($nb - 1) * $barGap) + ($n - 1) * $groupGap + 44 }
$hGen = ChartHeight $genTests.Count; $hPP = ChartHeight $ppTests.Count
$top = 150

# system card contents, measured before the canvas size is fixed
$builds = @()
foreach ($b in $bks) { $mb = $doc.meta.($b.key); if ($mb -and $mb.build_commit) { $builds += "$($b.label): llama.cpp $($mb.build_commit)" } }
if ($sys.pcuda) { $builds += "pcuda (CudaOnCPUCores) $($sys.pcuda)" }
$items = @(
    @('Processor', $sys.processor), @('Cores', $sys.cores), @('Memory', $sys.memory), @('Graphics', $sys.graphics),
    @('Operating system', $sys.os), @('Computer', $sys.computer), @('Builds', ($builds -join "`n"))
) | Where-Object { $_[1] }
$items = @($items)
$tmpBmp = New-Object System.Drawing.Bitmap(1, 1)
$tmpG = [System.Drawing.Graphics]::FromImage($tmpBmp)
$itemH = @(foreach ($it in $items) { 21 + $tmpG.MeasureString($it[1], $fValue, [int]($leftW - 56)).Height + 16 })
$tmpG.Dispose(); $tmpBmp.Dispose()
$cardH = 72 + ($itemH | Measure-Object -Sum).Sum + 10
$H = [int]([math]::Max($top + 60 + $hGen + 40 + $hPP, $top + $cardH) + 100)

$bmp = New-Object System.Drawing.Bitmap($W, $H)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = 'AntiAlias'; $g.TextRenderingHint = 'ClearTypeGridFit'; $g.PixelOffsetMode = 'HighQuality'
$g.Clear($surface)

# title
$modelLabel = $ModelName
if (-not $modelLabel) {
    $m0 = $doc.meta.PSObject.Properties | Select-Object -First 1
    if ($m0) {
        $mm = $m0.Value
        $modelLabel = "$($mm.model_type), $([math]::Round([double]$mm.model_size / 1GB, 2)) GiB, $([math]::Round([double]$mm.model_n_params / 1e9, 2))B parameters"
    }
}
$g.DrawString('llama.cpp on this PC: CPU vs pcuda vs GPU', $fTitle, $bInk, $M, 40)
$dot = "  $([char]0x00B7)  "   # kept out of the file's bytes: Windows PowerShell reads scripts as ANSI
$g.DrawString("$modelLabel${dot}llama-bench, tokens per second (higher is better)$dot$($sys.date)", $fSub, $bInk2, $M + 2, 92)

# system card
$g.FillRectangle($bCard, $M, $top, $leftW, $cardH)
$g.DrawString('System', $fHead, $bInk, $M + 28, $top + 24)
$y = $top + 72
foreach ($it in $items) {
    $g.DrawString($it[0], $fLabel, $bInk2, $M + 28, $y)
    $rect = New-Object System.Drawing.RectangleF(($M + 28), ($y + 21), ($leftW - 56), 400)
    $sz = $g.MeasureString($it[1], $fValue, [int]($leftW - 56))
    $g.DrawString($it[1], $fValue, $bInk, $rect)
    $y += 21 + $sz.Height + 16
}

# legend: one row above the charts, swatch + name + what it is
$lx = $chartX; $ly = $top + 4
foreach ($b in $bks) {
    Bar $g $lx ($ly + 4) 18 18 $seriesColor[$b.key]
    $g.DrawString($b.label, $fValueB, $bInk, $lx + 26, $ly)
    $w1 = $g.MeasureString($b.label, $fValueB).Width
    $g.DrawString($b.detail, $fLabel, $bInk2, $lx + 26 + $w1, $ly + 3)
    $lx += 26 + $w1 + $g.MeasureString($b.detail, $fLabel).Width + 28
}

function Draw-Chart($title, $testList, [float]$y0) {
    $g.DrawString($title, $fHead, $bInk, $chartX, $y0)
    $px = $chartX + $catW; $pw = $chartW - $catW - $labelRoom
    $vals = @(foreach ($t in $testList) { foreach ($b in $bks) { $k = "$($b.key)|$t"; if ($best.ContainsKey($k)) { $best[$k].avg_ts } } })
    $step = NiceStep (($vals | Measure-Object -Maximum).Maximum)
    $max = $step * [math]::Ceiling((($vals | Measure-Object -Maximum).Maximum) / $step)
    $py = $y0 + 46
    $plotH = $testList.Count * ($nb * $barH + ($nb - 1) * $barGap) + ($testList.Count - 1) * $groupGap
    # recessive vertical grid + tick labels
    for ($v = 0; $v -le $max + 1e-9; $v += $step) {
        $gx = $px + $pw * $v / $max
        $g.DrawLine($pGrid, $gx, $py - 6, $gx, $py + $plotH + 6)
        $tl = Fmt $v; if ($v -lt 1000 -and $step -ge 1) { $tl = $v.ToString('N0') }
        $tw = $g.MeasureString($tl, $fTick).Width
        $g.DrawString($tl, $fTick, $bInk2, $gx - $tw / 2, $py + $plotH + 10)
    }
    $g.DrawLine($pAxis, $px, $py - 6, $px, $py + $plotH + 6)
    $yy = $py
    foreach ($t in $testList) {
        $gh = $nb * $barH + ($nb - 1) * $barGap
        $cs = $g.MeasureString($t, $fCat)
        $g.DrawString($t, $fCat, $bInk, $chartX, $yy + ($gh - $cs.Height) / 2)
        $cpuV = $null; $kc = "cpu|$t"; if ($best.ContainsKey($kc)) { $cpuV = $best[$kc].avg_ts }
        foreach ($b in $bks) {
            $k = "$($b.key)|$t"
            if ($best.ContainsKey($k)) {
                $v = $best[$k].avg_ts
                $bw = [float]($pw * $v / $max)
                Bar $g $px $yy $bw $barH $seriesColor[$b.key]
                $label = Fmt $v
                $lw = $g.MeasureString($label, $fValueB).Width
                $g.DrawString($label, $fValueB, $bInk, $px + $bw + 8, $yy + ($barH - 24) / 2)
                if ($b.key -ne 'cpu' -and $cpuV) {
                    $ratio = $v / $cpuV
                    $note = if ($ratio -ge 2) { '{0:N1}x CPU' -f $ratio } else { '{0:+0;-0}% vs CPU' -f (($ratio - 1) * 100) }
                    $g.DrawString($note, $fLabel, $bInk2, $px + $bw + 12 + $lw, $yy + ($barH - 20) / 2)
                }
            }
            $yy += $barH + $barGap
        }
        $yy += $groupGap - $barGap
    }
}
$yc = $top + 60
Draw-Chart "Generation (tokens/s)" $genTests $yc
Draw-Chart "Prompt processing (tokens/s, by prompt length)" $ppTests ($yc + $hGen + 40)

# footer
$set = $doc.settings
$roundsText = if ([int]$set.rounds -eq 1) { '1 round' } else { "the best of $($set.rounds) rounds" }
$foot = "Each bar: $roundsText of llama-bench -r $($set.reps); the backends alternate within each round. " +
        "tgN = generating N tokens, ppN = processing an N-token prompt. pcuda runs llama.cpp with 1 ggml thread; its own pool uses the P-cores."
$g.DrawString($foot, $fFoot, $bInk2, (New-Object System.Drawing.RectangleF($M, ($H - 80), ($W - 2 * $M), 60)))

$png = Join-Path $OutDir 'bench_chart.png'
$bmp.Save($png, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
Write-Host "wrote $png"
Write-Host "      $(Join-Path $OutDir 'results.csv')"
$rows | Format-Table -AutoSize | Out-String | Write-Host
