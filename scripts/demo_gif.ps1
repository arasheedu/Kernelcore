<#
.SYNOPSIS
  Records a short demo (GIF + MP4): one CUDA kernel compiled by Booth for Kernelcore and by nvcc
  for the GPU, then both run and compared.

.DESCRIPTION
  Runs the real commands in a scratch folder and captures their output:
    1. type vector_add.cu
    2. kath --cpu vector_add.cu -o vector_add.o      (Booth, as Kernelcore's build does)
       kc_objcheck vector_add.o --kernel vector_add   (the check Kernelcore runs before loading)
    3. nvcc -arch=native -cubin vector_add.cu -o vector_add.cubin
    4. kc_vs_cuda --check   (built from the same .cu files by both toolchains; compares results)
  Then draws terminal frames (typed commands, step captions) and encodes them with ffmpeg.

  Needs: the windows-msvc-nvidia build (kc_vs_cuda, kc_objcheck), Booth's kath in WSL
  (scripts/install_kath_wsl.sh) or -Kath <kath.exe>, the CUDA Toolkit, ffmpeg on PATH.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\demo_gif.ps1
#>
param(
    [string]$OutDir = (Join-Path (Split-Path $PSScriptRoot -Parent) 'demo'),
    [string]$Kath = '',              # a kath.exe; default: kath inside WSL
    [string]$Build = (Join-Path (Split-Path $PSScriptRoot -Parent) 'build\windows-msvc-nvidia'),
    [int]$Width = 1280,
    [int]$Height = 720
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Add-Type -AssemblyName System.Drawing

# ---- Visual Studio environment (nvcc needs cl.exe) ----
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $env:PATH = "$(Split-Path $vswhere);$env:PATH"
    Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
}
# CUDA < 12.4 rejects newer MSVC; keep the shown nvcc command short (nvcc reads this variable).
$nvccVer = (& nvcc --version | Select-String 'release (\d+)\.(\d+)').Matches[0].Groups
if ([int]$nvccVer[1].Value -lt 12 -or ([int]$nvccVer[1].Value -eq 12 -and [int]$nvccVer[2].Value -lt 4)) {
    $env:NVCC_APPEND_FLAGS = '-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'
}

# ---- run the steps ----
$work = Join-Path ([System.IO.Path]::GetTempPath()) 'kernelcore_demo'
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $work | Out-Null
Copy-Item (Join-Path $root 'kernels\vector_add.cu') $work
if ($Kath) {
    Set-Content -Encoding ascii (Join-Path $work 'kath.cmd') "@`"$Kath`" %*"
} else {
    $pin = Get-Content -Raw (Join-Path $root 'cmake\BoothPin.cmake')
    $ver = [regex]::Match($pin, 'BOOTH_VERSION\s+"?([0-9.]+)').Groups[1].Value
    Set-Content -Encoding ascii (Join-Path $work 'kath.cmd') "@wsl.exe -e sh -c `"~/.local/share/kernelcore/booth-$ver/kath %*`""
}
$env:PATH = "$work;$Build;$env:PATH"

function Run([string]$cmd) {
    Push-Location $work
    try {
        $ErrorActionPreference = 'Continue'
        $out = & cmd.exe /c "$cmd 2>&1"
        if ($LASTEXITCODE -ne 0) { throw "demo step failed ($LASTEXITCODE): $cmd`n$($out -join "`n")" }
        return @($out | ForEach-Object { ([string]$_).TrimEnd() })
    } finally { Pop-Location }
}
$steps = @(
    @{ title = '1  One CUDA kernel'; sub = 'kernels/vector_add.cu, unchanged'; cmd = 'type vector_add.cu' },
    @{ title = '2  Kernelcore: Booth compiles it to x86-64'; sub = 'kath --cpu, then the check Kernelcore runs before loading it'
       cmd = 'kath --cpu vector_add.cu -o vector_add.o'; cmd2 = 'kc_objcheck vector_add.o --kernel vector_add' },
    @{ title = '3  NVIDIA: nvcc compiles the same file for the GPU'; sub = 'CUDA Toolkit, this PC''s RTX card'
       cmd = 'nvcc -arch=native -cubin vector_add.cu -o vector_add.cubin'; cmd2 = 'dir /-c vector_add.o vector_add.cubin | findstr vector_add' },
    @{ title = '4  Run both and compare'; sub = 'kc_vs_cuda: built from the same .cu files by both toolchains'; cmd = 'kc_vs_cuda --check' }
)
foreach ($s in $steps) {
    Write-Host "running: $($s.cmd)"
    $s.out = Run $s.cmd
    if ($s.cmd2) { Write-Host "running: $($s.cmd2)"; $s.out2 = Run $s.cmd2 }
}
# nvcc echoes the file name; keep it, it is what the user would see.

# ---- draw frames ----
$frames = Join-Path $work 'frames'
New-Item -ItemType Directory -Force $frames | Out-Null
$bg = [System.Drawing.Color]::FromArgb(24, 26, 31)
$bar = [System.Drawing.Color]::FromArgb(36, 39, 46)
$cText = [System.Drawing.Color]::FromArgb(222, 226, 232)
$cDim = [System.Drawing.Color]::FromArgb(140, 148, 160)
$cPrompt = [System.Drawing.Color]::FromArgb(110, 190, 120)
$cAccent = [System.Drawing.Color]::FromArgb(235, 104, 52)   # Kernelcore orange, as on the chart
$cGreen = [System.Drawing.Color]::FromArgb(27, 175, 122)
$fMono = New-Object System.Drawing.Font('Consolas', 14)
$fTitle = New-Object System.Drawing.Font('Segoe UI Semibold', 22)
$fSub = New-Object System.Drawing.Font('Segoe UI', 14)
$fBig = New-Object System.Drawing.Font('Segoe UI Semibold', 30)
$fMid = New-Object System.Drawing.Font('Segoe UI Semibold', 19)
$lineH = 21; $x0 = 40; $yText = 120
# columns that fit: measure the monospace font once
$tmpB = New-Object System.Drawing.Bitmap(1, 1); $tmpG = [System.Drawing.Graphics]::FromImage($tmpB)
$charW = ($tmpG.MeasureString(('x' * 100), $fMono).Width - $tmpG.MeasureString('x', $fMono).Width) / 99
$tmpG.Dispose(); $tmpB.Dispose()
$cols = [int][math]::Floor(($Width - 2 * $x0) / $charW) - 2
$script:n = 0
$list = New-Object System.Collections.Generic.List[string]

function Color-For([string]$line) {
    if ($line -match '\bPASS\b|: OK |exit=0') { return $cGreen }
    if ($line -match '^\s*//') { return $cDim }
    return $cText
}
# lines: list of @(text, color); duration in seconds
function Frame($step, $lines, [double]$dur, [bool]$cursor = $false) {
    $bmp = New-Object System.Drawing.Bitmap($Width, $Height)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.TextRenderingHint = 'ClearTypeGridFit'; $g.SmoothingMode = 'AntiAlias'
    $g.Clear($bg)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($bar)), 0, 0, $Width, 96)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($cAccent)), 0, 0, 8, 96)
    $g.DrawString($step.title, $fTitle, (New-Object System.Drawing.SolidBrush($cText)), 30, 14)
    $g.DrawString($step.sub, $fSub, (New-Object System.Drawing.SolidBrush($cDim)), 33, 56)
    $y = $yText
    foreach ($l in $lines) {
        $g.DrawString($l[0], $fMono, (New-Object System.Drawing.SolidBrush($l[1])), $x0, $y)
        $y += $lineH
    }
    if ($cursor) {
        $last = if ($lines.Count) { $lines[$lines.Count - 1][0] } else { '' }
        $w = $g.MeasureString($last + '|', $fMono).Width - $g.MeasureString('|', $fMono).Width
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($cText)), $x0 + $w - 2, $y - $lineH + 3, 9, 17)
    }
    $g.Dispose()
    $script:n++
    $name = 'f{0:D4}.png' -f $script:n
    $bmp.Save((Join-Path $frames $name), [System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
    $list.Add("file '$name'"); $list.Add(("duration {0:0.###}" -f $dur))
}
function Wrap([string]$s) {
    if ($s.Length -le $cols) { return , $s }
    $out = @(); $rest = $s
    while ($rest.Length -gt $cols) {
        $cut = $rest.LastIndexOf(' ', $cols); if ($cut -lt 20) { $cut = $cols }
        $out += $rest.Substring(0, $cut); $rest = '    ' + $rest.Substring($cut).TrimStart()
    }
    return $out + $rest
}
$prompt = 'PS demo> '
function Type-Command($step, $shown, [string]$cmd) {
    for ($i = 0; $i -le $cmd.Length; $i += 2) {
        $cur = @($shown) + , @(($prompt + $cmd.Substring(0, [math]::Min($i, $cmd.Length))), $cText)
        Frame $step $cur 0.045 $true
    }
    $done = @($shown) + , @(($prompt + $cmd), $cText)
    Frame $step $done 0.5 $true
    return , $done
}
function Show-Output($step, $shown, $out, [double]$perLine = 0.06) {
    foreach ($o in $out) {
        foreach ($w in (Wrap $o)) { $shown = @($shown) + , @($w, (Color-For $o)) }
        Frame $step $shown $perLine
    }
    return , $shown
}

foreach ($s in $steps) {
    $shown = @()
    Frame $s $shown 1.2
    $shown = Type-Command $s $shown $s.cmd
    $shown = Show-Output $s $shown $s.out
    if ($s.cmd2) {
        $shown = @($shown) + , @('', $cText)
        Frame $s $shown 0.6
        $shown = Type-Command $s $shown $s.cmd2
        $shown = Show-Output $s $shown $s.out2
    }
    Frame $s $shown 3.0
}

# closing card
$oSize = (Get-Item (Join-Path $work 'vector_add.o')).Length
$cSize = (Get-Item (Join-Path $work 'vector_add.cubin')).Length
$va = ($steps[3].out | Select-String 'GPU vs Kernelcore (\d+) differ' | Select-Object -First 1).Matches[0].Groups[1].Value
$bmp = New-Object System.Drawing.Bitmap($Width, $Height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.TextRenderingHint = 'ClearTypeGridFit'; $g.SmoothingMode = 'AntiAlias'; $g.Clear($bg)
$arrow = [string][char]0x2192
$bT = New-Object System.Drawing.SolidBrush($cText); $bD = New-Object System.Drawing.SolidBrush($cDim)
$g.DrawString('One .cu file, two compilers, same answer', $fBig, $bT, 70, 90)
$g.DrawString('vector_add.cu', $fTitle, (New-Object System.Drawing.SolidBrush($cAccent)), 70, 200)
$g.DrawString("$arrow  Booth (kath --cpu)  $arrow  x86-64 object, $oSize bytes  $arrow  Kernelcore on the CPU cores", $fMid, $bT, 110, 262)
$g.DrawString("$arrow  nvcc  $arrow  GPU cubin, $cSize bytes  $arrow  NVIDIA GPU", $fMid, $bT, 110, 310)
$g.DrawString("vector_add over 1.1M elements: $va results differ between the CPU and the GPU.", $fTitle, (New-Object System.Drawing.SolidBrush($cGreen)), 70, 400)
$g.DrawString('Kernelcore  ' + [char]0x00B7 + '  CUDA C kernels on CPU cores  ' + [char]0x00B7 + '  github.com/arasheedu/Kernelcore', $fSub, $bD, 70, 600)
$g.DrawString('CUDA is a trademark of NVIDIA Corporation. Kernelcore is not affiliated with NVIDIA.', $fSub, $bD, 70, 630)
$g.Dispose()
$script:n++; $name = 'f{0:D4}.png' -f $script:n
$bmp.Save((Join-Path $frames $name), [System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
$list.Add("file '$name'"); $list.Add('duration 5'); $list.Add("file '$name'")   # concat needs the last file twice
[System.IO.File]::WriteAllLines((Join-Path $frames 'list.txt'), $list)

# ---- encode ----
New-Item -ItemType Directory -Force $OutDir | Out-Null
$gif = Join-Path $OutDir 'kernelcore_vs_nvcc.gif'; $mp4 = Join-Path $OutDir 'kernelcore_vs_nvcc.mp4'
Push-Location $frames
try {
    $ErrorActionPreference = 'Continue'
    & ffmpeg -y -v error -f concat -safe 0 -i list.txt -vf 'fps=30,format=yuv420p' -c:v libx264 -crf 20 -movflags +faststart $mp4
    & ffmpeg -y -v error -f concat -safe 0 -i list.txt -vf 'fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle' $gif
} finally { Pop-Location }
Get-Item $gif, $mp4 | ForEach-Object { '{0}  {1:N0} KB' -f $_.FullName, ($_.Length / 1KB) }
