<#
.SYNOPSIS
  Benchmarks llama.cpp on the CPU backend, Kernelcore and the GPU (llama-bench), then draws one PNG with
  this machine's system info and the results.

.DESCRIPTION
  Backends:
    CPU    llama.cpp's CPU backend    Kernelcore build, GGML_KERNELCORE_DISABLE=1, -t <CpuThreads>
    Kernelcore  CUDA C kernels on CPU cores  Kernelcore build, -t <KcThreads> for generation and
                                        -t <KcBatchThreads> for prompts (as llama-server
                                        -t 4 -tb 4: llama-bench has no -tb)
    GPU    llama.cpp's CUDA backend   CUDA build, -ngl 99
  Each round runs every backend once (llama-bench -r <Reps>); rounds alternate the backends so
  that heat and turbo state hit all of them alike. The chart shows each test's best round.

  Real chat (skip with -NoChat): each round also starts llama-server with -ChatModel (default:
  Ollama's qwen3:8b), on the CPU backend and on Kernelcore, and sends -ChatPrompt through
  /v1/chat/completions (chat template, greedy, no thinking unless -ChatThink) -ChatReps times
  after one warm-up request. It records the server's own prompt and reply speeds and the time for
  the whole answer, and checks that both backends give the same reply.

  Kernelcore on the GPU (skip with -NoKernels; also skipped by -NoGpu): there is no llama.cpp backend
  that runs Kernelcore's kernels on the GPU, so this compares the kernels alone. Each round runs
  kc_kquant_vs_cuda (-KernelApp, from the windows-msvc-nvidia preset): one Llama 3.2 3B layer's
  matmuls (2 activation quantizations + 7 matrices, random weights) with the same .cu files on
  Kernelcore (CPU) and compiled by nvcc (GPU), checked against each other. tok/s = tokens / (28 layers x
  layer time): matmuls only, not comparable with the llama-bench bars. 1 token uses activations
  scaled per 32 values and batches per 256, as the ggml backend does.

  Writes results.json (raw records + system info), results.csv and bench_chart.png to -OutDir.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\bench_chart.ps1
  powershell -ExecutionPolicy Bypass -File scripts\bench_chart.ps1 -RenderOnly   # redraw from results.json
  powershell -ExecutionPolicy Bypass -File scripts\bench_chart.ps1 -NoGpu -Prompt 32,128 -Rounds 1
  powershell -ExecutionPolicy Bypass -File scripts\bench_chart.ps1 -ChatOllamaModel llama3.1:8b -ChatPrompt "Write a haiku."
#>
param(
    [string]$Model = "$env:USERPROFILE\.ollama\models\blobs\sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff",
    [string]$ModelName = '',                         # shown on the chart; default: llama-bench's model type
    [string]$KcBench = 'C:\BitsPlus\VisOllamInference\build-kernelcore\bin\llama-bench.exe',
    [string]$GpuBench = 'C:\BitsPlus\VisOllamInference\build-gpu\bin\llama-bench.exe',
    [int]$CpuThreads = 6,
    [int]$KcThreads = 4,        # Kernelcore generation: ggml threads (attention runs on them; more than 4 compete with Kernelcore's pool)
    [int]$KcBatchThreads = 4,   # Kernelcore prompts: ggml threads for the ops Kernelcore leaves to the CPU (attention)
    [string]$Prompt = '32,128,512',
    [int]$Gen = 64,
    [int]$Reps = 3,
    [int]$Rounds = 2,
    [int]$CooldownSeconds = 15,
    [string]$OutDir = (Join-Path (Get-Location) 'bench_chart'),
    [switch]$NoGpu,
    [switch]$RenderOnly,
    # real chat through llama-server
    [string]$ChatOllamaModel = 'qwen3:8b',          # resolved in Ollama's model store
    [string]$ChatModel = '',                        # a GGUF path instead of -ChatOllamaModel
    [string]$ChatPrompt = 'Why is the sky blue?',
    [int]$ChatTokens = 256,                         # reply limit (the reply may end earlier)
    [int]$ChatReps = 3,
    [int]$ChatPort = 8093,
    [string]$KcServer = '',                      # default: llama-server.exe next to -KcBench
    [switch]$ChatThink,                             # let the model think first (Qwen3); default: --reasoning-budget 0
    [switch]$NoChat,
    # Kernelcore's kernels on the GPU (kernel-only comparison)
    [string]$KernelApp = (Join-Path (Split-Path $PSScriptRoot -Parent) 'build\windows-msvc-nvidia\kc_kquant_vs_cuda.exe'),
    [switch]$NoKernels
)
if ($NoGpu) { $NoKernels = $true }
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$jsonPath = Join-Path $OutDir 'results.json'
if (-not $KcServer) { $KcServer = Join-Path (Split-Path $KcBench -Parent) 'llama-server.exe' }

# Ollama stores a model as blobs named by hash; the manifest names the GGUF layer.
function Resolve-OllamaModel([string]$name) {
    $n, $tag = $name -split ':', 2
    if (-not $tag) { $tag = 'latest' }
    $man = Join-Path $env:USERPROFILE ".ollama\models\manifests\registry.ollama.ai\library\$n\$tag"
    if (-not (Test-Path $man)) { return '' }
    $m = Get-Content -Raw $man | ConvertFrom-Json
    $d = ($m.layers | Where-Object { $_.mediaType -eq 'application/vnd.ollama.image.model' } | Select-Object -First 1).digest
    if (-not $d) { return '' }
    Join-Path $env:USERPROFILE (".ollama\models\blobs\" + ($d -replace ':', '-'))
}
if (-not $NoChat -and -not $RenderOnly -and -not $ChatModel) {
    $ChatModel = Resolve-OllamaModel $ChatOllamaModel
    if (-not $ChatModel) { throw "Ollama model $ChatOllamaModel not found (pull it, pass -ChatModel <gguf>, or use -NoChat)" }
}

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
    $kcCommit = ''
    try { $kcCommit = (& git -C (Split-Path $PSScriptRoot -Parent) rev-parse --short HEAD 2>$null) } catch {}
    [ordered]@{
        processor = ($cpu.Name -replace '\s+', ' ').Trim()
        cores     = "$($cpu.NumberOfCores) cores, $($cpu.NumberOfLogicalProcessors) threads, $([math]::Round($cpu.MaxClockSpeed / 1000, 1)) GHz base clock"
        memory    = "$totalGB GB $memType-$speed ($sizes)"
        graphics  = $gpu
        os        = "$($os.Caption -replace 'Microsoft ', '') (build $($os.BuildNumber))"
        computer  = "$($cs.Manufacturer) $($cs.Model)".Trim()
        Kernelcore     = $kcCommit
        date      = (Get-Date).ToString('yyyy-MM-dd HH:mm')
    }
}

# ---------------------------------------------------------------- benchmark runs
$backends = @(
    [ordered]@{ key = 'cpu'; label = 'CPU'; detail = "llama.cpp CPU backend, $CpuThreads threads"; exe = $KcBench; env = @{ GGML_KERNELCORE_DISABLE = '1' }; args = @('-t', "$CpuThreads") },
    [ordered]@{ key = 'Kernelcore'; label = 'Kernelcore'; detail = 'CUDA C kernels on the CPU cores'; exe = $KcBench; env = @{}; args = @();
      gen_args = @('-t', "$KcThreads"); prompt_args = @('-t', "$KcBatchThreads") },
    [ordered]@{ key = 'gpu'; label = 'GPU'; detail = 'llama.cpp CUDA backend, all layers on the GPU'; exe = $GpuBench; env = @{}; args = @('-ngl', '99') }
)
if ($NoGpu) { $backends = @($backends | Where-Object { $_.key -ne 'gpu' }) }

function Invoke-Bench($b) {
    $saved = @{}
    foreach ($k in $b.env.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $b.env[$k]) }
    try {
        # Windows PowerShell turns a native program's stderr lines into errors under 'Stop'
        $ErrorActionPreference = 'Continue'
        # Separate thread counts for prompts and generation: two runs (prompts, then generation)
        $runs = New-Object System.Collections.ArrayList
        if ($b.prompt_args) {
            [void]$runs.Add(@('-p', $Prompt, '-n', '0') + $b.prompt_args)
            [void]$runs.Add(@('-p', '0', '-n', "$Gen") + $b.gen_args)
        } else {
            [void]$runs.Add(@('-p', $Prompt, '-n', "$Gen") + $b.args)
        }
        $all = @()
        foreach ($a in $runs) {
            $out = & $b.exe -m $Model -r "$Reps" -o json @a 2>$null
            if ($LASTEXITCODE -ne 0) { throw "llama-bench exited with $LASTEXITCODE" }
            $parsed = ($out -join "`n") | ConvertFrom-Json
            foreach ($o in $parsed) { foreach ($x in @($o)) { $all += ,$x } }  # Windows PowerShell may nest the array
        }
        return $all
    } finally {
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
}

# Real chat: llama-server per backend, the same request several times, the server's own timings.
$chatBackends = @(
    [ordered]@{ key = 'cpu'; env = @{ GGML_KERNELCORE_DISABLE = '1' }; args = @('-t', "$CpuThreads", '-tb', "$CpuThreads") },
    [ordered]@{ key = 'Kernelcore'; env = @{}; args = @('-t', "$KcThreads", '-tb', "$KcBatchThreads") }
)
function Invoke-Chat($cb) {
    $saved = @{}
    foreach ($k in $cb.env.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $cb.env[$k]) }
    $log = Join-Path $OutDir "chat_server_$($cb.key).log"
    $a = @('-m', "`"$ChatModel`"", '--port', "$ChatPort", '-c', '4096', '-np', '1', '--no-webui') + $cb.args
    if (-not $ChatThink) { $a += @('--reasoning-budget', '0') }
    $p = Start-Process -FilePath $KcServer -ArgumentList $a -PassThru -WindowStyle Hidden -RedirectStandardError $log -RedirectStandardOutput "$log.out"
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    $url = "http://127.0.0.1:$ChatPort"
    try {
        $ok = $false
        for ($i = 0; $i -lt 600 -and -not $ok; $i++) {
            if ($p.HasExited) { throw "llama-server ($($cb.key)) exited; see $log" }
            Start-Sleep -Milliseconds 500
            try { $ok = (Invoke-RestMethod "$url/health" -TimeoutSec 2).status -eq 'ok' } catch {}
        }
        if (-not $ok) { throw "llama-server ($($cb.key)) did not start; see $log" }
        $info = $null
        try { $info = (Invoke-RestMethod "$url/v1/models" -TimeoutSec 10).data | Select-Object -First 1 } catch {}
        $req = { param([int]$n)
            $body = @{ messages = @(@{ role = 'user'; content = $ChatPrompt }); max_tokens = $n; temperature = 0; top_k = 1; cache_prompt = $false } | ConvertTo-Json -Depth 4
            $sw = [Diagnostics.Stopwatch]::StartNew()
            $c = Invoke-RestMethod "$url/v1/chat/completions" -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 1800
            $sw.Stop()
            [ordered]@{ seconds = $sw.Elapsed.TotalSeconds; timings = $c.timings; text = [string]$c.choices[0].message.content }
        }
        [void](& $req 8)  # warm-up
        $out = @()
        for ($i = 1; $i -le $ChatReps; $i++) { $out += , (& $req $ChatTokens) }
        return [ordered]@{ runs = $out; info = $info }
    } finally {
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
}

# kc_kquant_vs_cuda rows: "tokens | ok ... | KC ms  GPU ms | KC tok/s  GPU tok/s | ratio",
# in a section per activation format; keep the ggml backend's choice (1 token: per 32, batches: per 256).
function Invoke-Kernels {
    $ErrorActionPreference = 'Continue'
    $out = & $KernelApp 2>&1
    if ($LASTEXITCODE -ne 0) { throw "kc_kquant_vs_cuda exited with $LASTEXITCODE" }
    $fmt = 0; $rows = @()
    foreach ($line in $out) {
        $l = [string]$line
        if ($l -match '^Activations scaled per (\d+)') { $fmt = [int]$Matches[1]; continue }
        if ($l -match '^\s*(\d+)\s*\|\s*(ok|BAD)\b.*\|\s*([\d.]+)\s+([\d.]+)\s*\|\s*([\d.]+)\s+([\d.]+)\s*\|') {
            $n = [int]$Matches[1]
            if (($n -eq 1 -and $fmt -eq 32) -or ($n -gt 1 -and $fmt -eq 256)) {
                $rows += , [ordered]@{ tokens = $n; ok = ($Matches[2] -eq 'ok'); kc_tps = [double]$Matches[5]; gpu_tps = [double]$Matches[6] }
            }
        }
    }
    if (-not $rows.Count) { throw "kc_kquant_vs_cuda: no results (no CUDA device?)" }
    return $rows
}

if (-not $RenderOnly) {
    if (-not (Test-Path $Model)) { throw "model not found: $Model" }
    if (-not $NoKernels -and -not (Test-Path $KernelApp)) { throw "kc_kquant_vs_cuda not found at $KernelApp (build the windows-msvc-nvidia preset, or use -NoKernels)" }
    $kernelOk = $true
    if (-not $NoChat) {
        if (-not (Test-Path $ChatModel)) { throw "chat model not found: $ChatModel" }
        if (-not (Test-Path $KcServer)) { throw "llama-server not found at $KcServer (or use -NoChat)" }
    }
    $chatText = @{}
    $chatInfo = $null
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
        if (-not $NoKernels) {
            Write-Host "round $r/${Rounds}: Kernelcore kernels on CPU and GPU ..."
            foreach ($kr in (Invoke-Kernels)) {
                $test = if ($kr.tokens -eq 1) { 'kern tg1' } else { "kern pp$($kr.tokens)" }
                if (-not $kr.ok) { $kernelOk = $false }
                [void]$records.Add([ordered]@{ backend = 'Kernelcore'; round = $r; test = $test; avg_ts = $kr.kc_tps; stddev_ts = 0.0 })
                [void]$records.Add([ordered]@{ backend = 'kcgpu'; round = $r; test = $test; avg_ts = $kr.gpu_tps; stddev_ts = 0.0 })
                Write-Host ("  {0,-6} Kernelcore {1,8:N1}  GPU {2,8:N1} tok/s{3}" -f ($test -replace '^kern ', ''), $kr.kc_tps, $kr.gpu_tps, $(if ($kr.ok) { '' } else { '  RESULTS DIFFER' }))
            }
        }
        if (-not $NoChat) {
            foreach ($cb in $chatBackends) {
                Write-Host "round $r/${Rounds}: chat on $($cb.key) ..."
                $res = Invoke-Chat $cb
                if (-not $chatInfo -and $res.info) { $chatInfo = $res.info }
                foreach ($run in $res.runs) {
                    $t = $run.timings
                    foreach ($tv in @(@('chat reply', $t.predicted_per_second), @('chat prompt', $t.prompt_per_second))) {
                        [void]$records.Add([ordered]@{ backend = $cb.key; round = $r; test = $tv[0]; avg_ts = [double]$tv[1]; stddev_ts = 0.0;
                                                       prompt_n = [int]$t.prompt_n; predicted_n = [int]$t.predicted_n; seconds = [double]$run.seconds })
                    }
                    Write-Host ("  prompt {0} tokens {1,7:N1} tok/s, reply {2} tokens {3,6:N1} tok/s, whole answer {4:N1} s" -f $t.prompt_n, [double]$t.prompt_per_second, $t.predicted_n, [double]$t.predicted_per_second, $run.seconds)
                    $chatText[$cb.key] = $run.text
                }
                Start-Sleep -Seconds $CooldownSeconds
            }
        }
    }
    $chat = $null
    if (-not $NoChat) {
        $name = $ChatOllamaModel
        if ($PSBoundParameters.ContainsKey('ChatModel')) { $name = Split-Path $ChatModel -Leaf }
        $chat = [ordered]@{
            model = $ChatModel; name = $name; prompt = $ChatPrompt; max_tokens = $ChatTokens; reps = $ChatReps; think = [bool]$ChatThink
            n_params = $(if ($chatInfo -and $chatInfo.meta) { $chatInfo.meta.n_params } else { $null })
            size = $(if ($chatInfo -and $chatInfo.meta) { $chatInfo.meta.size } else { $null })
            same_reply = ($chatText['cpu'] -eq $chatText['Kernelcore'])
            same_words = $(  # leading words the two replies share
                $wa = @(([string]$chatText['cpu']) -split '\s+' | Where-Object { $_ }); $wb = @(([string]$chatText['Kernelcore']) -split '\s+' | Where-Object { $_ })
                $i = 0; while ($i -lt $wa.Count -and $i -lt $wb.Count -and $wa[$i] -ceq $wb[$i]) { $i++ }; $i)
            words = @(@(([string]$chatText['cpu']) -split '\s+' | Where-Object { $_ }).Count, @(([string]$chatText['Kernelcore']) -split '\s+' | Where-Object { $_ }).Count)
            replies = $chatText
        }
    }
    $doc = [ordered]@{
        system   = Get-SystemInfo
        settings = [ordered]@{ model = $Model; prompt = $Prompt; gen = $Gen; reps = $Reps; rounds = $Rounds; cpu_threads = $CpuThreads; kc_threads = $KcThreads; kc_batch_threads = $KcBatchThreads }
        backends = @(@($backends | ForEach-Object { [ordered]@{ key = $_.key; label = $_.label; detail = $_.detail } }) +
                     $(if (-not $NoKernels) { , [ordered]@{ key = 'kcgpu'; label = 'Kernelcore on GPU'; detail = 'the same .cu kernels, nvcc (matmuls only)' } } else { @() }))
        meta     = $meta
        chat     = $chat
        kernels  = $(if (-not $NoKernels) { [ordered]@{ app = $KernelApp; results_match = $kernelOk } } else { $null })
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
$chatTests = @('chat reply', 'chat prompt' | Where-Object { $tests -contains $_ })
$chatBks = @($bks | Where-Object { $_.key -eq 'cpu' -or $_.key -eq 'Kernelcore' })
$kernTests = @($tests | Where-Object { $_ -like 'kern *' } | Sort-Object { if ($_ -eq 'kern tg1') { 0 } else { [int]($_ -replace '^kern pp', '') } })
$kernBks = @($bks | Where-Object { $_.key -eq 'Kernelcore' -or $_.key -eq 'kcgpu' })
$allBks = $bks                                            # legend and table
$bks = @($bks | Where-Object { $_.key -ne 'kcgpu' })   # llama-bench charts

# table view next to the image
$rows = foreach ($t in $genTests + $ppTests + $chatTests + $kernTests) {
    $row = [ordered]@{ test = $t }
    foreach ($b in $allBks) { $k = "$($b.key)|$t"; $row[$b.label] = if ($best.ContainsKey($k)) { [math]::Round($best[$k].avg_ts, 1) } else { '' } }
    [pscustomobject]$row
}
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $OutDir 'results.csv')

# ---------------------------------------------------------------- drawing
Add-Type -AssemblyName System.Drawing
function C([string]$hex) { [System.Drawing.ColorTranslator]::FromHtml($hex) }
$surface = C '#fcfcfb'; $ink = C '#0b0b0b'; $ink2 = C '#52514e'; $grid = C '#e6e5e1'; $card = C '#f4f3f0'
# categorical slots 1-3 (validated all-pairs, light mode); values are always labeled on the bars
# Kernelcore on GPU: slot 7 (violet), only ever charted next to Kernelcore's orange (that pair passes all checks)
$seriesColor = @{ cpu = C '#2a78d6'; Kernelcore = C '#eb6834'; gpu = C '#1baf7a'; kcgpu = C '#6250d6' }

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
function ChartHeight($n, $nbk = $nb) { 46 + $n * ($nbk * $barH + ($nbk - 1) * $barGap) + ($n - 1) * $groupGap + 44 }
$hGen = ChartHeight $genTests.Count; $hPP = ChartHeight $ppTests.Count
$hChat = 0
if ($chatTests.Count) { $hChat = 40 + (ChartHeight $chatTests.Count $chatBks.Count) + 56 }
$hKern = 0
if ($kernTests.Count) { $hKern = 40 + (ChartHeight $kernTests.Count $kernBks.Count) + 56 }
$top = 150

# system card contents, measured before the canvas size is fixed
$builds = @()
foreach ($b in $bks) { $mb = $doc.meta.($b.key); if ($mb -and $mb.build_commit) { $builds += "$($b.label): llama.cpp $($mb.build_commit)" } }
if ($sys.Kernelcore) { $builds += "Kernelcore $($sys.Kernelcore)" }
$items = @(
    @('Processor', $sys.processor), @('Cores', $sys.cores), @('Memory', $sys.memory), @('Graphics', $sys.graphics),
    @('Operating system', $sys.os), @('Computer', $sys.computer), @('Builds', ($builds -join "`n"))
) | Where-Object { $_[1] }
$items = @($items)
$tmpBmp = New-Object System.Drawing.Bitmap(1, 1)
$tmpG = [System.Drawing.Graphics]::FromImage($tmpBmp)
$itemH = @(foreach ($it in $items) { 21 + $tmpG.MeasureString($it[1], $fValue, [int]($leftW - 56)).Height + 16 })
# legend entries (swatch + name + detail), wrapped to the chart width
function LegendWidth($gr, $b) { 26 + $gr.MeasureString($b.label, $fValueB).Width + $gr.MeasureString($b.detail, $fLabel).Width }
$legendPos = @{}; $lx = 0; $lrow = 0
foreach ($b in $allBks) {
    $lw = LegendWidth $tmpG $b
    if ($lx -gt 0 -and $lx + $lw -gt $W - $M - $chartX) { $lx = 0; $lrow++ }
    $legendPos[$b.key] = @($lx, $lrow)
    $lx += $lw + 28
}
$legendExtra = 32 * $lrow
$tmpG.Dispose(); $tmpBmp.Dispose()
$cardH = 72 + ($itemH | Measure-Object -Sum).Sum + 10
$H = [int]([math]::Max($top + 60 + $legendExtra + $hGen + 40 + $hPP + $hChat + $hKern, $top + $cardH) + 120)

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
$g.DrawString('llama.cpp on this PC: CPU vs Kernelcore vs GPU', $fTitle, $bInk, $M, 40)
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
foreach ($b in $allBks) {
    $lx = $chartX + $legendPos[$b.key][0]; $ly = $top + 4 + 32 * $legendPos[$b.key][1]
    Bar $g $lx ($ly + 4) 18 18 $seriesColor[$b.key]
    $g.DrawString($b.label, $fValueB, $bInk, $lx + 26, $ly)
    $w1 = $g.MeasureString($b.label, $fValueB).Width
    $g.DrawString($b.detail, $fLabel, $bInk2, $lx + 26 + $w1, $ly + 3)
}

function Draw-Chart($title, $testList, [float]$y0, $bkList = $bks, $baseKey = 'cpu', $baseLabel = 'CPU') {
    $nb = $bkList.Count
    $g.DrawString($title, $fHead, $bInk, $chartX, $y0)
    $px = $chartX + $catW; $pw = $chartW - $catW - $labelRoom
    $vals = @(foreach ($t in $testList) { foreach ($b in $bkList) { $k = "$($b.key)|$t"; if ($best.ContainsKey($k)) { $best[$k].avg_ts } } })
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
        $tn = $t -replace '^(chat|kern) ', ''
        $cs = $g.MeasureString($tn, $fCat)
        $g.DrawString($tn, $fCat, $bInk, $chartX, $yy + ($gh - $cs.Height) / 2)
        $cpuV = $null; $kc = "$baseKey|$t"; if ($best.ContainsKey($kc)) { $cpuV = $best[$kc].avg_ts }
        foreach ($b in $bkList) {
            $k = "$($b.key)|$t"
            if ($best.ContainsKey($k)) {
                $v = $best[$k].avg_ts
                $bw = [float]($pw * $v / $max)
                Bar $g $px $yy $bw $barH $seriesColor[$b.key]
                $label = Fmt $v
                $lw = $g.MeasureString($label, $fValueB).Width
                $g.DrawString($label, $fValueB, $bInk, $px + $bw + 8, $yy + ($barH - 24) / 2)
                if ($b.key -ne $baseKey -and $cpuV) {
                    $ratio = $v / $cpuV
                    $note = if ($ratio -ge 2) { ('{0:N1}x ' -f $ratio) + $baseLabel } else { ('{0:+0;-0}% vs ' -f (($ratio - 1) * 100)) + $baseLabel }
                    $g.DrawString($note, $fLabel, $bInk2, $px + $bw + 12 + $lw, $yy + ($barH - 20) / 2)
                }
            }
            $yy += $barH + $barGap
        }
        $yy += $groupGap - $barGap
    }
}
$yc = $top + 60 + $legendExtra
Draw-Chart "Generation (tokens/s)" $genTests $yc
Draw-Chart "Prompt processing (tokens/s, by prompt length)" $ppTests ($yc + $hGen + 40)

# real chat: llama-server, one prompt, CPU backend vs Kernelcore
if ($chatTests.Count) {
    $ch = $doc.chat
    $cname = $ch.name
    if ($ch.size -and $ch.n_params) { $cname += ", $([math]::Round([double]$ch.size / 1GB, 2)) GiB, $([math]::Round([double]$ch.n_params / 1e9, 2))B parameters" }
    $y0 = $yc + $hGen + 40 + $hPP + 40
    Draw-Chart "Real chat: $cname, `"$($ch.prompt)`" (llama-server, tokens/s)" $chatTests $y0 $chatBks
    $parts = @()
    foreach ($b in $chatBks) {
        $rs = @($doc.records | Where-Object { $_.backend -eq $b.key -and $_.test -eq 'chat reply' })
        if ($rs.Count) {
            $fast = $rs | Sort-Object { [double]$_.seconds } | Select-Object -First 1
            $parts += "$($b.label) $(([double]$fast.seconds).ToString('N1')) s ($($fast.prompt_n) + $($fast.predicted_n) tokens)"
        }
    }
    $same = if ($ch.same_reply) { 'Both backends give the same reply.' }
            elseif ($null -ne $ch.same_words) { "The replies share their first $($ch.same_words) words, then word things differently at a near-tie (both in results.json)." }
            else { 'The replies differ (both in results.json).' }
    $noteRect = New-Object System.Drawing.RectangleF($chartX, ($y0 + (ChartHeight $chatTests.Count $chatBks.Count) - 6), $chartW, 48)
    $g.DrawString("Whole answer, fastest request (prompt + reply tokens): $($parts -join ', '). $same", $fLabel, $bInk2, $noteRect)
}

# Kernelcore's kernels on the CPU and on the GPU (nvcc): one layer's matmuls, not a llama.cpp run
if ($kernTests.Count) {
    $y0 = $yc + $hGen + 40 + $hPP + $hChat + 40
    Draw-Chart "Kernelcore kernels on the CPU vs the GPU: matmuls only (tokens/s)" $kernTests $y0 $kernBks 'Kernelcore' 'Kernelcore'
    $match = if ($doc.kernels.results_match) { 'GPU results checked against Kernelcore: same 8-bit activations, outputs within 3e-7.' } else { 'GPU and Kernelcore results DIFFER (see kc_kquant_vs_cuda).' }
    $kn = "kc_kquant_vs_cuda: one Llama 3.2 3B layer's 2 quantizations + 7 matmuls (random weights), tok/s = tokens / (28 x layer time); " +
          "no attention, norms or output layer, so not comparable with the bars above. $match"
    $kRect = New-Object System.Drawing.RectangleF($chartX, ($y0 + (ChartHeight $kernTests.Count $kernBks.Count) - 6), $chartW, 48)
    $g.DrawString($kn, $fLabel, $bInk2, $kRect)
}

# footer
$set = $doc.settings
$roundsText = if ([int]$set.rounds -eq 1) { '1 round' } else { "the best of $($set.rounds) rounds" }
$pt = if ($set.kc_threads) { [int]$set.kc_threads } else { 1 }
$pbt = if ($set.kc_batch_threads) { [int]$set.kc_batch_threads } else { $pt }
$kcText = if ($pt -eq $pbt) { "Kernelcore runs llama.cpp with $pt ggml thread$(if ($pt -ne 1) { 's' })" }
             else { "Kernelcore runs llama.cpp with $pt ggml thread$(if ($pt -ne 1) { 's' }) for generation and $pbt for prompts (llama-server -t $pt -tb $pbt)" }
$foot = "Each bar: $roundsText of llama-bench -r $($set.reps); the backends alternate within each round. " +
        "tgN = generating N tokens, ppN = processing an N-token prompt. $kcText; " +
        "its own thread pool runs prompt matmuls on all cores and everything else on the P-cores." +
        $(if ($chatTests.Count) { " Real chat: the best of $($set.rounds * $doc.chat.reps) request$(if ($set.rounds * $doc.chat.reps -ne 1) { 's' }) per backend through /v1/chat/completions (chat template, greedy$(if (-not $doc.chat.think) { ', no thinking' }), up to $($doc.chat.max_tokens) tokens), the server's own timings." } else { '' })
$g.DrawString($foot, $fFoot, $bInk2, (New-Object System.Drawing.RectangleF($M, ($H - 100), ($W - 2 * $M), 84)))

$png = Join-Path $OutDir 'bench_chart.png'
$bmp.Save($png, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
Write-Host "wrote $png"
Write-Host "      $(Join-Path $OutDir 'results.csv')"
$rows | Format-Table -AutoSize | Out-String | Write-Host
