<#
.SYNOPSIS
  Like-for-like token generation: real llama.cpp (Ollama, llama3.2 = Llama 3.2 3B Q4_K_M)
  versus the same decode loop built from Kernelcore's .cu kernels on every code path and on CUDA.

  Matched: model shapes (Llama 3.2 3B), weight bytes per token (Q4: 2.0 GB vs 2.02 GB),
  prompt length (taken from Ollama's own prompt_eval_count), number of generated tokens,
  attention over the growing context, logits returned to the host every token.
  Not matched (inherent): Ollama runs real weights through hand-tuned llama.cpp kernels
  (AVX2 / CUDA); Kernelcore runs one CUDA C source per kernel through Booth, MSVC or nvcc.

.EXAMPLE
  .\scripts\compare_llm.ps1                    # needs the windows-msvc-nvidia build and Ollama running
  .\scripts\compare_llm.ps1 -Tokens 64 -Budget 10
#>
[CmdletBinding()]
param(
    [string]$Model = 'llama3.2',
    [int]$Tokens = 128,
    [double]$Budget = 20,
    [string]$Server = 'http://127.0.0.1:11434',
    [string]$Bench = '',
    # Kernelcore paths to time; booth and native are slow and heat the CPU before the simd run
    [string]$Paths = 'simd,cuda'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $Bench) { $Bench = Join-Path $root 'build\windows-msvc-nvidia\kc_llm_bench.exe' }
if (-not (Test-Path $Bench)) { throw "$Bench not found: build with .\scripts\build.ps1 -Preset windows-msvc-nvidia" }
try { $v = Invoke-RestMethod "$Server/api/version" -TimeoutSec 5 }
catch { throw "No Ollama server at $Server. Start the Ollama app (or 'ollama serve') first." }

$prompt = 'Write a detailed, factual history of the printing press, from Gutenberg to the industrial era.'

function Invoke-Ollama([int]$numGpu) {
    $opts = @{ num_predict = $Tokens; temperature = 0; seed = 1; num_gpu = $numGpu }
    $warm = @{ model = $Model; prompt = $prompt; stream = $false; keep_alive = '5m'
               options = @{ num_predict = 4; temperature = 0; seed = 1; num_gpu = $numGpu } } | ConvertTo-Json -Depth 4
    Invoke-RestMethod "$Server/api/generate" -Method Post -Body $warm -ContentType 'application/json' -TimeoutSec 900 | Out-Null
    $body = @{ model = $Model; prompt = $prompt; stream = $false; keep_alive = 0; options = $opts } | ConvertTo-Json -Depth 4
    Invoke-RestMethod "$Server/api/generate" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 3600
}

# Weight bytes per token for Ollama: the GGUF file (tied embeddings are read by the output layer).
$fileBytes = $null
try {
    $manifest = Get-Content (Join-Path $env:USERPROFILE ".ollama\models\manifests\registry.ollama.ai\library\$Model\latest") -Raw | ConvertFrom-Json
    $fileBytes = ($manifest.layers | Where-Object mediaType -eq 'application/vnd.ollama.image.model').size
} catch { }

Write-Host "Ollama $($v.version), model $Model, $Tokens generated tokens, temperature 0"
$gpu = Invoke-Ollama 999
$cpu = Invoke-Ollama 0
$promptTokens = [int]$gpu.prompt_eval_count
Write-Host ("Ollama: prompt {0} tokens; GPU {1:N1} tok/s, CPU {2:N1} tok/s" -f $promptTokens,
    ($gpu.eval_count / ($gpu.eval_duration / 1e9)), ($cpu.eval_count / ($cpu.eval_duration / 1e9)))
# Same conditions for both: wait until Ollama has unloaded the model (its runner exits), then
# let the CPU cool for a moment, so the bench does not start while the runner is shutting down.
$deadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $deadline) {
    $ps = Invoke-RestMethod "$Server/api/ps" -TimeoutSec 5
    if (-not $ps.models -or $ps.models.Count -eq 0) { break }
    Start-Sleep -Seconds 2
}
Start-Sleep -Seconds 15
Write-Host ""
Write-Host "Running kc_llm_bench --model 3b --weights q4x8q8 --prompt $promptTokens --tokens $Tokens --paths $Paths ..."
$out = & $Bench --model 3b --weights q4x8q8 --prompt $promptTokens --tokens $Tokens --budget $Budget --paths $Paths
$out | Where-Object { $_ -notmatch '^RESULT' } | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) { throw "kc_llm_bench failed ($LASTEXITCODE)" }

$info = ($out | Where-Object { $_ -match '^RESULT_INFO,' }) -split ','
$benchGb = [double]$info[1]
$rows = @()
foreach ($r in ($out | Where-Object { $_ -match '^RESULT,' })) {
    $f = $r -split ','
    $rows += [pscustomobject]@{ Run = "Kernelcore kernels: $($f[1])"; 'ms/token' = [math]::Round([double]$f[2], 1)
        'tok/s' = [math]::Round([double]$f[3], 2); 'weights GB/s' = [math]::Round([double]$f[4], 1)
        'GB/token' = [math]::Round($benchGb, 2); 'tokens timed' = "$($f[5])/$Tokens" }
}
foreach ($o in @(@{n = 'Ollama llama.cpp, GPU'; r = $gpu }, @{n = 'Ollama llama.cpp, CPU only'; r = $cpu })) {
    $tps = $o.r.eval_count / ($o.r.eval_duration / 1e9)
    $gbt = if ($fileBytes) { $fileBytes / 1e9 } else { [double]::NaN }
    $rows += [pscustomobject]@{ Run = $o.n; 'ms/token' = [math]::Round(1000 / $tps, 1); 'tok/s' = [math]::Round($tps, 2)
        'weights GB/s' = [math]::Round($gbt * $tps, 1); 'GB/token' = [math]::Round($gbt, 2)
        'tokens timed' = "$($o.r.eval_count)/$Tokens" }
}
Write-Host ""
Write-Host "Like-for-like decode: Llama 3.2 3B shapes, ~2.0 GB of 4-bit weights per token, prompt $promptTokens, $Tokens tokens"
$rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
