<#
.SYNOPSIS
  Real-model check: Ollama's llama3.2 (Llama 3.2 3B Instruct, Q4_K_M GGUF) run through pcuda's CUDA C
  kernels on the CPU, compared with llama.cpp on the same file.

  1. Finds the model blob in Ollama's store (no copy).
  2. Starts llama.cpp's llama-server on the CPU, tokenizes the prompt and generates -Tokens tokens
     greedily with the top-5 log-probabilities of every step (the reference).
  3. Times Ollama on the CPU on the same prompt (raw, greedy) for the speed comparison.
  4. Runs pcuda_llm_real: bit-exactness against its C++ mirror, teacher forcing on llama.cpp's tokens,
     and a timed greedy free run.

.EXAMPLE
  .\scripts\real_model_check.ps1
  .\scripts\real_model_check.ps1 -Prompt "Write a haiku about the sea." -Tokens 64
#>
[CmdletBinding()]
param(
    [string]$Prompt = 'The history of the Roman Empire began when',
    [int]$Tokens = 128,
    [string]$Model = 'llama3.2',
    [string]$LlamaServer = 'C:\BitsPlus\VisOllamInference\build-llama\bin\Release\llama-server.exe',
    [string]$Ollama = 'http://127.0.0.1:11434',
    [string]$Bench = '',
    [int]$Port = 8089
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $Bench) { $Bench = Join-Path $root 'build\windows-msvc-nvidia\pcuda_llm_real.exe' }
if (-not (Test-Path $Bench)) { $Bench = Join-Path $root 'build\windows-msvc\pcuda_llm_real.exe' }

# 1. The GGUF file inside Ollama's model store.
$models = if ($env:OLLAMA_MODELS) { $env:OLLAMA_MODELS } else { Join-Path $env:USERPROFILE '.ollama\models' }
$name, $tag = $Model.Split(':')
if (-not $tag) { $tag = 'latest' }
$manifest = Get-Content (Join-Path $models "manifests\registry.ollama.ai\library\$name\$tag") | ConvertFrom-Json
$digest = ($manifest.layers | Where-Object { $_.mediaType -eq 'application/vnd.ollama.image.model' }).digest
$gguf = Join-Path $models ("blobs\" + $digest.Replace(':', '-'))
Write-Host "Model file: $gguf"

# 2. llama.cpp reference (CPU).
$ref = Join-Path $env:TEMP 'pcuda_real_ref.txt'
$server = Start-Process -FilePath $LlamaServer -PassThru -WindowStyle Hidden `
    -ArgumentList @('-m', $gguf, '--port', "$Port", '-ngl', '0', '-c', '4096', '--no-webui') `
    -RedirectStandardError (Join-Path $env:TEMP 'pcuda_real_server.txt') -RedirectStandardOutput (Join-Path $env:TEMP 'pcuda_real_server_out.txt')
try {
    $ok = $false
    for ($i = 0; $i -lt 120 -and -not $ok; $i++) {
        Start-Sleep -Milliseconds 500
        try { $ok = (Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).status -eq 'ok' } catch {}
    }
    if (-not $ok) { throw 'llama-server did not start' }
    $ids = (Invoke-RestMethod "http://127.0.0.1:$Port/tokenize" -Method Post -ContentType 'application/json' `
            -Body (@{ content = $Prompt; add_special = $true } | ConvertTo-Json)).tokens
    $body = @{ prompt = @($ids); n_predict = $Tokens; temperature = 0; top_k = 1; n_probs = 5; cache_prompt = $false; ignore_eos = $true }
    $c = Invoke-RestMethod "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -Body ($body | ConvertTo-Json -Depth 4)
    $lines = @('prompt ' + ($ids -join ' '))
    foreach ($p in $c.completion_probabilities) {
        $top = ($p.top_logprobs | ForEach-Object { "$($_.id) $($_.logprob)" }) -join ' '
        $lines += "gen $($p.id) $($p.logprob) $top"
    }
    Set-Content -Path $ref -Value $lines -Encoding ascii
    Write-Host "llama.cpp: prompt $($ids.Count) tokens, generated $($c.completion_probabilities.Count); $([math]::Round($c.timings.predicted_per_second, 2)) tok/s"
} finally {
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
}

# 3. Ollama on the CPU, same prompt, greedy.
$o = Invoke-RestMethod "$Ollama/api/generate" -Method Post -ContentType 'application/json' -Body (@{
        model = $Model; prompt = $Prompt; raw = $true; stream = $false
        options = @{ temperature = 0; top_k = 1; num_predict = $Tokens; num_gpu = 0; seed = 1 } } | ConvertTo-Json -Depth 4)
$ollamaTps = $o.eval_count / ($o.eval_duration / 1e9)
Write-Host ("Ollama CPU: {0} tokens, {1:N2} tok/s" -f $o.eval_count, $ollamaTps)
for ($i = 0; $i -lt 60; $i++) {  # let Ollama unload and the CPU cool down
    $ps = Invoke-RestMethod "$Ollama/api/ps"
    if (-not $ps.models -or $ps.models.Count -eq 0) { break }
    Start-Sleep -Seconds 2
}
Start-Sleep -Seconds 15

# 4. pcuda.
Write-Host ""
$out = & $Bench --gguf $gguf --ref $ref
$out | Where-Object { $_ -notmatch '^RESULT' } | ForEach-Object { Write-Host $_ }
$code = $LASTEXITCODE
$r = (($out | Where-Object { $_ -match '^RESULT_REAL,' }) -split ',')
if ($r.Count -ge 11) {
    Write-Host ""
    Write-Host ("Decode speed on the real model (Llama 3.2 3B Q4_K_M, {0} GB/token):" -f $r[10])
    [pscustomobject]@{ Run = 'pcuda (CUDA C kernels, simd path)'; 'tok/s' = [math]::Round([double]$r[9], 2); 'ms/token' = [math]::Round([double]$r[8], 1) },
    [pscustomobject]@{ Run = 'Ollama (llama.cpp), CPU only'; 'tok/s' = [math]::Round($ollamaTps, 2); 'ms/token' = [math]::Round(1000 / $ollamaTps, 1) } |
        Format-Table -AutoSize | Out-String -Width 200 | Write-Host
}
exit $code
