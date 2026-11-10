<#
.SYNOPSIS
  Real llama.cpp decode speed for comparison with kc_llm_bench: asks a local Ollama server
  to generate with the model on the GPU and on the CPU only, and reports tokens per second
  from Ollama's own timings (eval_count / eval_duration).

.EXAMPLE
  .\scripts\ollama_reference.ps1                    # llama3.2 (3B, Q4_K_M), 128 tokens
  .\scripts\ollama_reference.ps1 -Model llama3.2 -Tokens 256
#>
[CmdletBinding()]
param(
    [string]$Model = 'llama3.2',
    [int]$Tokens = 128,
    [string]$Server = 'http://127.0.0.1:11434'
)
$ErrorActionPreference = 'Stop'

try { $v = Invoke-RestMethod "$Server/api/version" -TimeoutSec 5 }
catch { throw "No Ollama server at $Server. Start the Ollama app (or 'ollama serve') first." }

$prompt = 'Write a detailed, factual history of the printing press, from Gutenberg to the industrial era.'

function Measure-Ollama([string]$label, [int]$numGpu) {
    $body = @{
        model      = $Model
        prompt     = $prompt
        stream     = $false
        keep_alive = 0  # unload afterwards so the next run starts clean
        options    = @{ num_predict = $Tokens; temperature = 0; seed = 1; num_gpu = $numGpu }
    } | ConvertTo-Json -Depth 4
    # One short warm-up request loads the model with these settings.
    $warm = $body | ConvertFrom-Json
    $warm.options.num_predict = 4
    $warm.keep_alive = '5m'
    Invoke-RestMethod "$Server/api/generate" -Method Post -Body ($warm | ConvertTo-Json -Depth 4) -ContentType 'application/json' -TimeoutSec 900 | Out-Null
    $r = Invoke-RestMethod "$Server/api/generate" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 1800
    [pscustomobject]@{
        Run              = $label
        'decode tok/s'   = [math]::Round($r.eval_count / ($r.eval_duration / 1e9), 2)
        'decode ms/token'= [math]::Round(($r.eval_duration / 1e6) / $r.eval_count, 1)
        'prompt tok/s'   = [math]::Round($r.prompt_eval_count / ($r.prompt_eval_duration / 1e9), 1)
        tokens           = $r.eval_count
    }
}

Write-Host "Ollama $($v.version), model $Model, $Tokens tokens, temperature 0"
$rows = @(
    Measure-Ollama 'GPU (all layers offloaded)' 999
    Measure-Ollama 'CPU only (num_gpu 0)' 0
)
$rows | Format-Table -AutoSize
