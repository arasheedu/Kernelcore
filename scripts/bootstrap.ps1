<#
.SYNOPSIS
  Checks the Windows toolchain for Kernelcore and prints a pass/fail table.
  Never installs anything unless -FetchKath is given.

.PARAMETER FetchKath
  Download the pinned Booth Windows release into third_party/, verify the archive
  and kath.exe SHA-256 against cmake/BoothPin.cmake, and extract it.

.EXAMPLE
  .\scripts\bootstrap.ps1
  .\scripts\bootstrap.ps1 -FetchKath
#>
[CmdletBinding()]
param([switch]$FetchKath)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

# ---- pins (single source of truth: cmake/BoothPin.cmake) ---------------------
$pin = @{}
foreach ($line in Get-Content (Join-Path $root 'cmake\BoothPin.cmake')) {
    if ($line -match '^\s*set\((\w+)\s+"([^"]*)"\)') { $pin[$Matches[1]] = $Matches[2] }
}
$ver = $pin['BOOTH_VERSION']
foreach ($k in @($pin.Keys)) { $pin[$k] = $pin[$k].Replace('${BOOTH_VERSION}', $ver) }
$kathDir = Join-Path $root "third_party\booth-$ver\dist\booth-$ver-windows-x86_64"
$kathExe = Join-Path $kathDir 'kath.exe'

if ($FetchKath) {
    $dir = Join-Path $root "third_party\booth-$ver"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $zip = Join-Path $dir "booth-$ver-windows-x86_64.zip"
    $url = "https://github.com/Zaneham/Booth/releases/download/$($pin['BOOTH_TAG'])/booth-$ver-windows-x86_64.zip"
    if (-not (Test-Path $zip) -or (Get-FileHash $zip).Hash.ToLower() -ne $pin['BOOTH_WIN_ZIP_SHA256']) {
        Write-Host "downloading $url"
        Invoke-WebRequest $url -OutFile $zip -UserAgent 'kernelcore-bootstrap'
    }
    $h = (Get-FileHash $zip).Hash.ToLower()
    if ($h -ne $pin['BOOTH_WIN_ZIP_SHA256']) {
        Remove-Item $zip -Force
        throw "archive SHA-256 $h does not match pinned $($pin['BOOTH_WIN_ZIP_SHA256']); deleted"
    }
    Expand-Archive $zip -DestinationPath (Join-Path $dir 'dist') -Force
    $k = (Get-FileHash $kathExe).Hash.ToLower()
    if ($k -ne $pin['BOOTH_WIN_KATH_SHA256']) { throw "kath.exe SHA-256 $k does not match pinned value" }
    Write-Host "kath $ver ready at $kathExe"
}

# ---- checks ------------------------------------------------------------------
$rows = New-Object System.Collections.Generic.List[object]
function Add-Row($item, $required, $ok, $info) {
    $rows.Add([pscustomobject]@{ Item = $item; Required = $required; Status = $(if ($ok) { 'ok' } else { 'MISSING' }); Detail = $info })
}

Add-Row 'Windows x64' 'yes' ($env:PROCESSOR_ARCHITECTURE -eq 'AMD64') $env:PROCESSOR_ARCHITECTURE

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = $null
if (Test-Path $vswhere) {
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
Add-Row 'Visual Studio C++ x64 tools' 'yes' ([bool]$vs) $(if ($vs) { $vs } else { 'install VS 2022 "Desktop development with C++"' })
$ml64 = $null; $ninja = $null
if ($vs) {
    $ml64 = Get-ChildItem (Join-Path $vs 'VC\Tools\MSVC\*\bin\Hostx64\x64\ml64.exe') -ErrorAction SilentlyContinue | Select-Object -Last 1
    $ninja = Get-ChildItem (Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe') -ErrorAction SilentlyContinue
}
if (-not $ninja) { $ninja = Get-Command ninja.exe -ErrorAction SilentlyContinue }
Add-Row 'ml64 (MASM)' 'yes' ([bool]$ml64) $(if ($ml64) { $ml64.FullName } else { 'part of the VS C++ tools' })
Add-Row 'Ninja' 'yes' ([bool]$ninja) $(if ($ninja) { 'found' } else { 'bundled with VS CMake tools, or install Ninja' })

$cmake = Get-Command cmake.exe -ErrorAction SilentlyContinue
$cmakeOk = $false; $cmakeInfo = 'install CMake >= 3.26'
if ($cmake) {
    $v = [version]((& cmake --version | Select-Object -First 1) -replace '[^0-9.]', '')
    $cmakeOk = $v -ge [version]'3.26'
    $cmakeInfo = "$v"
}
Add-Row 'CMake >= 3.26' 'yes' $cmakeOk $cmakeInfo

# Kernel object sources: at least one must work.
$exeOk = (Test-Path $kathExe) -and ((Get-FileHash $kathExe).Hash.ToLower() -eq $pin['BOOTH_WIN_KATH_SHA256'])
Add-Row "kath.exe $ver (pinned)" 'one of 3' $exeOk $(if ($exeOk) { $kathExe } else { 'run: .\scripts\bootstrap.ps1 -FetchKath' })
$wslOk = $false
if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'   # a missing WSL kath writes to stderr; that is a result, not an error
    try {
        $out = & wsl.exe -e sh -c "$($pin['BOOTH_WSL_KATH_PATH']) --version 2>/dev/null"
        $wslOk = ($LASTEXITCODE -eq 0) -and ("$out" -match "Booth $ver")
    } catch {
        $wslOk = $false
    } finally {
        $ErrorActionPreference = $prev
    }
}
Add-Row "kath $ver in WSL" 'one of 3' $wslOk $(if ($wslOk) { $pin['BOOTH_WSL_KATH_PATH'] } else { 'optional: wsl.exe -e sh scripts/install_kath_wsl.sh' })
$pre = Get-ChildItem (Join-Path $root 'kernels\prebuilt\*.o') -ErrorAction SilentlyContinue
Add-Row 'prebuilt kernel objects' 'one of 3' ([bool]$pre) "$(@($pre).Count) object(s)"

$clang = Get-Command clang-cl.exe -ErrorAction SilentlyContinue
if (-not $clang -and $vs) { $clang = Get-ChildItem (Join-Path $vs 'VC\Tools\Llvm\x64\bin\clang-cl.exe') -ErrorAction SilentlyContinue }
Add-Row 'clang-cl (second compiler)' 'no' ([bool]$clang) $(if ($clang) { 'found' } else { 'optional: VS component "C++ Clang tools"' })

$rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

$requiredMissing = $rows | Where-Object { $_.Required -eq 'yes' -and $_.Status -ne 'ok' }
$anyKernelSource = $exeOk -or $wslOk -or [bool]$pre
if ($requiredMissing -or -not $anyKernelSource) {
    Write-Host 'bootstrap: FAILED - fix the MISSING required items above.' -ForegroundColor Red
    exit 1
}
Write-Host 'bootstrap: OK - build with .\scripts\build.ps1' -ForegroundColor Green
exit 0
