<#
.SYNOPSIS
  Configure, build and (optionally) test pcuda from any PowerShell window.
  Locates Visual Studio 2022 with vswhere and loads the x64 developer environment,
  so cl, ml64 and ninja are found without opening a Developer prompt.

.EXAMPLE
  .\scripts\build.ps1                         # windows-msvc preset, build + test
  .\scripts\build.ps1 -Preset windows-msvc-asan
  .\scripts\build.ps1 -NoTest
  .\scripts\build.ps1 -Fresh                  # delete the preset's build directory first
#>
[CmdletBinding()]
param(
    [string]$Preset = 'windows-msvc',
    [switch]$NoTest,
    [switch]$Fresh
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

function Enter-VsDevEnvironment {
    if ((Get-Command cl.exe -ErrorAction SilentlyContinue) -and (Get-Command ml64.exe -ErrorAction SilentlyContinue) -and $env:VSCMD_ARG_TGT_ARCH -eq 'x64') {
        return
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { throw 'vswhere.exe not found: install Visual Studio 2022 (Desktop development with C++).' }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw 'No Visual Studio installation with the x64 C++ tools was found.' }
    $env:PATH = "$(Split-Path $vswhere);$env:PATH"  # VsDevCmd itself calls vswhere
    $devshell = Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
    Import-Module $devshell
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
    foreach ($t in 'cl.exe', 'ml64.exe', 'ninja.exe', 'cmake.exe') {
        if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { throw "$t not found after loading the VS environment." }
    }
}

Enter-VsDevEnvironment
Push-Location $root
try {
    if ($Fresh) {
        $dir = Join-Path $root "build\$Preset"
        if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    }
    # Native tools write warnings to stderr. When this script's output is captured, Windows
    # PowerShell turns those lines into errors, so rely on exit codes instead.
    $ErrorActionPreference = 'Continue'
    cmake --preset $Preset
    if ($LASTEXITCODE -ne 0) { throw "configure failed ($LASTEXITCODE)" }
    cmake --build --preset $Preset
    if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }
    if (-not $NoTest) {
        ctest --preset $Preset
        if ($LASTEXITCODE -ne 0) { throw "tests failed ($LASTEXITCODE)" }
    }
} finally {
    $ErrorActionPreference = 'Stop'
    Pop-Location
}
