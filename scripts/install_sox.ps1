param(
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-SoxCommandPath {
    $cmd = Get-Command sox -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        return $null
    }
    return $cmd.Source
}

function Find-WingetSoxDir {
    $root = Join-Path $env:LOCALAPPDATA "Microsoft\\WinGet\\Packages"
    if (-not (Test-Path $root)) {
        return $null
    }

    $pkgDirs = Get-ChildItem -Path $root -Directory -Filter "ChrisBagwell.SoX*" -ErrorAction SilentlyContinue
    foreach ($pkgDir in $pkgDirs) {
        $exe = Get-ChildItem -Path $pkgDir.FullName -Recurse -Filter "sox.exe" -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -ne $exe) {
            return $exe.DirectoryName
        }
    }
    return $null
}

function Ensure-UserPathContains([string]$dirPath) {
    if ([string]::IsNullOrWhiteSpace($dirPath)) {
        return
    }
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($null -eq $userPath) {
        $userPath = ""
    }
    $parts = $userPath.Split(';', [System.StringSplitOptions]::RemoveEmptyEntries)
    if ($parts -contains $dirPath) {
        return
    }
    $newPath = if ([string]::IsNullOrWhiteSpace($userPath)) {
        $dirPath
    } else {
        "$userPath;$dirPath"
    }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "[install_sox] Added to user PATH: $dirPath"
}

function Ensure-SessionPathContains([string]$dirPath) {
    if ([string]::IsNullOrWhiteSpace($dirPath)) {
        return
    }
    $currentPath = $env:PATH
    if ($currentPath -match [Regex]::Escape($dirPath)) {
        return
    }
    $env:PATH = "$dirPath;$currentPath"
}

$found = Get-SoxCommandPath
if ($found) {
    Write-Host "[install_sox] SoX already available: $found"
    & sox --version
    exit 0
}

$wingetDir = Find-WingetSoxDir
if ($wingetDir) {
    Ensure-SessionPathContains -dirPath $wingetDir
    Ensure-UserPathContains -dirPath $wingetDir
    $found = Get-SoxCommandPath
    if ($found) {
        Write-Host "[install_sox] SoX activated from WinGet cache: $found"
        & sox --version
        exit 0
    }
}

if ($SkipInstall) {
    throw "SoX not found and -SkipInstall specified."
}

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "winget is not available. Install SoX manually or use a shell with winget."
}

Write-Host "[install_sox] Installing SoX via winget..."
winget install -e --id ChrisBagwell.SoX --scope user --accept-package-agreements --accept-source-agreements --silent

$wingetDir = Find-WingetSoxDir
if (-not $wingetDir) {
    throw "SoX installation finished but sox.exe was not found in WinGet cache."
}

Ensure-SessionPathContains -dirPath $wingetDir
Ensure-UserPathContains -dirPath $wingetDir

$found = Get-SoxCommandPath
if (-not $found) {
    throw "SoX installation completed, but current shell cannot resolve `sox`. Restart shell and retry."
}

Write-Host "[install_sox] SoX installed: $found"
& sox --version

