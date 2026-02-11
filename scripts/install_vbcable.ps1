param(
    [string]$DownloadUrl = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip",
    [string]$WorkDir = "tmp\\vbcable\\install",
    [switch]$NoElevate,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Ensure-Administrator {
    if ($DryRun) {
        return
    }
    if (Test-IsAdministrator) {
        return
    }
    if ($NoElevate) {
        throw "Administrator privilege is required. Re-run PowerShell as Administrator."
    }

    $argList = @(
        "-ExecutionPolicy", "Bypass",
        "-File", ('"{0}"' -f $PSCommandPath),
        "-NoElevate",
        "-DownloadUrl", ('"{0}"' -f $DownloadUrl),
        "-WorkDir", ('"{0}"' -f $WorkDir)
    )
    if ($DryRun) {
        $argList += "-DryRun"
    }

    Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList ($argList -join " ")
    exit 0
}

function Select-InfPath([string]$ExtractDir) {
    $candidates = @(
        "vbMmeCable64_win7.inf",
        "vbMmeCable64_vista.inf",
        "vbMmeCable64_2003.inf"
    )
    foreach ($name in $candidates) {
        $path = Join-Path $ExtractDir $name
        if (Test-Path $path) {
            return $path
        }
    }
    throw "Unable to find a supported x64 VB-CABLE INF file in $ExtractDir"
}

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
}

function Get-DetectedCableDevices {
    $devices = Get-CimInstance Win32_SoundDevice |
        Where-Object { $_.Name -match "CABLE|VB-Audio" } |
        Select-Object -ExpandProperty Name
    return @($devices)
}

function Resolve-SetupExePath([string]$ExtractDir) {
    $candidate = if ([Environment]::Is64BitOperatingSystem) {
        "VBCABLE_Setup_x64.exe"
    } else {
        "VBCABLE_Setup.exe"
    }
    $path = Join-Path $ExtractDir $candidate
    if (Test-Path $path) {
        return $path
    }
    return $null
}

Ensure-Administrator

$repoRoot = Get-RepoRoot
$targetDir = Join-Path $repoRoot $WorkDir
$zipPath = Join-Path $targetDir "VBCABLE_Driver_Pack.zip"
$extractDir = Join-Path $targetDir "pack"

New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
Write-Host "[install_vbcable] Downloading package..."
Invoke-WebRequest -Uri $DownloadUrl -OutFile $zipPath
Write-Host "[install_vbcable] Extracting package..."
Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

$infPath = Select-InfPath -ExtractDir $extractDir
$setupExePath = Resolve-SetupExePath -ExtractDir $extractDir
$pnputilArgs = @("/add-driver", ('"{0}"' -f $infPath), "/install")
Write-Host ("[install_vbcable] pnputil {0}" -f ($pnputilArgs -join " "))

if (-not $DryRun) {
    $proc = Start-Process -FilePath "pnputil.exe" -ArgumentList ($pnputilArgs -join " ") -Wait -PassThru -NoNewWindow
    if ($proc.ExitCode -ne 0) {
        throw "pnputil failed with exit code $($proc.ExitCode)"
    }
    Start-Sleep -Seconds 3
}

$detected = @(Get-DetectedCableDevices | Where-Object { $_ -and $_.Trim() })
if ($detected.Length -gt 0) {
    Write-Host "[install_vbcable] Detected virtual audio devices:"
    $detected | ForEach-Object { Write-Host (" - {0}" -f $_) }
    Write-Host "[install_vbcable] You can now select 'Realtime Voice Changer Virtual Mic (...)' in the web UI."
}
else {
    Write-Warning "No VB-CABLE device detected yet. Reboot Windows once, then check /api/v1/audio/devices again."
    if ($setupExePath) {
        Write-Warning "If reboot still does not expose CABLE devices, run official setup as Administrator:"
        Write-Host ("  Start-Process -Verb RunAs `"{0}`"" -f $setupExePath)
    }
}
