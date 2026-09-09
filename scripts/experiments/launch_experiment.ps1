# AIC-VideoHighlight unified experiment launcher (Windows).
#
# Usage:
#   .\scripts\experiments\launch_experiment.ps1 <experiment> [-Resume] [-ValidateOnly] [-DryRun]
#                  [-Target WINDOWS|AUTODL] [-List] [-Help] [-CloseDelaySec N] [-Inline]
#
# The experiment registry (target, GPU requirement, canonical runner) lives in
# the Python stage launchers (scripts/experiments/<stage>/run.py) and is queried
# at runtime via scripts/experiments/registry.py. This script never hardcodes
# experiment names. See scripts/experiments/README.md.

param(
    [Parameter(Position = 0)]
    [string] $Experiment = "",
    [switch] $Resume,
    [switch] $ValidateOnly,
    [switch] $DryRun,
    [string] $Target = "",
    [switch] $List,
    [switch] $Help,
    [int] $CloseDelaySec = 5,
    [switch] $Inline,
    [switch] $Child,
    [switch] $ImportOnly
)

$script:ValidTargets = @("WINDOWS", "AUTODL")

function Get-LauncherRepoRoot {
    $experimentsDir = Split-Path -Parent $PSScriptRoot
    return Split-Path -Parent $experimentsDir
}

function Resolve-PythonPath {
    if ($env:AIC_PYTHON) { return $env:AIC_PYTHON }
    return "python"
}

function Test-ExperimentNameFormat {
    param([string] $Name)
    return ($Name -match '^[A-Za-z0-9_.\-]+$')
}

function Invoke-Registry {
    # Returns @{ Ok = $true; Data = <psobject> } or @{ Ok = $false; Message = <string> }
    param(
        [string] $PythonPath,
        [string] $RegistryScript,
        [string[]] $RegistryArgs
    )
    $output = & $PythonPath $RegistryScript @RegistryArgs 2>&1
    $exitCode = $LASTEXITCODE
    $text = ($output | ForEach-Object { $_.ToString() }) -join "`n"
    if ($exitCode -ne 0) {
        return @{ Ok = $false; Message = $text.Trim() }
    }
    try {
        $data = $text | ConvertFrom-Json
        return @{ Ok = $true; Data = $data }
    } catch {
        return @{ Ok = $false; Message = "registry output was not valid JSON:`n$text" }
    }
}

function Get-RegistryList {
    param([hashtable] $Context)
    return Invoke-Registry -PythonPath $Context.PythonPath `
        -RegistryScript $Context.RegistryScript -RegistryArgs @("list")
}

function Get-LaunchSpec {
    param([hashtable] $Context, [string] $Name)
    return Invoke-Registry -PythonPath $Context.PythonPath `
        -RegistryScript $Context.RegistryScript -RegistryArgs @("describe", "--experiment", $Name)
}

function Resolve-SshHostCandidates {
    if ($env:AIC_AUTODL_SSH_HOST) { return @($env:AIC_AUTODL_SSH_HOST) }
    return @("autodl", "autodl-stage1")
}

function Test-SshConnectivity {
    param([string] $SshHost)
    $output = & ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new $SshHost "echo aic_ssh_ok" 2>$null
    return ($LASTEXITCODE -eq 0 -and ($output -join " ") -match "aic_ssh_ok")
}

function Resolve-SshHost {
    # Returns the first reachable SSH host alias, or throws with guidance.
    $candidates = Resolve-SshHostCandidates
    $attempted = @()
    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        $attempted += $candidate
        Write-Host ("[launcher] SSH preflight: testing host '{0}' ..." -f $candidate)
        if (Test-SshConnectivity -SshHost $candidate) {
            Write-Host ("[launcher] SSH preflight OK: using host '{0}'" -f $candidate)
            return $candidate
        }
        Write-Host ("[launcher] SSH preflight failed for host '{0}'" -f $candidate)
    }
    throw @"
No reachable SSH host for AutoDL (tried: $($attempted -join ', ')).
Check:
  1. The AutoDL instance is running (the launcher never starts/stops instances).
  2. A Host alias exists in ~/.ssh/config (e.g. 'autodl' or 'autodl-stage1'),
     or set the environment variable AIC_AUTODL_SSH_HOST to your alias.
  3. Key-based (non-interactive) authentication works: ssh <alias> echo ok
Credentials, IPs and keys are never stored in this repository.
"@
}

function Test-RemoteGpuAvailable {
    param([string] $SshHost)
    $output = & ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new $SshHost "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L" 2>$null
    return ($LASTEXITCODE -eq 0 -and ($output -join " ") -match "GPU")
}

function Assert-SafeToken {
    param([string] $Token, [string] $Kind)
    if ($Token -match '[`"''\r\n;$]') {
        throw "unsafe character in $Kind (refusing to build command): $Token"
    }
}

function Build-RunnerArgs {
    param([string] $Experiment, [switch] $Resume, [switch] $DryRun, [switch] $ValidateOnly)
    Assert-SafeToken -Token $Experiment -Kind "experiment name"
    $flags = @("--experiment", $Experiment)
    if ($Resume) { $flags += "--resume" }
    if ($DryRun) { $flags += "--dry-run" }
    if ($ValidateOnly) { $flags += "--validate-only" }
    return $flags
}

function Build-RemoteCommand {
    param([psobject] $Spec, [string[]] $RunnerArgs, [string] $RemotePython)
    if (-not $Spec.remote_repo) {
        throw "launch spec has no remote_repo (configs/environments/autodl.json missing 'repo')."
    }
    Assert-SafeToken -Token $Spec.remote_repo -Kind "remote repo path"
    Assert-SafeToken -Token $Spec.stage_launcher -Kind "stage launcher path"
    Assert-SafeToken -Token $RemotePython -Kind "remote python"
    foreach ($arg in $RunnerArgs) { Assert-SafeToken -Token $arg -Kind "runner argument" }
    $inner = "cd {0} && export PYTHONPATH=src PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 && {1} {2} {3}" -f `
        $Spec.remote_repo, $RemotePython, $Spec.stage_launcher, ($RunnerArgs -join " ")
    return "bash -lc '$inner'"
}

function Show-LaunchHeader {
    param(
        [psobject] $Spec, [string] $SshHost, [string] $TargetOverride,
        [string] $RemotePython, [string] $ResolvedCommand
    )
    $target = $Spec.target
    if ($TargetOverride) { $target = "$target (override: $TargetOverride)" }
    Write-Host "=================================================================="
    Write-Host " AIC-VideoHighlight Experiment Launcher"
    Write-Host "=================================================================="
    Write-Host (" Experiment : {0}" -f $Spec.experiment)
    Write-Host (" Execution  : {0}" -f $target)
    Write-Host (" GPU        : {0}" -f $Spec.gpu)
    if ($Spec.target -eq "AUTODL" -or $TargetOverride -eq "AUTODL") {
        Write-Host (" SSH host   : {0}" -f $SshHost)
        Write-Host (" Remote repo: {0}" -f $Spec.remote_repo)
        Write-Host (" Remote py  : {0}   (override via AIC_AUTODL_PYTHON)" -f $RemotePython)
    }
    Write-Host (" Runner     : {0}" -f $Spec.stage_launcher)
    Write-Host (" Config     : {0}" -f $Spec.config)
    Write-Host (" Command    : {0}" -f $ResolvedCommand)
    Write-Host "=================================================================="
}

function Start-StreamingProcess {
    # Runs a process attached to the current console so native stdout/stderr
    # (progress bars using CR, SSH output and interactive prompts) stream live
    # and byte-exact; returns the process exit code.
    param([string] $FilePath, [string[]] $ArgumentList, [string] $WorkingDirectory)
    $quoted = foreach ($arg in $ArgumentList) {
        if ($arg -match '\s') { '"{0}"' -f $arg } else { $arg }
    }
    $process = Start-Process -FilePath $FilePath -ArgumentList $quoted `
        -WorkingDirectory $WorkingDirectory -NoNewWindow -Wait -PassThru
    return $process.ExitCode
}

function Write-SuccessBanner {
    param([string] $ExperimentName, [int] $CloseDelaySec, [bool] $WindowMode)
    Write-Host ""
    Write-Host "=================================================================="
    Write-Host " AIC-VideoHighlight"
    Write-Host " Experiment completed successfully."
    Write-Host ""
    Write-Host " Experiment:"
    Write-Host " $ExperimentName"
    Write-Host ""
    Write-Host " Exit code:"
    Write-Host " 0"
    Write-Host "=================================================================="
    if ($WindowMode) {
        Write-Host (" Closing in {0} seconds..." -f $CloseDelaySec)
        Start-Sleep -Seconds $CloseDelaySec
    }
}

function Write-FailureBanner {
    param([string] $ExperimentName, [int] $ExitCode, [string] $Detail, [bool] $WindowMode)
    Write-Host ""
    Write-Host "=================================================================="
    Write-Host " AIC-VideoHighlight"
    Write-Host " EXPERIMENT FAILED"
    Write-Host ""
    Write-Host " Experiment:"
    Write-Host " $ExperimentName"
    Write-Host ""
    Write-Host " Exit code:"
    Write-Host " $ExitCode"
    Write-Host "=================================================================="
    if ($Detail) { Write-Host $Detail }
    if ($WindowMode) {
        Write-Host ""
        Write-Host " Press Enter to close..."
        try { [void](Read-Host) } catch { }
    }
}

function Show-Help {
    Write-Host @"
AIC-VideoHighlight unified experiment launcher.

Usage:
  .\scripts\experiments\launch_experiment.ps1 <experiment> [options]

Options:
  -Resume         forward --resume to the canonical runner
  -ValidateOnly   forward --validate-only to the canonical runner
  -DryRun         resolve the launch spec and print the command; do NOT execute
  -Target <t>     override the registry execution target: WINDOWS | AUTODL
  -List           list all registered experiments (from the Python registry)
  -CloseDelaySec  seconds before a successful child window closes (default 5)
  -Inline         run in this console instead of a new PowerShell window
  -Help           show this help

Notes:
  - The experiment registry (target / GPU / canonical runner) is owned by the
    Python stage launchers; this script never hardcodes experiments.
  - AutoDL experiments require a running instance and an SSH alias in
    ~/.ssh/config (or the AIC_AUTODL_SSH_HOST environment variable).
  - A failing experiment never auto-closes its window, so the traceback,
    stderr and exit code stay visible.

Examples:
  .\scripts\experiments\launch_experiment.ps1 -List
  .\scripts\experiments\launch_experiment.ps1 <experiment> -DryRun
  .\scripts\experiments\launch_experiment.ps1 <experiment>
  .\scripts\experiments\launch_experiment.ps1 <experiment> -Resume
"@
}

function Invoke-ExperimentRun {
    # Core lifecycle. Returns the process exit code; never closes the console itself.
    param(
        [hashtable] $Context,
        [string] $ExperimentName,
        [switch] $Resume,
        [switch] $DryRun,
        [switch] $ValidateOnly,
        [string] $TargetOverride,
        [bool] $WindowMode
    )
    if ([string]::IsNullOrWhiteSpace($ExperimentName)) {
        Write-FailureBanner -ExperimentName "<none>" -ExitCode 2 `
            -Detail "No experiment specified. Run with -Help for usage, or -List for registered experiments." `
            -WindowMode $WindowMode
        return 2
    }
    if (-not (Test-ExperimentNameFormat -Name $ExperimentName)) {
        Write-FailureBanner -ExperimentName $ExperimentName -ExitCode 2 `
            -Detail "Invalid experiment name (allowed: letters, digits, '_', '.', '-')." `
            -WindowMode $WindowMode
        return 2
    }
    if ($TargetOverride -and ($script:ValidTargets -notcontains $TargetOverride.ToUpper())) {
        Write-FailureBanner -ExperimentName $ExperimentName -ExitCode 2 `
            -Detail "Invalid -Target '$TargetOverride' (allowed: $($script:ValidTargets -join ', '))." `
            -WindowMode $WindowMode
        return 2
    }

    # Resolve the launch spec from the Python experiment registry (single source of truth).
    $specResult = Get-LaunchSpec -Context $Context -Name $ExperimentName
    if (-not $specResult.Ok) {
        $listResult = Get-RegistryList -Context $Context
        $known = ""
        if ($listResult.Ok) {
            $known = "Registered experiments:`n" + (
                ($listResult.Data.experiments | ForEach-Object {
                    "  {0,-28} target={1,-8} gpu={2}" -f $_.experiment, $_.target, $_.gpu
                }) -join "`n")
        } else {
            $known = "Registry error: $($listResult.Message)"
        }
        $message = $specResult.Message
        if ($message -notmatch "unknown experiment") { $message = "Launch spec lookup failed.`n$message" }
        Write-FailureBanner -ExperimentName $ExperimentName -ExitCode 2 `
            -Detail "$message`n`n$known" -WindowMode $WindowMode
        return 2
    }
    $spec = $specResult.Data
    $target = $Spec.target
    if ($TargetOverride) { $target = $TargetOverride.ToUpper() }

    # Resolve the concrete command without executing it.
    $remotePython = if ($env:AIC_AUTODL_PYTHON) { $env:AIC_AUTODL_PYTHON } else { "python" }
    try {
        $runnerArgs = Build-RunnerArgs -Experiment $Spec.experiment -Resume:$Resume -DryRun:$DryRun -ValidateOnly:$ValidateOnly
        if ($target -eq "AUTODL") {
            $resolvedCommand = "ssh <host> " + (Build-RemoteCommand -Spec $Spec -RunnerArgs $runnerArgs -RemotePython $remotePython)
        } else {
            $resolvedCommand = "& {0} {1} {2}" -f $Context.PythonPath, $Spec.stage_launcher, ($runnerArgs -join " ")
        }
    } catch {
        Write-FailureBanner -ExperimentName $ExperimentName -ExitCode 2 `
            -Detail "Failed to build the canonical command.`n$($_.Exception.Message)" -WindowMode $WindowMode
        return 2
    }

    # Dry-run: show everything, execute nothing.
    if ($DryRun) {
        $sshDisplay = "(not resolved in dry-run)"
        if ($target -eq "AUTODL") {
            $candidates = Resolve-SshHostCandidates
            $sshDisplay = "{0}   (candidate; not verified in dry-run)" -f ($candidates -join ", ")
        }
        Show-LaunchHeader -Spec $Spec -SshHost $sshDisplay -TargetOverride $TargetOverride `
            -RemotePython $remotePython -ResolvedCommand $resolvedCommand
        Write-Host " DRY RUN: launch spec resolved; the experiment was NOT executed."
        Write-SuccessBanner -ExperimentName $Spec.experiment -CloseDelaySec $Context.CloseDelaySec -WindowMode $WindowMode
        return 0
    }

    $sshHost = ""
    if ($target -eq "AUTODL") {
        try {
            $sshHost = Resolve-SshHost
        } catch {
            Write-FailureBanner -ExperimentName $Spec.experiment -ExitCode 3 `
                -Detail "SSH connection failed.`n$($_.Exception.Message)" -WindowMode $WindowMode
            return 3
        }
        if ($Spec.gpu -eq "REQUIRED") {
            if (-not (Test-RemoteGpuAvailable -SshHost $sshHost)) {
                Write-FailureBanner -ExperimentName $Spec.experiment -ExitCode 3 `
                    -Detail ("This experiment requires a GPU (GPU: REQUIRED), but the remote host " +
                            "'$sshHost' exposes no GPU (no-card mode?). Start a GPU instance and retry.") `
                    -WindowMode $WindowMode
                return 3
            }
        }
    }

    Show-LaunchHeader -Spec $Spec -SshHost $sshHost -TargetOverride $TargetOverride `
        -RemotePython $remotePython -ResolvedCommand $resolvedCommand

    $exitCode = 0
    if ($target -eq "AUTODL") {
        $remoteCommand = Build-RemoteCommand -Spec $Spec -RunnerArgs $runnerArgs -RemotePython $remotePython
        Write-Host "[launcher] streaming remote output (Ctrl+C interrupts; resume with -Resume) ..."
        $exitCode = Start-StreamingProcess -FilePath "ssh" `
            -ArgumentList @("-o", "ConnectTimeout=15", $sshHost, $remoteCommand) `
            -WorkingDirectory $Context.RepoRoot
    } else {
        Write-Host "[launcher] running locally (Ctrl+C interrupts; resume with -Resume) ..."
        $env:PYTHONPATH = Join-Path $Context.RepoRoot "src"
        $env:PYTHONUTF8 = "1"
        $env:PYTHONIOENCODING = "utf-8"
        $launcherPath = Join-Path $Context.RepoRoot ($Spec.stage_launcher -replace "/", "\")
        $exitCode = Start-StreamingProcess -FilePath $Context.PythonPath `
            -ArgumentList (@($launcherPath) + $runnerArgs) `
            -WorkingDirectory $Context.RepoRoot
    }

    if ($exitCode -eq 0) {
        Write-SuccessBanner -ExperimentName $Spec.experiment -CloseDelaySec $Context.CloseDelaySec -WindowMode $WindowMode
        return 0
    }
    Write-FailureBanner -ExperimentName $Spec.experiment -ExitCode $exitCode `
        -Detail "The runner process exited with a failure. Review the traceback, stderr and exit code above. Nothing was auto-cleaned." `
        -WindowMode $WindowMode
    return $exitCode
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if ($ImportOnly) { return }

$repoRoot = Get-LauncherRepoRoot
$context = @{
    RepoRoot       = $repoRoot
    PythonPath     = Resolve-PythonPath
    RegistryScript = Join-Path $repoRoot "scripts\experiments\registry.py"
    CloseDelaySec  = $CloseDelaySec
}

if ($Help) { Show-Help; exit 0 }

if (-not (Get-Command $context.PythonPath -ErrorAction SilentlyContinue)) {
    Write-Host "EXPERIMENT FAILED: python executable not found: $($context.PythonPath) (override with AIC_PYTHON)."
    exit 2
}

if ($List) {
    $listResult = Get-RegistryList -Context $context
    if (-not $listResult.Ok) {
        Write-Host "EXPERIMENT FAILED: registry list failed."
        Write-Host $listResult.Message
        exit 2
    }
    Write-Host "Registered experiments (source: Python experiment registry):"
    Write-Host ""
    Write-Host (" {0,-28} {1,-10} {2,-8} {3}" -f "EXPERIMENT", "STAGE", "TARGET", "GPU")
    foreach ($entry in $listResult.Data.experiments) {
        Write-Host (" {0,-28} {1,-10} {2,-8} {3}" -f $entry.experiment, $entry.stage, $entry.target, $entry.gpu)
    }
    exit 0
}

if ($Child -or $Inline) {
    $windowMode = [bool]$Child
    try {
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
        $OutputEncoding = [System.Text.Encoding]::UTF8
        chcp 65001 > $null
    } catch { }
    if ($windowMode) {
        if ($Experiment) {
            try { $Host.UI.RawUI.WindowTitle = "AIC Experiment: $Experiment" } catch { }
        }
    }
    $code = Invoke-ExperimentRun -Context $context -ExperimentName $Experiment `
        -Resume:$Resume -DryRun:$DryRun -ValidateOnly:$ValidateOnly `
        -TargetOverride $Target -WindowMode $windowMode
    exit $code
}

# Parent mode: spawn a dedicated PowerShell window for the run, then leave.
if (-not $Experiment) { Show-Help; exit 0 }
$scriptPath = $PSCommandPath
$psArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"{0}"' -f $scriptPath), "-Child")
$psArgs += @("-Experiment", ('"{0}"' -f $Experiment))
if ($Resume) { $psArgs += "-Resume" }
if ($ValidateOnly) { $psArgs += "-ValidateOnly" }
if ($DryRun) { $psArgs += "-DryRun" }
if ($Target) { $psArgs += @("-Target", ('"{0}"' -f $Target)) }
$psArgs += @("-CloseDelaySec", "$CloseDelaySec")
Start-Process -FilePath "powershell.exe" -ArgumentList $psArgs -WorkingDirectory $repoRoot
Write-Host "[launcher] spawned a dedicated PowerShell window for '$Experiment'."
Write-Host "[launcher] this console can be closed; the experiment window stays open."
exit 0
