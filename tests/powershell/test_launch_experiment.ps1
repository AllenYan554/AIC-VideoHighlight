# Self-contained verification suite for scripts/experiments/launch_experiment.ps1.
# Run from the repository with:  powershell -NoProfile -ExecutionPolicy Bypass -File tests\powershell\test_launch_experiment.ps1
# Pure ASCII source (PS 5.1 reads BOM-less scripts as ANSI); Unicode is checked via code points.
#
# Note: B7/B8/B9 open real PowerShell windows for a few seconds and B8 leaves a
# visible failure window for ~18s; that is intentional lifecycle verification.

param()
$ErrorActionPreference = "Continue"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$launcher = Join-Path $repoRoot "scripts\experiments\launch_experiment.ps1"
$tmp = Join-Path $env:TEMP ("aic_launcher_tests_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $tmp | Out-Null
$script:Failures = 0

$BAR_CHAR = [string][char]0x2588            # U+2588 full block used by ProgressReporter
$SHUO_SHI = [string][char]0x7855 + [string][char]0x58EB   # the Chinese chars in the repo path

function Check {
    param([string] $Name, [bool] $Condition, [string] $Detail = "")
    if ($Condition) {
        Write-Host "[PASS] $Name"
    } else {
        $script:Failures += 1
        Write-Host "[FAIL] $Name"
        if ($Detail) { Write-Host "       $Detail" }
    }
}

function Invoke-LauncherCaptured {
    # Runs the launcher in a child powershell with raw byte redirection (UTF-8),
    # returns @{ ExitCode; Out; Err }.
    param([string] $ArgString, [string] $Tag)
    $outFile = Join-Path $tmp ($Tag + "_out.txt")
    $errFile = Join-Path $tmp ($Tag + "_err.txt")
    $launcherArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"{0}"' -f $launcher)) + ($ArgString -split ' ')
    $process = Start-Process -FilePath "powershell.exe" -ArgumentList $launcherArgs `
        -WorkingDirectory $repoRoot -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    $outText = ""
    $errText = ""
    if (Test-Path $outFile) { $outText = [System.IO.File]::ReadAllText($outFile, [System.Text.Encoding]::UTF8) }
    if (Test-Path $errFile) { $errText = [System.IO.File]::ReadAllText($errFile, [System.Text.Encoding]::UTF8) }
    return @{ ExitCode = $process.ExitCode; Out = $outText; Err = $errText }
}

# ---------------------------------------------------------------------------
# Unit tests: import launcher functions without running the entry point.
# ---------------------------------------------------------------------------
. $launcher -ImportOnly

$context = @{
    RepoRoot       = $repoRoot
    PythonPath     = Resolve-PythonPath
    RegistryScript = Join-Path $repoRoot "scripts\experiments\registry.py"
    CloseDelaySec  = 5
}

Check "U1 repo root resolution" ((Get-LauncherRepoRoot) -eq $repoRoot) (Get-LauncherRepoRoot)
Check "U1b repo root has Chinese and space path handled" ($repoRoot.Contains($SHUO_SHI) -and $repoRoot.Contains(" "))

$launcherText = [System.IO.File]::ReadAllText($launcher)
Check "U2 no hardcoded experiment names in launcher" (-not ($launcherText -match "stage5_\d|stage5_infra"))
Check "U3 no IP addresses or credential words in launcher" (-not ($launcherText -match '\d+\.\d+\.\d+\.\d+|password|passwd|api[_-]?key'))

Check "U4 experiment name format accepted" (Test-ExperimentNameFormat -Name "stage5_3_formal")
Check "U4b experiment name format rejects shell metacharacters" (-not (Test-ExperimentNameFormat -Name "stage5;rm -rf"))

$args1 = Build-RunnerArgs -Experiment "stage5_3_formal"
$args2 = Build-RunnerArgs -Experiment "stage5_3_formal" -Resume -DryRun -ValidateOnly
Check "U5 runner args base" (($args1 -join " ") -eq "--experiment stage5_3_formal") ($args1 -join " ")
Check "U5b runner args with resume/dry-run/validate-only" (($args2 -join " ") -eq "--experiment stage5_3_formal --resume --dry-run --validate-only") ($args2 -join " ")

$specResult = Get-LaunchSpec -Context $context -Name "stage5_3_formal"
$spec = $specResult.Data
$remoteProvenance = Get-LaunchProvenance -Target "AUTODL" -WindowMode $true
$remoteCmd = Build-RemoteCommand -Spec $spec -RunnerArgs (Build-RunnerArgs -Experiment "stage5_3_formal" -Resume) -RemotePython "python" -LaunchProvenance $remoteProvenance
$expectedRemote = "bash -lc 'cd /root/autodl-tmp/AIC-VideoHighlight-run && export PYTHONPATH=src PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 AIC_EXPERIMENT_LAUNCHER=powershell_v1 AIC_EXPERIMENT_LAUNCH_MODE=interactive_child AIC_EXPERIMENT_INTERACTIVE_CHILD=true AIC_EXPERIMENT_LAUNCH_TARGET=AUTODL && python scripts/experiments/stage5/run.py --experiment stage5_3_formal --resume'"
Check "U6 remote command construction exact" ($remoteCmd -eq $expectedRemote) $remoteCmd
Check "U6b remote command single-quoted for ssh" ($remoteCmd.StartsWith("bash -lc '") -and $remoteCmd.EndsWith("'"))
Check "U6c remote command records launcher provenance" ($remoteCmd -match "AIC_EXPERIMENT_LAUNCHER=powershell_v1" -and $remoteCmd -match "AIC_EXPERIMENT_INTERACTIVE_CHILD=true")

$unsafeThrew = $false
try { Build-RunnerArgs -Experiment 'stage5";rm' | Out-Null } catch { $unsafeThrew = $true }
Check "U7 unsafe token rejected" $unsafeThrew

Check "U8 default ssh candidates" (((Resolve-SshHostCandidates) -join ",") -eq "autodl,autodl-stage1")
$env:AIC_AUTODL_SSH_HOST = "my-autodl-alias"
Check "U8b ssh host env override" (((Resolve-SshHostCandidates) -join ",") -eq "my-autodl-alias")
Remove-Item Env:AIC_AUTODL_SSH_HOST -ErrorAction SilentlyContinue

$originalConnectivity = ${function:Test-SshConnectivity}
try {
    ${function:Test-SshConnectivity} = { param([string] $SshHost) $false }
    $noHostThrew = $false
    $noHostMessage = ""
    try { Resolve-SshHost | Out-Null } catch { $noHostThrew = $true; $noHostMessage = $_.Exception.Message }
    Check "U9 unreachable ssh hosts raise clear error" ($noHostThrew -and $noHostMessage -match "AIC_AUTODL_SSH_HOST" -and $noHostMessage -match "instance is running") $noHostMessage
    ${function:Test-SshConnectivity} = { param([string] $SshHost) $true }
    Check "U10 first reachable candidate selected" ((Resolve-SshHost) -eq "autodl")
} finally {
    ${function:Test-SshConnectivity} = $originalConnectivity
}

$listData = (Get-RegistryList -Context $context).Data
Check "U11 registry list has eight experiments" ($listData.experiments.Count -eq 8)
$gpuByExperiment = @{}
foreach ($entry in $listData.experiments) { $gpuByExperiment[$entry.experiment] = $entry }
Check "U11b stage5_3_formal targets AUTODL" ($gpuByExperiment["stage5_3_formal"].target -eq "AUTODL")
Check "U11c smoke/tiny_fake target WINDOWS" ($gpuByExperiment["stage5_3_smoke"].target -eq "WINDOWS" -and $gpuByExperiment["stage5_infra_tiny_fake"].target -eq "WINDOWS")
Check "U11d launch spec exposes gpu requirement" ((Get-LaunchSpec -Context $context -Name "stage5_3_formal").Data.gpu -eq "NONE")
Check "U11e stage5_4_formal targets AUTODL with no GPU" ($gpuByExperiment["stage5_4_formal"].target -eq "AUTODL" -and $gpuByExperiment["stage5_4_formal"].gpu -eq "NONE")
Check "U11f stage5_4_amendment_smoke targets AUTODL with no GPU" ($gpuByExperiment["stage5_4_amendment_smoke"].target -eq "AUTODL" -and $gpuByExperiment["stage5_4_amendment_smoke"].gpu -eq "NONE")
Check "U11g stage5_4_amendment2_smoke targets AUTODL with no GPU" ($gpuByExperiment["stage5_4_amendment2_smoke"].target -eq "AUTODL" -and $gpuByExperiment["stage5_4_amendment2_smoke"].gpu -eq "NONE")

# ---------------------------------------------------------------------------
# Behavioral tests (real child processes).
# ---------------------------------------------------------------------------
$helpResult = Invoke-LauncherCaptured -ArgString "-Help" -Tag "help"
Check "B1 -Help exits 0" ($helpResult.ExitCode -eq 0) ("exit=" + $helpResult.ExitCode)
Check "B1b -Help mentions usage" ($helpResult.Out -match "Usage:")

$listResult = Invoke-LauncherCaptured -ArgString "-List" -Tag "list"
Check "B2 -List exits 0" ($listResult.ExitCode -eq 0) ("exit=" + $listResult.ExitCode)
Check "B2b -List shows all registered experiments" ($listResult.Out -match "stage5_3_formal" -and $listResult.Out -match "stage5_infra_tiny_fake")
Check "B2c -List shows stage5_4_formal" ($listResult.Out -match "stage5_4_formal")

$unknown = Invoke-LauncherCaptured -ArgString "stage5_9_bogus -Inline" -Tag "unknown"
Check "B3 unknown experiment exits non-zero" ($unknown.ExitCode -ne 0) ("exit=" + $unknown.ExitCode)
Check "B3b unknown experiment lists valid names" ($unknown.Out -match "unknown experiment" -and $unknown.Out -match "stage5_3_formal")
Check "B3c no python traceback shown" (-not ($unknown.Out + $unknown.Err -match "Traceback"))

$dry = Invoke-LauncherCaptured -ArgString "stage5_3_formal -DryRun -Inline" -Tag "dry_autodl"
Check "B4 dry-run exits 0 without executing" ($dry.ExitCode -eq 0) ("exit=" + $dry.ExitCode)
Check "B4b dry-run shows AUTODL target and GPU requirement" ($dry.Out -match "Execution  : AUTODL" -and $dry.Out -match "GPU        : NONE")
Check "B4c dry-run shows remote command but does not ssh" ($dry.Out -match [regex]::Escape("bash -lc 'cd /root/autodl-tmp/AIC-VideoHighlight-run") -and $dry.Out -match "DRY RUN")

$dry54 = Invoke-LauncherCaptured -ArgString "stage5_4_formal -DryRun -Inline" -Tag "dry_stage54_autodl"
Check "B4d stage5_4_formal dry-run exits 0" ($dry54.ExitCode -eq 0) ("exit=" + $dry54.ExitCode)
Check "B4e stage5_4_formal dry-run is AUTODL and GPU NONE" ($dry54.Out -match "Execution  : AUTODL" -and $dry54.Out -match "GPU        : NONE" -and $dry54.Out -match "DRY RUN")
$dry54Amendment = Invoke-LauncherCaptured -ArgString "stage5_4_amendment_smoke -DryRun -Inline" -Tag "dry_stage54_amendment_autodl"
Check "B4f stage5_4_amendment_smoke dry-run exits 0" ($dry54Amendment.ExitCode -eq 0) ("exit=" + $dry54Amendment.ExitCode)
Check "B4g stage5_4_amendment_smoke dry-run is AUTODL and GPU NONE" ($dry54Amendment.Out -match "Execution  : AUTODL" -and $dry54Amendment.Out -match "GPU        : NONE" -and $dry54Amendment.Out -match "DRY RUN")
$dry54Amendment2 = Invoke-LauncherCaptured -ArgString "stage5_4_amendment2_smoke -DryRun -Inline" -Tag "dry_stage54_amendment2_autodl"
Check "B4h stage5_4_amendment2_smoke dry-run exits 0" ($dry54Amendment2.ExitCode -eq 0) ("exit=" + $dry54Amendment2.ExitCode)
Check "B4i stage5_4_amendment2_smoke dry-run is AUTODL and GPU NONE" ($dry54Amendment2.Out -match "Execution  : AUTODL" -and $dry54Amendment2.Out -match "GPU        : NONE" -and $dry54Amendment2.Out -match "DRY RUN")

$dryLocal = Invoke-LauncherCaptured -ArgString "stage5_infra_tiny_fake -DryRun -Inline" -Tag "dry_windows"
Check "B5 dry-run WINDOWS target shows local command" ($dryLocal.ExitCode -eq 0 -and $dryLocal.Out -match [regex]::Escape("run.py --experiment stage5_infra_tiny_fake --dry-run"))

$badTarget = Invoke-LauncherCaptured -ArgString "stage5_3_formal -Target LINUX -Inline" -Tag "bad_target"
Check "B6 invalid -Target rejected" ($badTarget.ExitCode -eq 2 -and $badTarget.Out -match "Invalid -Target")

$stderrProbeScript = Join-Path $tmp "streaming_stderr_probe.ps1"
$stderrProbeOut = Join-Path $tmp "streaming_stderr_out.txt"
$stderrProbeErr = Join-Path $tmp "streaming_stderr_err.txt"
$escapedLauncher = $launcher.Replace("'", "''")
$escapedRepoRoot = $repoRoot.Replace("'", "''")
$stderrProbeSource = @"
. '$escapedLauncher' -ImportOnly
`$code = Start-StreamingProcess -FilePath "powershell.exe" ``
    -ArgumentList @("-NoProfile", "-Command", "[Console]::Error.WriteLine('AIC_STDERR_VISIBLE')") ``
    -WorkingDirectory '$escapedRepoRoot'
exit `$code
"@
[System.IO.File]::WriteAllText($stderrProbeScript, $stderrProbeSource, [System.Text.UTF8Encoding]::new($true))
$stderrProcess = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"{0}"' -f $stderrProbeScript)) `
    -WorkingDirectory $repoRoot -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $stderrProbeOut -RedirectStandardError $stderrProbeErr
$stderrCode = $stderrProcess.ExitCode
$stderrText = if (Test-Path $stderrProbeErr) { Get-Content $stderrProbeErr -Raw -Encoding UTF8 } else { "" }
Check "B6b native stderr remains visible through streaming process" (
    $stderrCode -eq 0 -and $stderrText -match "AIC_STDERR_VISIBLE"
) ("exit=" + $stderrCode + " stderr=" + $stderrText)

# B7: success path auto-closes the dedicated child window (dry-run, 1s delay).
$successChild = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"{0}"' -f $launcher), "-Child", "-Experiment", "stage5_infra_tiny_fake", "-DryRun", "-CloseDelaySec", "1") `
    -WorkingDirectory $repoRoot -PassThru
$exitedInTime = $successChild.WaitForExit(45000)
Check "B7 success child window auto-closes" ($exitedInTime -and $successChild.ExitCode -eq 0) ("exited=$exitedInTime code=" + $successChild.ExitCode)

# B8: failure path keeps the dedicated child window open (SSH failure blocks on Enter).
$env:AIC_AUTODL_SSH_HOST = "aic_invalid_host_for_launcher_test"
try {
    $failChild = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"{0}"' -f $launcher), "-Child", "-Experiment", "stage5_3_formal", "-Target", "AUTODL") `
        -WorkingDirectory $repoRoot -PassThru
    Start-Sleep -Seconds 18
    $stillOpen = -not $failChild.HasExited
    Check "B8 failure child window stays open (no auto-close)" $stillOpen ("hasExited=" + $failChild.HasExited)
    if ($stillOpen) {
        & taskkill /PID $failChild.Id /T /F | Out-Null
    } else {
        Check "B8b failure exit code visible" ($failChild.ExitCode -eq 3) ("code=" + $failChild.ExitCode)
    }
} finally {
    Remove-Item Env:AIC_AUTODL_SSH_HOST -ErrorAction SilentlyContinue
}

# B8e: scientific validation exit code 2 follows the same retained-window path.
$exitTwoChild = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"{0}"' -f $launcher), "-Child", "-Experiment", "stage5_4_formal", "-Target", "INVALID") `
    -WorkingDirectory $repoRoot -PassThru
Start-Sleep -Seconds 3
$exitTwoStillOpen = -not $exitTwoChild.HasExited
Check "B8e exit code 2 child window stays open" $exitTwoStillOpen ("hasExited=" + $exitTwoChild.HasExited)
if ($exitTwoStillOpen) { & taskkill /PID $exitTwoChild.Id /T /F | Out-Null }

# B8c: the same failure inline captures the FAILED banner and exit code 3.
$env:AIC_AUTODL_SSH_HOST = "aic_invalid_host_for_launcher_test"
try {
    $failInline = Invoke-LauncherCaptured -ArgString "stage5_3_formal -Target AUTODL -Inline" -Tag "fail_inline"
    Check "B8c inline ssh failure shows EXPERIMENT FAILED" ($failInline.Out -match "EXPERIMENT FAILED" -and $failInline.ExitCode -eq 3) ("exit=" + $failInline.ExitCode)
    Check "B8d ssh failure explains remedies" ($failInline.Out -match "AIC_AUTODL_SSH_HOST")
} finally {
    Remove-Item Env:AIC_AUTODL_SSH_HOST -ErrorAction SilentlyContinue
}

# B9: parent mode spawns a dedicated child window and exits immediately.
$parent = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"{0}"' -f $launcher), "-Experiment", "stage5_infra_tiny_fake", "-DryRun") `
    -WorkingDirectory $repoRoot -PassThru
$childSeen = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 400
    $childSeen = [bool](Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
        Where-Object { $_.CommandLine -match "launch_experiment\.ps1" -and $_.CommandLine -match " -Child" } |
        Select-Object -First 1)
    if ($childSeen) { break }
}
$parentDone = $parent.WaitForExit(30000)
Check "B9 parent exits after spawning child window" ($parentDone -and $parent.ExitCode -eq 0) ("done=$parentDone code=" + $parent.ExitCode)
Check "B9b spawned child carries -Child flag (recursion protected)" $childSeen

# B10: real local WINDOWS run with resume, UTF-8 progress bar, Chinese paths.
# Isolate runtime state so an old strict run identity from another HEAD cannot
# affect this launcher behavior test.
$runtimeRoot = Join-Path $tmp "runtime"
$testEnvironment = Join-Path $tmp "environment.json"
$environmentPayload = @{
    name = "launcher-test"
    repo = $repoRoot
    datasets = (Join-Path $runtimeRoot "datasets")
    models = (Join-Path $runtimeRoot "models")
    hf_cache = (Join-Path $runtimeRoot "hf-cache")
    outputs = (Join-Path $runtimeRoot "outputs")
    logs = (Join-Path $runtimeRoot "logs")
    cache = (Join-Path $runtimeRoot "cache")
    tmp = (Join-Path $runtimeRoot "tmp")
    archive = (Join-Path $runtimeRoot "archive")
} | ConvertTo-Json
[System.IO.File]::WriteAllText($testEnvironment, $environmentPayload, [System.Text.UTF8Encoding]::new($false))
$env:AIC_EXPERIMENT_ENVIRONMENT = $testEnvironment
try {
    $real = Invoke-LauncherCaptured -ArgString "stage5_infra_tiny_fake -Inline -Resume" -Tag "real_local"
} finally {
    Remove-Item Env:AIC_EXPERIMENT_ENVIRONMENT -ErrorAction SilentlyContinue
}
Check "B10 local resume run exits 0" ($real.ExitCode -eq 0) ("exit=" + $real.ExitCode)
$envJson = Get-Content $testEnvironment -Raw -Encoding UTF8 | ConvertFrom-Json
$statusPath = Join-Path $envJson.outputs "stage5\stage5_infra_tiny_fake\status.json"
$manifestPath = Join-Path $envJson.outputs "stage5\stage5_infra_tiny_fake\run_manifest.json"
Check "B10b runtime status COMPLETED" ((Test-Path $statusPath) -and ((Get-Content $statusPath -Raw -Encoding UTF8) -match "COMPLETED"))
Check "B10c progress bar (U+2588) streamed" ($real.Out.Contains($BAR_CHAR))
Check "B10d no mojibake in raw report path" ($real.Out -match "experiment_raw_report")
Check "B10e resume flag forwarded to runner" ($real.Out -match "--resume")

$manifestRaw = ""
if (Test-Path $manifestPath) { $manifestRaw = Get-Content $manifestPath -Raw -Encoding UTF8 }
Check "B11 run_manifest resume_mode true" ($manifestRaw -match '"resume_mode":\s*true')
Check "B11b run_manifest records PowerShell inline provenance" (
    $manifestRaw -match '"launch_source":\s*"powershell_v1"' -and
    $manifestRaw -match '"launch_mode":\s*"inline"' -and
    $manifestRaw -match '"interactive_child":\s*false' -and
    $manifestRaw -match '"target":\s*"WINDOWS"'
)

Write-Host ""
if ($script:Failures -eq 0) {
    Write-Host "ALL LAUNCHER TESTS PASSED"
    exit 0
} else {
    Write-Host ("LAUNCHER TESTS FAILED: {0}" -f $script:Failures)
    exit 1
}
