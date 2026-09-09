# Unified Experiment Launcher (Windows PowerShell)

One command to launch any registered experiment. The launcher spawns a dedicated
PowerShell window, resolves the experiment from the Python experiment registry
(single source of truth), streams the canonical runner's output live, closes the
window automatically on success, and keeps it open on failure.

## Layout

Stage-specific experiment entrypoints live in `scripts/experiments/<stage>/`
(one subdirectory per research stage; each has a README describing its scripts).
Generic utilities (`setup/`, `tools/`, `analysis/`, `validation/` and the
non-stage helpers in `scripts/`) stay outside this directory. See
`docs/infrastructure/DIRECTORY_STANDARD.md` for the repository-wide standard.

## Launch

```powershell
.\scripts\experiments\launch_experiment.ps1 <experiment>
```

Examples:

```powershell
.\scripts\experiments\launch_experiment.ps1 -List
.\scripts\experiments\launch_experiment.ps1 stage5_infra_tiny_fake
.\scripts\experiments\launch_experiment.ps1 stage5_3_formal -Resume
```

Options: `-Resume`, `-ValidateOnly`, `-DryRun`, `-Target WINDOWS|AUTODL`,
`-List`, `-Help`, `-CloseDelaySec <n>` (default 5), `-Inline` (run in the
current console instead of a new window).

## Interactive Experiment Execution Policy

All user-visible Smoke, Formal, Ablation and Validation runs started from a
Windows agent environment MUST use:

```powershell
.\scripts\experiments\launch_experiment.ps1 <experiment>
```

The default (without `-Inline`) is the interactive contract: a dedicated child
PowerShell window, live stdout/stderr, and lifecycle handling based on the
runner's final process exit code. Agents must not replace it with direct `ssh`
or `python scripts/.../run.py` commands merely to save a step.

Direct runner execution remains supported only for CI, unit tests,
`--validate-only`, and explicit developer debugging. Such runs are recorded in
`run_manifest.json` as `NON_INTERACTIVE_DIRECT_RUN`. PowerShell-launched runs
record `launch_source`, `launch_mode`, `interactive_child`, and `target` as
operational provenance. These fields are intentionally outside strict run
identity and scientific semantic hashes.

## Dry run

```powershell
.\scripts\experiments\launch_experiment.ps1 <experiment> -DryRun
```

Prints the resolved execution target, GPU requirement, SSH host candidate,
remote repo, config and the exact canonical command. Nothing is executed.

## Windows vs AutoDL targets

The registry entry in `scripts/experiments/<stage>/run.py` (`LAUNCH`) decides
the execution target:

- `WINDOWS`: runs in the local repository with `PYTHONPATH=src`,
  `PYTHONUTF8=1` and the environment from `configs/environments/windows_local.json`.
- `AUTODL`: connects over SSH, `cd`s into the remote repo, exports
  `PYTHONPATH=src PYTHONUTF8=1`, and runs the same canonical launcher there
  (which auto-selects `configs/environments/autodl.json`).

Override with `-Target` only for deliberate cross-target checks.

## GPU requirement

The header always shows the registry's GPU requirement (`NONE`, `OPTIONAL`,
`REQUIRED`). If an experiment is `REQUIRED` and the remote host exposes no GPU
(for example a no-card AutoDL instance, where `nvidia-smi` is absent), the
launcher stops with a clear error instead of silently running on CPU.

## SSH alias configuration

The launcher never stores hosts, IPs or credentials in the repository. It
resolves the AutoDL host from, in order:

1. the `AIC_AUTODL_SSH_HOST` environment variable (e.g. `autodl-stage1`);
2. the SSH config aliases `autodl`, then `autodl-stage1`.

Configure the alias yourself in `~/.ssh/config`; the launcher only tests
reachability (key-based, `BatchMode`) and reports a clear error if the instance
is off or no alias works. Remote Python defaults to `python` and can be
overridden with `AIC_AUTODL_PYTHON`. The launcher never starts/stops AutoDL
instances and never modifies `~/.ssh/config`.

## Success / failure window lifecycle

- **Success** (process exit code 0): the window prints
  `Experiment completed successfully`, waits `-CloseDelaySec` seconds (default
  5) and closes itself. A progress bar reaching 100% is NOT success — only the
  full process lifecycle (exit code) decides.
- **Failure** (non-zero exit code, SSH failure, GPU gate, unknown experiment):
  the window prints `EXPERIMENT FAILED`, the exit code and all stderr /
  traceback above, and waits for `Press Enter to close...`. It never
  auto-closes.

Scientific validation failures use this same rule. For example, exit code 2
after a complete Formal run remains a real non-zero result: it is displayed as
failed/not passed and the child window stays open. Reaching 100% frames never
overrides the final exit status.

## Resume

Resume is owned entirely by the experiment runtime (strict run-manifest
identity checks). Pass `-Resume` and the launcher forwards `--resume` to the
canonical runner. The launcher never resets experiments, deletes shards or
bypasses `RunIdentityMismatch`.

## AutoDL prerequisites

1. The AutoDL instance must already be running (no-card mode is enough for
   CPU experiments and all smoke/validate/dry-run work; GPU experiments need a
   GPU instance).
2. The remote repo `/root/autodl-tmp/AIC-VideoHighlight-run` is pulled to the
   same commit you are launching.
3. SSH alias works non-interactively (`ssh <alias> echo ok`).

## Adding a new experiment

Register it in the canonical stage launcher
(`scripts/experiments/<stage>/run.py`): add entries to `RUNNERS`, `CONFIGS` and
`LAUNCH`. The PowerShell launcher and `scripts/experiments/registry.py` pick it
up automatically — no launcher edits, ever.
