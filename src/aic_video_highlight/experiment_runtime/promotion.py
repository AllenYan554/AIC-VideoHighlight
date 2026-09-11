"""Fail-closed preregistered experiment-promotion decisions."""

from __future__ import annotations

from typing import Any


def evaluate_smoke_promotion(validation: dict, *, identity_ok: bool) -> dict:
    """Authorize Formal only when every Smoke identity and mandatory gate passes."""
    engineering = validation.get("gates", {})
    scientific = validation.get("scientific_gates") or {}
    engineering_pass = bool(engineering) and all(value is True for value in engineering.values())
    scientific_pass = scientific.get("status", "PASS") == "PASS" and scientific.get(
        "all_pass"
    ) is True
    authorized = (
        identity_ok
        and validation.get("status") == "PASS"
        and engineering_pass
        and scientific_pass
    )
    return {
        "rule": "SMOKE_PASS_TO_FORMAL",
        "formal_authorized": authorized,
        "identity_ok": bool(identity_ok),
        "engineering_pass": engineering_pass,
        "scientific_pass": scientific_pass,
        "override_allowed": False,
        "decision": "AUTHORIZE_FORMAL" if authorized else "STOP_BEFORE_FORMAL",
    }


def build_promotion_marker(
    validation: dict,
    *,
    identity_ok: bool,
    method: str,
    smoke_manifest_identity: str,
    execution_head: str,
    protocol_byte_sha256: str,
    protocol_semantic_sha256: str,
    config_sha256: str,
    validation_sha256: str,
) -> dict[str, Any]:
    """Build the immutable machine-readable Smoke→Formal evidence marker."""
    decision = evaluate_smoke_promotion(validation, identity_ok=identity_ok)
    return {
        "schema_version": "aic.smoke-promotion/v1",
        "rule": "SMOKE_PASS_TO_FORMAL",
        "method": method,
        "smoke_manifest_identity": smoke_manifest_identity,
        "execution_head": execution_head,
        "protocol_byte_sha256": protocol_byte_sha256,
        "protocol_semantic_sha256": protocol_semantic_sha256,
        "config_sha256": config_sha256,
        "validation_sha256": validation_sha256,
        "all_pass": decision["formal_authorized"],
        "decision": decision["decision"],
        "reason": (
            "all preregistered engineering, mechanism, temporal and spatial gates passed"
            if decision["formal_authorized"]
            else "one or more mandatory Smoke gates or frozen identities failed"
        ),
        "override_allowed": False,
    }


def adjudicate_ts5_formal(validation: dict) -> dict:
    """Exact, non-rounded Amendment 4 terminal decision."""
    engineering = validation.get("gates", {})
    scientific = validation.get("scientific_gates") or {}
    deterministic = validation.get("deterministic_replay") is True
    passed = (
        validation.get("status") == "PASS"
        and bool(engineering)
        and all(value is True for value in engineering.values())
        and scientific.get("status") == "PASS"
        and scientific.get("all_pass") is True
        and deterministic
    )
    return {
        "status": "TS5_FINAL_FROZEN" if passed else "TS5_NOT_READY_FOR_FREEZE",
        "stage5_4_terminal": None if passed else "STAGE5_4_CLOSE_NO_AMENDMENT5",
        "all_pass": passed,
        "near_pass_allowed": False,
        "automatic_amendment5_allowed": False,
    }
