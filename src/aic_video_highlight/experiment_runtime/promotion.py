"""Fail-closed preregistered experiment-promotion decisions."""

from __future__ import annotations


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
