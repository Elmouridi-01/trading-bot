# core/startup_validator.py
"""
core/startup_validator.py

Startup validation — runs before any trading begins.

Validates that the deployed model's training parameters match the
current settings, and that the feature schema is compatible with
the live feature pipeline. If any check fails, raises RuntimeError
and the engine must not start.

This prevents the silent misconfiguration described in Phase 2:
  W-1  : Model trained with ordinal symbol encoding → AI dark
  W-2  : TB parameters disagree between model and settings → wrong R:R
  MC-1 : No validation existed to catch these at startup

Called from core/engine.py during the startup sequence, before
data collectors or strategies are initialised.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Tolerance for floating-point parameter comparison.
# JSON round-trip can introduce rounding at the last decimal place.
_PARAM_TOLERANCE: float = 0.001


def validate_model_parameter_alignment(
    metadata_path: str | Path = "ai/models/metadata.json",
) -> None:
    """
    Validates the deployed model against current settings.

    Checks:
      1. metadata.json tp_pct  matches settings.TB_TP_PCT   (±tolerance)
      2. metadata.json sl_pct  matches settings.TB_SL_PCT   (±tolerance)
      3. Feature schema does not contain 'symbol_id' (ordinal encoding)
      4. Feature schema contains One-Hot symbol columns

    Raises:
        RuntimeError if any check fails. The caller must not start
        the trading engine in this state.

    Does nothing (passes silently) if:
        - metadata.json does not exist (first run, no model deployed yet).
          The engine will train before any trading begins.
    """
    from config.settings import settings

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        log.info("startup_validator.no_model_found", extra={
            "path": str(metadata_path),
            "note": "No deployed model — will train before trading begins.",
        })
        print("[Startup] No model found — will train before trading begins.")
        return

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"[STARTUP BLOCKED] Cannot read {metadata_path}: {e}\n"
            f"The metadata file may be corrupt. "
            f"Delete it and retrain the model."
        ) from e

    errors: list[str] = []

    # ── Check 1: Take Profit parameter ───────────────────────────────────────
    model_tp = float(meta.get("tp_pct", -1.0))
    live_tp  = settings.TB_TP_PCT

    if model_tp < 0:
        errors.append(
            "metadata.json missing 'tp_pct' field. "
            "The model was trained with an old pipeline that did not "
            "record TB parameters. Retrain the model."
        )
    elif abs(model_tp - live_tp) > _PARAM_TOLERANCE:
        errors.append(
            f"Take-profit mismatch: "
            f"model trained with tp_pct={model_tp:.4f} "
            f"but settings.TB_TP_PCT={live_tp:.4f}. "
            f"Either update settings.TB_TP_PCT to {model_tp:.4f} "
            f"(to match the deployed model) or retrain the model "
            f"with the current settings value."
        )

    # ── Check 2: Stop Loss parameter ─────────────────────────────────────────
    model_sl = float(meta.get("sl_pct", -1.0))
    live_sl  = settings.TB_SL_PCT

    if model_sl < 0:
        errors.append(
            "metadata.json missing 'sl_pct' field. "
            "Retrain the model with the current pipeline."
        )
    elif abs(model_sl - live_sl) > _PARAM_TOLERANCE:
        errors.append(
            f"Stop-loss mismatch: "
            f"model trained with sl_pct={model_sl:.4f} "
            f"but settings.TB_SL_PCT={live_sl:.4f}. "
            f"Either update settings.TB_SL_PCT to {model_sl:.4f} "
            f"or retrain the model."
        )

    # ── Check 3: No ordinal symbol encoding ───────────────────────────────────
    feature_cols = meta.get("feature_cols", [])

    if "symbol_id" in feature_cols:
        errors.append(
            "Deployed model uses ordinal symbol encoding ('symbol_id'). "
            "The live feature builder (build_features()) now produces "
            "One-Hot encoding: 'symbol_is_btc', 'symbol_is_eth', "
            "'symbol_is_sol'. "
            "Every predict_signal() call returns 0.0 until this is fixed. "
            "Action: run python scripts/retrain_now.py"
        )

    # ── Check 4: One-Hot columns present ──────────────────────────────────────
    if "symbol_id" not in feature_cols and feature_cols:
        required_onehot = {"symbol_is_btc", "symbol_is_eth", "symbol_is_sol"}
        missing_onehot  = required_onehot - set(feature_cols)
        if missing_onehot:
            errors.append(
                f"Feature schema missing One-Hot symbol columns: "
                f"{missing_onehot}. "
                f"This indicates an inconsistent model state. "
                f"Retrain the model."
            )

    # ── Report ────────────────────────────────────────────────────────────────
    if errors:
        model_version = meta.get("version", "unknown")
        error_block   = "\n".join(f"  • {e}" for e in errors)
        raise RuntimeError(
            f"\n\n{'='*70}\n"
            f"[STARTUP BLOCKED] Model parameter validation failed\n"
            f"Model version: {model_version}\n"
            f"{'='*70}\n\n"
            f"{error_block}\n\n"
            f"The engine will not start in this state.\n"
            f"Resolve all issues above before restarting.\n"
            f"{'='*70}\n"
        )

    log.info("startup_validator.passed", extra={
        "model_version": meta.get("version", "unknown"),
        "tp_pct":        model_tp,
        "sl_pct":        model_sl,
        "feature_count": len(feature_cols),
        "encoding":      "one_hot",
    })

    print(
        f"[Startup] ✅ Model validation passed | "
        f"version={meta.get('version', 'unknown')} | "
        f"tp={model_tp:.4f} sl={model_sl:.4f} | "
        f"features={len(feature_cols)} (One-Hot)"
    )