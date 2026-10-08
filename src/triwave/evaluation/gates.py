"""Persisted development gates; passing these is never a SOTA certificate."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from triwave.data.manifest import canonical_patient

from .data import json_digest


def write_report(path: str | Path | None, report: dict) -> dict:
    report["evidence_digest"] = json_digest({k: v for k, v in report.items()
                                              if k != "evidence_digest"})
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, target)
    return report


class GateLedger:
    """A versioned local audit ledger, not a cryptographically trusted attestation."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self.state = {"format_version": 1, "clinical_validity": "not_established",
                      "gates": {}, "history": [], "g3_operations": []}
        if self.path is not None and self.path.exists():
            value = json.loads(self.path.read_text())
            if value.get("format_version") != 1:
                raise ValueError("unsupported evaluation gate ledger version")
            expected = json_digest({k: v for k, v in value.items() if k != "evidence_digest"})
            if value.get("evidence_digest") != expected:
                raise ValueError("evaluation gate ledger digest is stale")
            self.state = value

    def _save(self) -> None:
        if self.path is not None:
            write_report(self.path, self.state)

    def record(self, report: dict) -> None:
        stage = report["stage"]
        if stage not in {"G0", "G1", "G2"}:
            raise ValueError("only development reports can open evaluation gates")
        expected = json_digest({k: v for k, v in report.items() if k != "evidence_digest"})
        if report.get("evidence_digest") != expected:
            raise ValueError("evaluation evidence digest is missing or stale")
        if stage == "G1":
            self.require("G0", report["data_signature"])
        if stage == "G2":
            self.require("G1", report["data_signature"])
        # Updating a gate invalidates later gates, including a formerly passing
        # configuration. Failed later runs are preserved rather than ignored.
        for later in ("G0", "G1", "G2")[("G0", "G1", "G2").index(stage) + 1:]:
            self.state["gates"].pop(later, None)
        self.state["gates"][stage] = {"passed": bool(report.get("gate_passed", False)),
                                      "data_signature": report["data_signature"],
                                      "candidate": report.get("candidate"),
                                      "evidence_digest": report["evidence_digest"]}
        self.state["history"].append({"recorded_at": datetime.now(timezone.utc).isoformat(),
                                      "report": report})
        self._save()

    def record_failure(self, stage: str, error: Exception | str, *, seed: int | None = None) -> None:
        if stage in {"G0", "G1", "G2"}:
            for name in ("G0", "G1", "G2")[("G0", "G1", "G2").index(stage):]:
                self.state["gates"].pop(name, None)
        self.state["history"].append({"recorded_at": datetime.now(timezone.utc).isoformat(),
                                      "stage": stage, "status": "failed", "seed": seed,
                                      "error": str(error)})
        self._save()

    def require(self, stage: str, data_signature: str | None = None) -> dict:
        gate = self.state["gates"].get(stage, {})
        if not gate.get("passed", False):
            raise ValueError(f"a passing recorded {stage} gate is required")
        if data_signature is not None and gate.get("data_signature") != data_signature:
            raise ValueError("evaluation data differ from the recorded gate evidence")
        return gate

    def require_g3(self) -> dict:
        gate = self.require("G2")
        self.require("G1", gate["data_signature"])
        self.require("G0", gate["data_signature"])
        return gate


def record_g2(ledger: GateLedger, evidence: list[dict | str | Path], *,
              candidate: str = "spikeweave", output: str | Path | None = None) -> dict:
    """Confirm three paired development runs, preserving all imported evidence.

Each entry is a complete G1-format run report. Identical input/condition and
training budgets, distinct seeds, valid quality screens, and the same required
baseline set are mandatory. The gate tests mean improvement plus 2/3 positive
paired differences against the strongest baseline by mean development AP.
"""
    runs = [json.loads(Path(item).read_text()) if isinstance(item, (str, Path)) else item
            for item in evidence]
    if len(runs) != 3:
        raise ValueError("G2 requires exactly three predeclared paired seed reports")
    for report in runs:
        if report.get("stage") != "G1" or report.get("candidate") != candidate:
            raise ValueError("G2 evidence must be matching G1-format candidate reports")
        if report.get("evidence_digest") != json_digest(
                {k: v for k, v in report.items() if k != "evidence_digest"}):
            raise ValueError("imported run evidence digest is missing or stale")
    signature = runs[0]["data_signature"]
    ledger.require("G1", signature)
    if len({r["seed"] for r in runs}) != 3:
        raise ValueError("G2 seed reports must contain three distinct seeds")
    if any(r["data_signature"] != signature or r["budget_digest"] != runs[0]["budget_digest"]
           or r["required_baselines"] != runs[0]["required_baselines"] for r in runs):
        raise ValueError("G2 requires identical data, conditions, baseline set and run budgets")
    names = runs[0]["required_baselines"]
    if not names:
        raise ValueError("G2 cannot confirm a candidate without strong baselines")
    for name in [candidate, *names]:
        imports = [r.get("imports", {}).get(name, {}) for r in runs]
        fixed_fields = ("algorithm", "code_commit", "implementation_sha256", "generator_recipe_digest",
                        "generator_condition_budget", "budget", "source_train_digest", "condition_origin")
        for index, item in enumerate(imports):
            missing = [key for key in fixed_fields if item.get(key) is None]
            if missing:
                raise ValueError(f"G2 method {name!r}, seed {runs[index]['seed']} lacks fixed evidence: "
                                 + ", ".join(missing))
            for key in ("implementation_sha256", "generator_recipe_digest", "source_train_digest"):
                digest = item[key]
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)):
                    raise ValueError(f"G2 method {name!r} needs a valid {key}")
            if (not isinstance(item["algorithm"], str) or not item["algorithm"].strip()
                    or not isinstance(item["code_commit"], str) or len(item["code_commit"]) not in (40, 64)
                    or any(c not in "0123456789abcdef" for c in item["code_commit"])):
                raise ValueError(f"G2 method {name!r} needs explicit algorithm and stable code_commit")
            if any(not isinstance(item[key], dict) or not item[key]
                   for key in ("budget", "generator_condition_budget")):
                raise ValueError(f"G2 method {name!r} needs nonempty generator/condition budgets")
            if item["condition_origin"] != "source_request":
                raise ValueError(f"G2 method {name!r} is not strict source-request generation")
        for key in fixed_fields:
            if any(item[key] != imports[0][key] for item in imports[1:]):
                raise ValueError(f"G2 method {name!r} differs across training seeds in fixed {key}")
        training_seeds = [item.get("generator_training_seed") for item in imports]
        checkpoints = [item.get("generator_checkpoint_digest") for item in imports]
        if (any(type(seed) is not int for seed in training_seeds)
                or training_seeds != [r["seed"] for r in runs]
                or any(not isinstance(digest, str) or len(digest) != 64
                       or any(c not in "0123456789abcdef" for c in digest) for digest in checkpoints)
                or len(set(checkpoints)) != 3):
            raise ValueError("G2 needs three distinct generator checkpoints with paired audited training seeds; "
                             "downstream/request seeds alone are insufficient")
    method_names = ["real_only", "repeat", "normalaug", *names]
    if any(not r.get("quality_minimum_passed") or r.get("status") != "complete"
           or any(name not in r["downstream"] for name in [candidate, *method_names])
           for r in runs):
        raise ValueError("G2 requires completed fair baseline runs and passing quality checks")
    means = {}
    for name in [candidate, *method_names]:
        values = [r["downstream"][name]["macro_patient_ap"] for r in runs]
        if any(value is None or not np.isfinite(value) for value in values):
            raise ValueError("G2 AP must be computable in all required runs")
        means[name] = float(np.mean(values))
    strongest = max(method_names, key=lambda name: means[name])
    differences = [float(r["downstream"][candidate]["macro_patient_ap"]
                         - r["downstream"][strongest]["macro_patient_ap"]) for r in runs]
    passed = means[candidate] > means[strongest] and sum(d > 0 for d in differences) >= 2
    report = {"stage": "G2", "status": "complete", "candidate": candidate,
              "data_signature": signature, "budget_digest": runs[0]["budget_digest"],
              "seeds": [r["seed"] for r in runs], "mean_macro_patient_ap": means,
              "strongest_baseline": strongest, "paired_differences": differences,
              "gate_passed": bool(passed), "test_used": False,
              "conclusion": "development_confirmation_only_not_SOTA_or_clinical_significance",
              "run_evidence_digests": [r["evidence_digest"] for r in runs],
              "imported_runs": runs}
    write_report(output, report)
    ledger.record(report)
    return report


def authorize_g3(ledger: GateLedger, operation: str, *, test=None,
                 output: str | Path | None = None) -> dict:
    """Register an explicit locked-test operation; do not launch full evaluation."""
    gate = ledger.require_g3()
    if not operation.strip():
        raise ValueError("G3 requires a named predeclared operation")
    if test is not None and test.split != "test":
        raise ValueError("G3 test operation requires a test split")
    if test is not None:
        if test.metadata.get("natural_distribution") is not True:
            raise ValueError("locked test must preserve natural distribution")
        g0 = next(item["report"] for item in reversed(ledger.state["history"])
                  if item.get("report", {}).get("stage") == "G0"
                  and item["report"]["data_signature"] == gate["data_signature"])
        identities = g0.get("identities", {})
        patients = {canonical_patient(p) for p in
                    (test.metadata.get("population_patient_ids") or test.patient_ids.tolist())}
        development = set(identities.get("train_patients", [])) | set(identities.get("validation_patients", []))
        if patients & development:
            raise ValueError("locked test patients overlap development/train identities")
        records = set(test.metadata.get("population_record_ids") or test.record_ids.tolist())
        if records & (set(identities.get("train_records", []))
                      | set(identities.get("validation_records", []))):
            raise ValueError("locked test records overlap development/train identities")
        hashes = set(test.metadata.get("population_source_hashes", []))
        if hashes & (set(identities.get("train_source_hashes", []))
                     | set(identities.get("validation_source_hashes", []))):
            raise ValueError("locked test source files overlap development/train identities")
    prior = ledger.state["g3_operations"]
    if any(item["operation"] == operation for item in prior):
        raise ValueError("G3 operation was already registered; avoid repeated test selection")
    report = {"stage": "G3", "operation": operation, "status": "authorized_not_executed",
              "gate_evidence_digest": gate["evidence_digest"],
              "data_signature": gate["data_signature"],
              "test_digest": test.digest if test is not None else None,
              "full_evaluation_executed": False, "SOTA_claim": False}
    write_report(output, report)
    prior.append(report)
    ledger._save()
    return report
