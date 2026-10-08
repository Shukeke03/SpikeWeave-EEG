import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from triwave.evaluation import (
    EvaluationData,
    G1Config,
    GateLedger,
    authorize_g3,
    average_precision,
    evaluation_data_from_dataset,
    load_evaluation_data,
    patient_average_precision,
    record_g2,
    run_g0,
    run_g1,
)
from triwave.evaluation.gates import write_report
from triwave.evaluation.metrics import (
    frechet_distance,
    nearest_copy_audit,
    source_scaled_mmd,
    spectral_ts_features,
)


def _data(split="train", seed=0, count=16, patient="p_train"):
    rng = np.random.default_rng(seed)
    labels = np.arange(count) % 2
    waveforms = rng.normal(0, 8, (count, 2, 64)).astype(np.float32)
    waveforms += labels[:, None, None] * np.sin(np.arange(64)[None, None] * 0.5) * 15
    metadata = {"lead_names": ["A-B", "B-C"], "label_namespace": "fixture.window_event",
                "label_scope": "1s_window_annotation_overlap", "natural_distribution": True,
                "engineering_fixture": True, "evaluation_protocol": "source-only"}
    return EvaluationData(waveforms, labels, np.asarray([patient] * count),
                          np.asarray([f"{split}_record"] * count),
                          np.asarray([f"{split}_{i}" for i in range(count)]),
                          split, 64, role="generated" if split == "generated" else "source",
                          metadata=metadata)


def _generated(train, name="spikeweave", seed=20):
    data = _data("generated", seed=seed, patient="synthetic")
    data.role = "generated"
    data.metadata.update({"algorithm": name, "code_commit": "fixture_commit_not_clinical_run",
                          "budget": {"generator_updates": 2, "engineering_fixture": True},
                          "source_train_digest": train.digest, "condition_origin": "source_request"})
    return data


def _save_npz(path, data):
    metadata = data.metadata | {"split": data.split, "role": data.role, "unit": data.unit,
                                "sample_rate": data.sample_rate}
    np.savez(path, waveforms=data.waveforms, labels=data.labels, patient_ids=data.patient_ids,
             record_ids=data.record_ids, sample_ids=data.sample_ids, metadata=json.dumps(metadata))


def test_average_precision_ties_and_undefined_positive_population():
    expected = 0.5 * 0.5 + 0.5 * (2 / 3)
    assert average_precision(np.array([1, 0, 1]), np.array([0.5, 0.5, 0.1])) == pytest.approx(expected)
    assert average_precision(np.array([0, 1, 1]), np.array([0.5, 0.5, 0.1])) == pytest.approx(expected)
    assert average_precision(np.zeros(3), np.arange(3)) is None
    assert average_precision(np.ones(3), np.arange(3)) == 1
    with pytest.raises(ValueError, match="finite scores"):
        average_precision(np.array([1, 0]), np.array([np.nan, 0]))


def test_primary_ap_equal_patient_weight_and_background_coverage():
    labels = np.array([1, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.1, 0.9, 0.8, 0.7])
    report = patient_average_precision(labels, scores, np.array(["a", "a", "b", "b", "c", "c"]))
    assert report["macro_patient_ap"] == pytest.approx(0.75)
    assert report["computable_patients"] == 2 and report["total_patients"] == 3
    assert report["patients"]["c"]["ap"] is None
    assert report["patients"]["c"]["background_above_0_5"] == 2
    assert report["pooled_ap"] != report["macro_patient_ap"]


def test_spectral_fid_is_phase_invariant_and_low_rank_frechet_exact():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, 2, 64))
    features = spectral_ts_features(x, 64)
    np.testing.assert_allclose(features, spectral_ts_features(np.roll(x, 7, axis=-1), 64), atol=1e-10)
    assert features.shape == (5, 1024)
    assert frechet_distance(features, features) == pytest.approx(0, abs=1e-8)
    real, fake = np.array([[0.0], [2.0], [4.0]]), np.array([[1.0], [5.0], [9.0]])
    expected = (real.mean() - fake.mean()) ** 2 + (real.std(ddof=1) - fake.std(ddof=1)) ** 2
    assert frechet_distance(real, fake) == pytest.approx(expected)


def test_physical_mmd_is_source_fitted_and_copy_audit_has_bounded_scope():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(5, 10))
    report = source_scaled_mmd(source, source, source)
    assert report["mmd_squared"] == pytest.approx(0)
    assert report["scale_fit"] == "source_train_only"
    waveforms = rng.normal(size=(8, 2, 64)).astype(np.float32)
    copy_report = nearest_copy_audit(waveforms[:2], waveforms, max_source=8)
    assert copy_report["near_exact_count"] == 2
    assert copy_report["source_windows_screened"] == 8
    assert "not_membership" in copy_report["scope"]


def test_g0_rejects_hidden_patient_alias_leakage_and_non_natural_validation(tmp_path):
    train, validation = _data(), _data("validation", seed=3, patient="p_val")
    ledger = GateLedger(tmp_path / "ledger.json")
    report = run_g0(train, validation, ledger=ledger)
    assert report["gate_passed"] and report["numerical"]["numerical_source_only"]
    assert report["numerical"]["montage"]["rank"] == 2
    validation.patient_ids[:] = "chb21"
    train.patient_ids[:] = "chb01"
    # Arrays are mutable, so runtime audits canonicalize population identities.
    train.metadata["population_patient_ids"] = ["chb01"]
    validation.metadata["population_patient_ids"] = ["chb21"]
    report = run_g0(train, validation, ledger=ledger)
    assert not report["gate_passed"] and any("patient" in e for e in report["errors"])
    with pytest.raises(ValueError, match="passing recorded G0"):
        ledger.require("G0")
    validation.metadata["natural_distribution"] = False
    assert any("natural-distribution" in e for e in run_g0(train, validation)["errors"])


def test_npz_load_requires_explicit_metadata_and_protects_test_before_waveform_read(tmp_path):
    data = _data("validation", seed=2, patient="v")
    path = tmp_path / "validation.npz"
    _save_npz(path, data)
    loaded = load_evaluation_data(path)
    np.testing.assert_array_equal(loaded.waveforms, data.waveforms)
    assert loaded.metadata["import_sha256"]
    metadata = data.metadata | {"split": "test", "role": "query", "unit": "uV", "sample_rate": 64}
    # This object array would fail pickle=False if read; the split gate fires first.
    np.savez(tmp_path / "test.npz", metadata=json.dumps(metadata), waveforms=np.array([object()]))
    with pytest.raises(ValueError, match="locked test"):
        load_evaluation_data(tmp_path / "test.npz", stage="G1")
    with pytest.raises(ValueError, match="passing recorded G2"):
        load_evaluation_data(tmp_path / "test.npz", stage="G3", ledger=GateLedger())


def test_dataset_adapter_never_calls_getitem_or_reads_conditions():
    class FakePhysicalAssets:
        def load(self, row):
            return np.ones((2, 64), dtype=np.float32) * (int(row["offset"]) + 1)

    class FakeDataset:
        split = "validation"
        config = type("Config", (), {"sample_rate": 64, "samples": 64,
                                     "lead_names": ("WRONG1", "WRONG2", "WRONG3")})()
        normalizer = None
        records = [{"canonical_subject_id": "v", "record_id": "r", "source_sha256": "hash_r",
                    "split": "validation", "unit": "uV", "lead_names": ["A-B", "B-C"]}]
        window_index = np.array([(0, 0, 0), (0, 64, 1)],
                                dtype=[("record", "i4"), ("offset", "i8"), ("label", "i1")])
        indices = np.arange(2)
        assets = FakePhysicalAssets()

        def __len__(self):
            return 2

        def __getitem__(self, index):
            raise AssertionError("evaluator accessed a condition-generating item")

    result = evaluation_data_from_dataset(FakeDataset())
    assert result.metadata["lead_names"] == ["A-B", "B-C"]
    assert result.metadata["full_identity_available"]
    assert result.metadata["natural_distribution"]
    assert result.patient_ids.tolist() == ["v", "v"]
    assert result.waveforms[1, 0, 0] == 65


def test_g1_runs_budget_matched_cnn_quality_and_declared_baseline_imports(tmp_path):
    torch.set_num_threads(2)
    train, validation = _data(), _data("validation", 5, count=16, patient="v")
    ledger = GateLedger(tmp_path / "ledger.json")
    assert run_g0(train, validation, ledger=ledger)["gate_passed"]
    imports = {name: _generated(train, name, seed=seed)
               for name, seed in (("spikeweave", 20), ("jet", 21), ("conditional_vp", 22))}
    report = run_g1(train, validation, imports, config=G1Config(updates=2, batch_size=4),
                    ledger=ledger, output=tmp_path / "g1.json")
    assert report["status"] == "complete" and not report["failures"]
    assert not report["SOTA_claim"] and not report["test_used"]
    assert set(report["downstream"]) == {"real_only", "repeat", "normalaug", *imports}
    for name in report["downstream"]:
        audit = report["downstream"][name]["training_audit"]
        assert audit["optimizer_updates"] == 2 and audit["batch_size"] == 4
        assert not audit["conditions_to_detector"]
    assert report["imports"]["jet"]["status"] == "explicit_precomputed_import"
    assert not report["imports"]["jet"]["faithful_author_reproduction_verified"]
    for row in report["quality"]["spikeweave"]["classes"].values():
        assert row["matched_windows"] == 8
        assert "physical_mmd" in row and "ts_fid" in row
    repeated = run_g1(train, validation, imports, config=G1Config(updates=2, batch_size=4))
    assert repeated["downstream"]["spikeweave"]["predictions"] == report["downstream"]["spikeweave"]["predictions"]


def test_missing_or_misnamed_strong_baseline_never_passes_g1_gate():
    torch.set_num_threads(2)
    train, validation = _data(), _data("validation", 5, patient="v")
    imposter = _generated(train, "spikeweave", seed=22)
    report = run_g1(train, validation, {"spikeweave": _generated(train), "jet": imposter},
                    config=G1Config(updates=1, batch_size=4))
    assert not report["gate_passed"] and report["status"] == "incomplete"
    assert "conditional_vp" in report["missing_methods"] and "jet" in report["missing_methods"]
    assert any("alias" in f["error"] for f in report["failures"])


def test_generated_source_provenance_and_quality_budget_cannot_be_silently_reduced():
    torch.set_num_threads(2)
    train, validation = _data(), _data("validation", 5, patient="v")
    wrong_source = _generated(train)
    wrong_source.metadata["source_train_digest"] = "another_source"
    report = run_g1(train, validation, {"spikeweave": wrong_source},
                    config=G1Config(updates=1, batch_size=4))
    assert any("allowed source" in f["error"] for f in report["failures"])
    small = _generated(train)
    chosen = np.r_[np.flatnonzero(small.labels == 0)[:4], np.flatnonzero(small.labels == 1)[:4]]
    limited = EvaluationData(small.waveforms[chosen], small.labels[chosen], small.patient_ids[chosen],
                             small.record_ids[chosen], small.sample_ids[chosen], "generated", 64,
                             role="generated", metadata=small.metadata)
    report = run_g1(train, validation, {"spikeweave": limited},
                    config=G1Config(updates=1, batch_size=4))
    assert not report["quality"]["spikeweave"]["minimum_screen_passed"]
    for row in report["quality"]["spikeweave"]["classes"].values():
        assert row["required_generated_windows"] == 8
        assert row["matched_windows"] == 0


def test_test_split_cannot_enter_g0_or_g1_directly():
    torch.set_num_threads(2)
    train, test = _data(), _data("test", 5, patient="test")
    assert not run_g0(train, test)["gate_passed"]
    with pytest.raises(ValueError, match="locked test"):
        run_g1(train, test, {}, config=G1Config(updates=1, batch_size=4))


def _confirmation_report(seed, candidate_ap=0.8, baseline_ap=0.5):
    methods = {name: {"macro_patient_ap": baseline_ap}
               for name in ("real_only", "repeat", "normalaug", "jet", "conditional_vp")}
    methods["spikeweave"] = {"macro_patient_ap": candidate_ap}
    imports = {name: {"generator_training_seed": seed,
                      "generator_checkpoint_digest": hashlib.sha256(f"{name}:{seed}".encode()).hexdigest(),
                      "algorithm": name, "code_commit": hashlib.sha256(f"code:{name}".encode()).hexdigest(),
                      "implementation_sha256": hashlib.sha256(f"impl:{name}".encode()).hexdigest(),
                      "generator_recipe_digest": hashlib.sha256(f"recipe:{name}".encode()).hexdigest(),
                      "source_train_digest": hashlib.sha256(b"source").hexdigest(),
                      "generator_condition_budget": {"class_counts": {"0": 4, "1": 4}},
                      "budget": {"training_optimizer_steps": 2, "sampling_steps": 2},
                      "condition_origin": "source_request"}
               for name in ("spikeweave", "jet", "conditional_vp")}
    return write_report(None, {"stage": "G1", "candidate": "spikeweave", "seed": seed,
                               "status": "complete", "data_signature": "fixture_signature",
                               "budget_digest": "fixture_budget", "required_baselines": ["jet", "conditional_vp"],
                               "quality_minimum_passed": True, "gate_passed": True,
                               "imports": imports,
                               "downstream": methods})


def _open_development_ledger(path=None):
    ledger = GateLedger(path)
    ledger.record(write_report(None, {"stage": "G0", "data_signature": "fixture_signature",
                                      "gate_passed": True}))
    ledger.record(_confirmation_report(0))
    return ledger


def test_g2_evidence_and_g3_registration_are_explicit_and_digest_bound(tmp_path):
    ledger = _open_development_ledger(tmp_path / "ledger.json")
    runs = [_confirmation_report(1), _confirmation_report(2, 0.45), _confirmation_report(3)]
    g2 = record_g2(ledger, runs)
    assert g2["gate_passed"] and sum(v > 0 for v in g2["paired_differences"]) == 2
    loaded = GateLedger(tmp_path / "ledger.json")
    result = authorize_g3(loaded, "locked_final_benchmark")
    assert result["status"] == "authorized_not_executed" and not result["SOTA_claim"]
    with pytest.raises(ValueError, match="already registered"):
        authorize_g3(loaded, "locked_final_benchmark")
    corrupted = copy.deepcopy(runs)
    corrupted[0]["downstream"]["spikeweave"]["macro_patient_ap"] = 0.99
    with pytest.raises(ValueError, match="digest"):
        record_g2(loaded, corrupted)


def test_g2_rejects_repeated_seeds_budgets_and_one_lucky_seed():
    ledger = _open_development_ledger()
    with pytest.raises(ValueError, match="distinct seeds"):
        record_g2(ledger, [_confirmation_report(1)] * 3)
    runs = [_confirmation_report(1), _confirmation_report(2), _confirmation_report(3)]
    runs[2]["budget_digest"] = "another_budget"
    write_report(None, runs[2])
    with pytest.raises(ValueError, match="identical data"):
        record_g2(ledger, runs)
    recycled = [_confirmation_report(i) for i in (1, 2, 3)]
    recycled[2]["imports"]["spikeweave"] = recycled[0]["imports"]["spikeweave"]
    write_report(None, recycled[2])
    with pytest.raises(ValueError, match="distinct generator checkpoints"):
        record_g2(ledger, recycled)
    one_lucky = [_confirmation_report(1, 0.99), _confirmation_report(2, 0.49),
                 _confirmation_report(3, 0.49)]
    report = record_g2(ledger, one_lucky)
    assert not report["gate_passed"]
    with pytest.raises(ValueError, match="passing recorded G2"):
        ledger.require_g3()


def test_failed_development_rerun_invalidates_later_test_access():
    ledger = _open_development_ledger()
    record_g2(ledger, [_confirmation_report(i) for i in (1, 2, 3)])
    ledger.require_g3()
    ledger.record_failure("G1", "changed configuration failed", seed=4)
    with pytest.raises(ValueError, match="passing recorded G2"):
        ledger.require_g3()
    assert ledger.state["history"][-1]["status"] == "failed"


@pytest.mark.parametrize("field,replacement", [
    ("budget", {"training_optimizer_steps": 2000, "sampling_steps": 2}),
    ("code_commit", "a" * 40),
    ("implementation_sha256", "b" * 64),
    ("generator_recipe_digest", "c" * 64),
    ("generator_condition_budget", {"class_counts": {"0": 4, "1": 8}}),
    ("source_train_digest", "d" * 64),
    ("algorithm", "another_candidate_recipe"),
])
def test_g2_rejects_different_generator_budget_code_recipe_and_conditions(field, replacement):
    ledger = _open_development_ledger()
    runs = [_confirmation_report(i) for i in (1, 2, 3)]
    runs[2]["imports"]["spikeweave"][field] = replacement
    write_report(None, runs[2])
    with pytest.raises(ValueError, match=f"fixed {field}"):
        record_g2(ledger, runs)


def test_g2_missing_generator_evidence_names_field_and_exception_closes_old_gate():
    ledger = _open_development_ledger()
    record_g2(ledger, [_confirmation_report(i) for i in (1, 2, 3)])
    ledger.require_g3()
    runs = [_confirmation_report(i) for i in (4, 5, 6)]
    del runs[1]["imports"]["jet"]["generator_recipe_digest"]
    write_report(None, runs[1])
    try:
        record_g2(ledger, runs)
    except ValueError as exception:
        assert "jet" in str(exception) and "generator_recipe_digest" in str(exception)
        # This is the CLI's explicit exception-recording path.
        ledger.record_failure("G2", exception, seed=5)
    else:
        raise AssertionError("G2 accepted missing evidence")
    ledger.require("G0")
    ledger.require("G1")
    with pytest.raises(ValueError, match="passing recorded G2"):
        authorize_g3(ledger, "must_not_open_after_failure")
    assert ledger.state["history"][-1]["stage"] == "G2"
    assert ledger.state["history"][-1]["status"] == "failed"
