from __future__ import annotations

import pandas as pd
import pytest
from notebook_runtime import (
    TARGETS,
    assert_patient_disjoint,
    exclude_development_patients,
    nested_patient_ids,
    resolve_target,
    save_json,
    select_patient_cohort,
)


def test_uncertainty_policy_matches_declared_protocol() -> None:
    assert resolve_target(-1, "Atelectasis") == 1
    assert resolve_target(-1, "Edema") == 1
    assert resolve_target(-1, "Cardiomegaly") == 0
    assert resolve_target(float("nan"), "Pleural Effusion") == 0
    assert resolve_target(1, "Consolidation") == 1


def test_patient_overlap_is_rejected() -> None:
    first = pd.DataFrame({"patient_id": ["1", "2"]})
    second = pd.DataFrame({"patient_id": ["2", "3"]})
    with pytest.raises(ValueError, match="leakage"):
        assert_patient_disjoint(first, second)


def test_pretraining_manifest_repair_filters_development_patients() -> None:
    pretrain = pd.DataFrame(
        {"patient_id": ["1", "1", "2", "3"], "image_path": ["a", "b", "c", "d"]}
    )
    development = pd.DataFrame({"patient_id": ["2"]})

    repaired, removed = exclude_development_patients(pretrain, development)

    assert repaired["patient_id"].tolist() == ["1", "1", "3"]
    assert removed == 1
    assert not set(repaired["patient_id"]) & set(development["patient_id"])


def test_patient_budget_keeps_all_studies_for_selected_patients(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["1", "1", "2", "2", "3", "3", "4", "4"],
            **{f"target_{target}": [0] * 8 for target in TARGETS},
        }
    )
    patients = nested_patient_ids(frame, [0.5], seed=3)[0.5]
    cohort = tmp_path / "cohort.json"
    save_json(cohort, {"seed": 3, "label_fraction": 0.5, "patient_ids": patients})
    sampled = select_patient_cohort(frame, cohort, fraction=0.5, seed=3)
    counts = sampled.groupby("patient_id").size()
    assert len(counts) == 2
    assert set(counts) == {2}
