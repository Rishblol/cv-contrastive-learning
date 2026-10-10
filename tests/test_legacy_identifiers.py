from __future__ import annotations

import pandas as pd
import pytest

from chexpert_ssl.data import assert_patient_disjoint, read_manifest, select_patient_cohort
from chexpert_ssl.utils import save_json


@pytest.mark.parametrize("manifest_ids", [["1", "2"], ["00001", "00002"]])
@pytest.mark.parametrize("cohort_ids", [["1", "2"], ["00001", "00002"]])
def test_legacy_padding_restoration_preserves_cohort_and_manifest(
    tmp_path, manifest_ids, cohort_ids
):
    manifest = tmp_path / "legacy.csv"
    pd.DataFrame(
        {
            "patient_id": manifest_ids,
            "Path": ["train/patient00001/study1/image.png", "train/patient00002/study1/image.png"],
            "image_path": ["a.png", "b.png"],
        }
    ).to_csv(manifest, index=False)
    cohort = tmp_path / "cohort.json"
    save_json(cohort, {"seed": 42, "label_fraction": 1.0, "patient_ids": cohort_ids})
    manifest_bytes, cohort_bytes = manifest.read_bytes(), cohort.read_bytes()
    frame = read_manifest(manifest, labeled=False)
    assert frame.patient_id.tolist() == ["00001", "00002"]
    assert frame.attrs["patient_id_padding_restored_rows"] == (2 if manifest_ids[0] == "1" else 0)
    selected = select_patient_cohort(frame, cohort, 1.0, 42)
    assert selected.image_path.tolist() == ["a.png", "b.png"]
    assert selected.attrs["cohort_patient_id_padding_restored"] == (
        2 if cohort_ids[0] == "1" else 0
    )
    assert manifest.read_bytes() == manifest_bytes
    assert cohort.read_bytes() == cohort_bytes


def test_genuine_path_patient_conflicts_still_fail(tmp_path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame(
        {
            "patient_id": ["2"],
            "Path": ["train/patient00001/study1/image.png"],
            "image_path": ["a.png"],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="inconsistent"):
        read_manifest(path, labeled=False)


def test_disjointness_detects_padding_aliases():
    first = pd.DataFrame({"patient_id": ["00001"]})
    second = pd.DataFrame({"patient_id": ["1"]})
    with pytest.raises(ValueError, match="leakage"):
        assert_patient_disjoint(first, second)


def test_duplicate_padding_aliases_in_cohort_are_rejected(tmp_path):
    frame = pd.DataFrame({"patient_id": ["00001", "00002"], "image_path": ["a", "b"]})
    path = tmp_path / "cohort.json"
    save_json(path, {"seed": 42, "label_fraction": 1.0, "patient_ids": ["1", "00001"]})
    with pytest.raises(ValueError, match="duplicate"):
        select_patient_cohort(frame, path, 1.0, 42)
