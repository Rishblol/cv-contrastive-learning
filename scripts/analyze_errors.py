"""Create an auditable error-analysis worksheet from prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from chexpert_ssl.data import TARGETS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    rows = []
    for _, record in frame.iterrows():
        for label in TARGETS:
            probability = record[f"probability_{label}"]
            truth = record[f"target_{label}"]
            predicted = int(probability >= float(thresholds[label]))
            if predicted != truth:
                rows.append({"run_id": args.run_id, "image_id": record["image_path"], "patient_id": record["patient_id"], "study_id": record["study_id"], "view": record["Frontal/Lateral"], "label": label, "true_label": truth, "predicted_probability": probability, "selected_threshold": float(thresholds[label]), "attribution_artifact_path": "", "reviewer_observation": ""})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
