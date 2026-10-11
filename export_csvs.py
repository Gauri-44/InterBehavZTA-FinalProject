"""
export_csvs.py

Exports the database tables cited by the paper to CSV files under
backend/results/database_exports/. Read-only. One run produces the
entire database_exports folder.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from models import (
    db,
    TrustHistory,
    CBACResult,
    AnomalyResult,
    ResourceAccessDecision,
    BehavioralTrustScore,
    Experiment,
    ExperimentObservation,
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "database_exports",
)


def export_table(model, filename, columns=None):
    """Export every row of a model to a CSV with a header row."""
    rows = model.query.all()
    print(f"  {model.__tablename__:<30} {len(rows):>6} rows -> {filename}")

    if not rows:
        # Still write an empty CSV with headers
        if columns:
            with open(os.path.join(OUT_DIR, filename), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(columns)
        return

    if columns is None:
        # Use model columns
        columns = [c.name for c in model.__table__.columns]

    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in rows:
            w.writerow([getattr(row, c, None) for c in columns])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    app = create_app()
    with app.app_context():
        print(f"📂 Exporting to {OUT_DIR}")
        export_table(TrustHistory, "trust_history.csv")
        export_table(CBACResult, "cbac_results.csv")
        export_table(AnomalyResult, "anomaly_results.csv")
        export_table(ResourceAccessDecision, "resource_access_decisions.csv")
        export_table(BehavioralTrustScore, "behavioral_trust_scores.csv")
        export_table(Experiment, "experiments.csv")
        export_table(ExperimentObservation, "experiment_observations.csv")
        print("✅ Done.")


if __name__ == "__main__":
    main()