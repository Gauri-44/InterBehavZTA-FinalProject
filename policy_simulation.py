"""
policy_simulation.py

Policy-level evaluation of InterBehav-ZTA (§V-B of the paper).

Four corrections, each grounded in a specific section of the paper:

  Fix A — severity is calibrated per user against the enrollment median,
          as specified in Section III-E. Sigmoid with slope 8 around the
          user's median maps raw combined_score to severity in (0,1)
          with 0.5 at the median.

  Fix C — trust resets to T0 = 0.75 at every session boundary. The threat
          model (Section III-A) is session-token compromise; a new login
          starts a fresh trust state and trust does not persist across
          sessions. Sessions are grouped by telemetry_session_id, with a
          5-minute gap heuristic for windows that lack a session link.

  Fix D — TTDRM trust is floored by the CBAC class at each window.
          Section III-F defines STRESS as a benign contextual state
          ("not a measurement of emotion") that must not be treated as
          credential misuse. Without a floor, TTDRM would treat STRESS
          and MALICIOUS identically because the fused severity does not
          carry the class. The floor encodes the paper's stated design:
            NORMAL    -> floor 0.60
            STRESS    -> floor 0.50
            MALICIOUS -> floor 0.00 (no floor)

  Fix E — evaluation is per-window, not per-session, so the 372 labeled
          windows remain the sample size.

Five configurations compared:
  P1  Naive anomaly threshold
  P2  SVM-only threshold
  P3  CBAC-only
  P4  Full pipeline (TTDRM + calibration + class floor + Eq. 23)
  P5  Full pipeline without TTDRM (CBAC signed toward NORMAL + Eq. 23)

Success definition (Option A):
  A malicious window is "restricted" if it receives STEP-UP or DENY.
  UAA = malicious windows that received ALLOW.

Output: results/policy_simulation_results.json
Read-only. No database writes.
"""

import os
import sys
import json
import math
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from models import (
    db,
    User,
    ExperimentObservation,
    FeatureVector,
    TelemetryWindow,
    AnomalyResult,
    CBACResult,
)

# ============================================================
# Config
# ============================================================

MIN_WINDOWS_PER_USER = 20

TTDRM_LAMBDA = 0.40
TTDRM_GAMMA = 0.02
TTDRM_T0 = 0.75

CALIBRATION_SLOPE = 8.0

# Fix D — CBAC class-dependent trust floors (Section III-F)
TRUST_FLOOR = {
    "normal":    0.60,
    "stress":    0.50,
    "malicious": 0.00,
}

FALLBACK_THRESHOLDS = {
    "Company Dashboard": 0.50,
    "Google Drive":      0.50,
    "Gmail":             0.55,
    "Calendar":          0.55,
    "Analytics":         0.65,
    "Database":          0.65,
    "Code Repository":   0.65,
    "HR Portal":         0.72,
    "Admin Console":     0.80,
}

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "policy_simulation_results.json",
)

# ============================================================
# Data loading
# ============================================================

def load_labeled_windows():
    print("📂 Loading labeled windows with complete pipeline output...")

    rows = (
        db.session.query(
            ExperimentObservation,
            FeatureVector,
            AnomalyResult,
            CBACResult,
            TelemetryWindow,
        )
        .join(FeatureVector, ExperimentObservation.feature_vector_id == FeatureVector.id)
        .join(AnomalyResult, AnomalyResult.feature_vector_id == FeatureVector.id)
        .join(CBACResult, CBACResult.feature_vector_id == FeatureVector.id)
        .outerjoin(TelemetryWindow, FeatureVector.telemetry_window_id == TelemetryWindow.id)
        .filter(ExperimentObservation.is_valid.is_(True))
        .filter(ExperimentObservation.ground_truth_label.in_(
            ["normal", "stress", "malicious"]
        ))
        .all()
    )

    print(f"   Found {len(rows)} complete labeled windows.")

    windows = []
    for obs, fv, ar, cb, tw in rows:
        if ar.combined_score is None:
            continue
        session_id = tw.telemetry_session_id if tw else None
        windows.append({
            "user_id": fv.user_id,
            "obs_id": obs.id,
            "session_id": session_id,
            "ground_truth": obs.ground_truth_label,
            "combined_score": float(ar.combined_score),
            "svm_score": float(ar.svm_score) if ar.svm_score is not None else 0.0,
            "lstm_score": float(ar.lstm_score) if ar.lstm_score is not None else 0.0,
            "cbac_class": (cb.predicted_class or "normal").lower(),
            "prob_normal": float(cb.prob_normal) if cb.prob_normal is not None else 0.0,
            "prob_stress": float(cb.prob_stress) if cb.prob_stress is not None else 0.0,
            "prob_malicious": float(cb.prob_malicious) if cb.prob_malicious is not None else 0.0,
            "created_at": obs.created_at,
        })

    user_ids = {w["user_id"] for w in windows}
    emails = {u.id: u.email for u in User.query.filter(User.id.in_(user_ids)).all()}
    for w in windows:
        w["user_email"] = emails.get(w["user_id"], f"user_{w['user_id']}")

    return windows


def group_by_user_and_session(windows):
    """
    Returns {user_id: [ [session_windows...], ... ]}
    Sessions keyed by telemetry_session_id; 5-min gap heuristic when
    session_id is None.
    """
    by_user = defaultdict(list)
    for w in windows:
        by_user[w["user_id"]].append(w)

    out = defaultdict(list)
    for uid, ws in by_user.items():
        ws.sort(key=lambda w: w["created_at"] or datetime.min)
        sessions = []
        current = []
        current_sid = None
        prev_time = None
        for w in ws:
            sid = w["session_id"]
            t = w["created_at"]
            new_session = False
            if sid is None or current_sid is None:
                if prev_time is not None and t is not None:
                    if (t - prev_time).total_seconds() > 300:
                        new_session = True
            else:
                if current_sid is not None and sid != current_sid:
                    new_session = True
            if new_session and current:
                sessions.append(current)
                current = []
            current.append(w)
            current_sid = sid
            prev_time = t
        if current:
            sessions.append(current)
        out[uid] = sessions

    return out


def load_resources():
    try:
        from models import Resource
        rows = Resource.query.all()
        if rows:
            out = {r.name: float(r.required_trust_score or 0.50) for r in rows}
            if out:
                return out
    except Exception:
        pass
    return dict(FALLBACK_THRESHOLDS)


# ============================================================
# Fix A — per-user severity calibration
# ============================================================

def compute_user_calibration(grouped):
    calib = {}
    for uid, sessions in grouped.items():
        all_scores = [w["combined_score"] for sess in sessions for w in sess]
        if not all_scores:
            continue
        all_scores.sort()
        n = len(all_scores)
        median = all_scores[n // 2]
        q1 = all_scores[n // 4]
        q3 = all_scores[(3 * n) // 4]
        spread = max((q3 - q1) / 2.0, 0.05)
        calib[uid] = (median, spread)
    return calib


def calibrate_severity(raw_score, median, spread):
    x = (raw_score - median) / spread * CALIBRATION_SLOPE
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ============================================================
# Decision primitives
# ============================================================

def decide_naive(combined_score):
    return "step_up" if combined_score >= 0.5 else "allow"


def decide_svm(svm_score):
    return "step_up" if svm_score >= 0.5 else "allow"


def decide_cbac(cbac_class):
    c = (cbac_class or "").lower()
    if c == "malicious":
        return "deny"
    if c == "stress":
        return "step_up"
    return "allow"


def ttdrm_update_with_floor(trust, severity, cbac_class):
    """Fix D — TTDRM update with a class-dependent floor."""
    decay = TTDRM_LAMBDA * severity * trust
    recovery = TTDRM_GAMMA * (1.0 - severity) * (1.0 - trust)
    t = trust - decay + recovery
    floor = TRUST_FLOOR.get((cbac_class or "").lower(), 0.0)
    t = max(t, floor)
    return max(0.0, min(1.0, t))


def decide_policy(bts, threshold):
    hi = max(0.70, threshold)
    if bts >= hi:
        return "allow"
    if bts >= threshold:
        return "step_up"
    return "deny"


# ============================================================
# Evaluation
# ============================================================

def evaluate_config(grouped, calib, mode, resource_threshold):
    stats = {
        "malicious_total": 0, "malicious_allow": 0,
        "malicious_step_up": 0, "malicious_deny": 0,
        "normal_total": 0, "normal_allow": 0,
        "normal_step_up": 0, "normal_deny": 0,
        "stress_total": 0, "stress_allow": 0,
        "stress_step_up": 0, "stress_deny": 0,
        "time_to_restrict": [],
        "recovery_windows": [],
        "n_sessions": 0,
    }

    for uid, sessions in grouped.items():
        for sess in sessions:
            stats["n_sessions"] += 1
            # Fix C — trust resets at session boundary
            trust = TTDRM_T0
            median, spread = calib.get(uid, (0.5, 0.15))

            first_malicious_idx = None
            first_restriction_idx = None
            first_false_restriction_idx = None
            first_recovery_idx = None

            for idx, w in enumerate(sess):
                gt = w["ground_truth"]
                cbac_class = w["cbac_class"]

                if mode == "p1":
                    decision = decide_naive(w["combined_score"])
                elif mode == "p2":
                    decision = decide_svm(w["svm_score"])
                elif mode == "p3":
                    decision = decide_cbac(cbac_class)
                elif mode == "p4":
                    sev = calibrate_severity(w["combined_score"], median, spread)
                    # Fix D — pass CBAC class into TTDRM update
                    trust = ttdrm_update_with_floor(trust, sev, cbac_class)
                    decision = decide_policy(trust, resource_threshold)
                elif mode == "p5":
                    c = w["prob_normal"] - (w["prob_stress"] + w["prob_malicious"])
                    bts = max(0.0, min(1.0, (c + 1.0) / 2.0))
                    decision = decide_policy(bts, resource_threshold)
                else:
                    decision = "allow"

                if gt == "malicious":
                    stats["malicious_total"] += 1
                    if first_malicious_idx is None:
                        first_malicious_idx = idx
                    if decision == "allow":
                        stats["malicious_allow"] += 1
                    elif decision == "step_up":
                        stats["malicious_step_up"] += 1
                        if first_restriction_idx is None:
                            first_restriction_idx = idx
                    else:
                        stats["malicious_deny"] += 1
                        if first_restriction_idx is None:
                            first_restriction_idx = idx

                elif gt == "normal":
                    stats["normal_total"] += 1
                    if decision == "allow":
                        stats["normal_allow"] += 1
                        if first_false_restriction_idx is not None and first_recovery_idx is None:
                            first_recovery_idx = idx
                    elif decision == "step_up":
                        stats["normal_step_up"] += 1
                        if first_false_restriction_idx is None:
                            first_false_restriction_idx = idx
                    else:
                        stats["normal_deny"] += 1
                        if first_false_restriction_idx is None:
                            first_false_restriction_idx = idx

                else:  # stress
                    stats["stress_total"] += 1
                    if decision == "allow":
                        stats["stress_allow"] += 1
                    elif decision == "step_up":
                        stats["stress_step_up"] += 1
                    else:
                        stats["stress_deny"] += 1

            if first_malicious_idx is not None and first_restriction_idx is not None:
                stats["time_to_restrict"].append(
                    first_restriction_idx - first_malicious_idx
                )
            if (first_false_restriction_idx is not None
                    and first_recovery_idx is not None
                    and first_recovery_idx > first_false_restriction_idx):
                stats["recovery_windows"].append(
                    first_recovery_idx - first_false_restriction_idx
                )

    return finalize_stats(stats)


def finalize_stats(s):
    def rate(num, den):
        return (num / den) if den > 0 else float("nan")

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else float("nan")

    mal = s["malicious_total"]
    nor = s["normal_total"]
    str_ = s["stress_total"]

    return {
        "malicious_total": mal,
        "malicious_allow": s["malicious_allow"],
        "malicious_step_up": s["malicious_step_up"],
        "malicious_deny": s["malicious_deny"],
        "uaa_stepup_deny":   rate(s["malicious_allow"], mal),
        "detection_rate":    rate(s["malicious_step_up"] + s["malicious_deny"], mal),
        "malicious_deny_rate": rate(s["malicious_deny"], mal),

        "normal_total": nor,
        "normal_allow": s["normal_allow"],
        "normal_step_up": s["normal_step_up"],
        "normal_deny": s["normal_deny"],
        "false_stepup_rate": rate(s["normal_step_up"], nor),
        "false_deny_rate":   rate(s["normal_deny"], nor),

        "stress_total": str_,
        "stress_step_up_rate": rate(s["stress_step_up"], str_),
        "stress_deny_rate":    rate(s["stress_deny"], str_),

        "mean_time_to_restrict_windows": mean(s["time_to_restrict"]),
        "mean_recovery_windows": mean(s["recovery_windows"]),
        "n_users_with_malicious": len(s["time_to_restrict"]),
        "n_users_with_false_restrict": len(s["recovery_windows"]),
        "n_sessions": s["n_sessions"],
    }


# ============================================================
# Main
# ============================================================

def pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x*100:.1f}%"


def main():
    app = create_app()
    with app.app_context():
        windows = load_labeled_windows()
        if not windows:
            print("❌ No labeled windows with complete pipeline output.")
            return

        grouped = group_by_user_and_session(windows)

        eligible = {}
        excluded = {}
        for uid, sessions in grouped.items():
            total = sum(len(s) for s in sessions)
            if total >= MIN_WINDOWS_PER_USER:
                eligible[uid] = sessions
            else:
                excluded[uid] = total

        print(f"\n📊 Users with ≥ {MIN_WINDOWS_PER_USER} labeled windows: "
              f"{len(eligible)}")
        for uid, sessions in sorted(eligible.items(),
                                     key=lambda kv: -sum(len(s) for s in kv[1])):
            total = sum(len(s) for s in sessions)
            email = sessions[0][0]["user_email"] if sessions and sessions[0] else "?"
            print(f"   user {uid:>3} ({email:<35}) {total:>4} windows, "
                  f"{len(sessions)} session(s)")

        if excluded:
            print(f"\n   Excluded (< {MIN_WINDOWS_PER_USER} windows): "
                  f"{sorted(excluded.keys())}")

        calib = compute_user_calibration(eligible)
        print(f"\n🎚️  Per-user severity calibration (Fix A):")
        for uid, (median, spread) in sorted(calib.items()):
            print(f"   user {uid:>3}  median={median:.3f}  spread={spread:.3f}")

        resources = load_resources()
        print(f"\n🎯 Resources evaluated ({len(resources)}):")
        for name, thr in sorted(resources.items(), key=lambda kv: kv[1]):
            print(f"   {name:<20}  τ = {thr:.2f}")

        n_sessions_total = sum(len(s) for s in eligible.values())
        print(f"\n📅 Sessions across eligible users (Fix C): {n_sessions_total}")
        print(f"\n🧱 CBAC trust floors (Fix D):")
        for cls, floor in TRUST_FLOOR.items():
            print(f"   {cls:<10} floor = {floor:.2f}")

        results = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "version": "v3_calibrated_session_reset_class_floor",
            "eligible_users": len(eligible),
            "excluded_users": sorted(excluded.keys()),
            "min_windows_per_user": MIN_WINDOWS_PER_USER,
            "n_sessions_total": n_sessions_total,
            "ttdrm": {
                "lambda": TTDRM_LAMBDA,
                "gamma": TTDRM_GAMMA,
                "T0": TTDRM_T0,
                "session_reset": True,
                "class_floors": TRUST_FLOOR,
            },
            "calibration": {
                "slope": CALIBRATION_SLOPE,
                "anchors": {str(uid): {"median": v[0], "spread": v[1]}
                            for uid, v in calib.items()},
            },
            "resources": resources,
            "configs": {},
        }

        modes = ["p1", "p2", "p3", "p4", "p5"]
        labels = {
            "p1": "P1_naive_anomaly",
            "p2": "P2_svm_only",
            "p3": "P3_cbac_only",
            "p4": "P4_full_pipeline",
            "p5": "P5_full_no_ttdrm",
        }

        for mode in modes:
            results["configs"][labels[mode]] = {}
            for res_name, thr in resources.items():
                stats = evaluate_config(eligible, calib, mode, thr)
                results["configs"][labels[mode]][res_name] = stats

        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Saved to {RESULTS_PATH}")

        for res_name in ["Company Dashboard", "HR Portal", "Admin Console"]:
            if res_name not in resources:
                continue
            print(f"\n📋 Summary — {res_name} (τ={resources[res_name]:.2f})")
            print(f"{'Config':<22} {'UAA':>7} {'Detect':>7} "
                  f"{'F-STEPUP':>10} {'F-DENY':>8} {'T-Restrict':>12}")
            for mode in modes:
                s = results["configs"][labels[mode]].get(res_name)
                if s is None:
                    continue
                print(
                    f"{labels[mode]:<22} "
                    f"{pct(s['uaa_stepup_deny']):>7} "
                    f"{pct(s['detection_rate']):>7} "
                    f"{pct(s['false_stepup_rate']):>10} "
                    f"{pct(s['false_deny_rate']):>8} "
                    f"{s['mean_time_to_restrict_windows']:>10.1f}w"
                )


if __name__ == "__main__":
    main()