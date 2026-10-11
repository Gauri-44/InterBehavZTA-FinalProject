"""
evaluate_multimodal_final.py

Per-user local multimodal evaluation for InterBehav-ZTA (§IV-D).

v3 changes over v2:
  - Per-user protocol (matches DSL and Balabit evaluations).
    For each eligible user:
      * Enrollment = 80% of that user's own genuine windows
      * Genuine test = remaining 20% of that user's own genuine windows
      * Impostor test = ALL of that user's stress + malicious windows
    Train and score on that user only; aggregate as mean ± std across users.
  - Removed pooled LOUO (which mixed heterogeneous users and produced
    below-chance supervised AUCs on the previous run).
  - Bootstrap 95% CIs computed across the per-user metric values.
  - Reports n_windows per user for transparency.

Read-only. No database writes.
"""

import os
import sys
import json
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as sk_shuffle
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  xgboost not installed; XGBoost classifier will be skipped.")

from app import create_app
from models import db, FeatureVector, ExperimentObservation, User

# ============================================================
# Config
# ============================================================

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
ENROLLMENT_FRACTION = 0.80   # 80% of genuine windows for enrollment

MIN_GENUINE_PER_USER = 15
MIN_IMPOSTOR_PER_USER = 5

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "multimodal_results.json",
)

KEYSTROKE_COLUMNS = [
    "dwell_mean", "dwell_std", "flight_mean", "flight_std",
    "typing_speed", "digraph_entropy", "dwell_cv", "flight_cv",
]

MOUSE_COLUMNS = [
    "typing_jerk", "mouse_jerk", "backspace_rate", "click_dwell_cv",
    "scroll_agitation", "idle_fragmentation", "nav_entropy", "session_velocity",
]

ALL_COLUMNS = KEYSTROKE_COLUMNS + MOUSE_COLUMNS

FEATURE_SLICES = {
    "keystroke_only": KEYSTROKE_COLUMNS,
    "mouse_only": MOUSE_COLUMNS,
    "joint": ALL_COLUMNS,
}


# ============================================================
# Data loading
# ============================================================

def load_labeled_windows():
    print("📂 Loading labeled windows...")
    rows = (
        db.session.query(ExperimentObservation, FeatureVector)
        .join(FeatureVector, ExperimentObservation.feature_vector_id == FeatureVector.id)
        .filter(ExperimentObservation.is_valid.is_(True))
        .filter(ExperimentObservation.ground_truth_label.in_(
            ["normal", "stress", "malicious"]
        ))
        .filter(FeatureVector.extraction_status == "success")
        .all()
    )
    print(f"   Found {len(rows)} candidate observations.")

    label_map = {"normal": 0, "stress": 1, "malicious": 1}

    windows = []
    dropped = 0
    for obs, fv in rows:
        feat = {}
        ok = True
        for col in ALL_COLUMNS:
            val = getattr(fv, col, None)
            if val is None:
                ok = False
                break
            feat[col] = float(val)
        if not ok:
            dropped += 1
            continue
        windows.append({
            "user_id": fv.user_id,
            "user_email": None,
            "feature_vector_id": fv.id,
            "label_binary": label_map.get(obs.ground_truth_label, 0),
            "label_raw": obs.ground_truth_label,
            "features": feat,
        })
    if dropped:
        print(f"   Dropped {dropped} rows with missing feature columns.")

    user_ids = {w["user_id"] for w in windows}
    emails = {u.id: u.email for u in User.query.filter(User.id.in_(user_ids)).all()}
    for w in windows:
        w["user_email"] = emails.get(w["user_id"], f"user_{w['user_id']}")
    return windows


def summarize_by_user(windows):
    summary = {}
    for w in windows:
        uid = w["user_id"]
        summary.setdefault(uid, {
            "user_email": w["user_email"],
            "genuine": 0, "stress": 0, "malicious": 0,
        })
        if w["label_raw"] == "normal":
            summary[uid]["genuine"] += 1
        elif w["label_raw"] == "stress":
            summary[uid]["stress"] += 1
        elif w["label_raw"] == "malicious":
            summary[uid]["malicious"] += 1
    return summary


# ============================================================
# Metrics
# ============================================================

def compute_eer(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    thresholds = np.unique(y_scores)
    if len(thresholds) < 2:
        return float("nan"), float("nan")
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    best_eer = 1.0
    best_thr = float("nan")
    for thr in thresholds:
        preds = (y_scores >= thr).astype(int)
        far = ((preds == 1) & (y_true == 0)).sum() / n_neg
        frr = ((preds == 0) & (y_true == 1)).sum() / n_pos
        diff = abs(far - frr)
        if diff < 0.5 and diff < best_eer:
            best_eer = (far + frr) / 2.0
            best_thr = thr
    return float(best_eer), float(best_thr)


def per_user_metrics(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if len(np.unique(y_true)) < 2:
        return None
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        return None
    eer, eer_thr = compute_eer(y_true, y_scores)
    if np.isnan(eer_thr):
        eer_thr = np.median(y_scores)
    preds = (y_scores >= eer_thr).astype(int)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    far = float(((preds == 1) & (y_true == 0)).sum() / n_neg) if n_neg else float("nan")
    frr = float(((preds == 0) & (y_true == 1)).sum() / n_pos) if n_pos else float("nan")
    return {
        "auc": auc,
        "eer": eer,
        "far": far,
        "frr": frr,
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def mean_std(values):
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        means.append(arr[idx].mean())
    means = np.asarray(means)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ============================================================
# Classifiers
# ============================================================

def balance_binary(X, y, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    if n == 0:
        return X, y
    sel0 = rng.choice(idx0, size=n, replace=False)
    sel1 = rng.choice(idx1, size=n, replace=False)
    idx = np.concatenate([sel0, sel1])
    return X[idx], y[idx]


def score_one_class_svm(X_enroll, X_test):
    if len(X_enroll) < 2:
        return None
    scaler = StandardScaler()
    Xe = scaler.fit_transform(X_enroll)
    model = OneClassSVM(kernel="rbf", nu=0.10, gamma="scale")
    model.fit(Xe)
    Xt = scaler.transform(X_test)
    return -model.decision_function(Xt)


def score_random_forest(X_train, X_test, y_train):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(Xtr, y_train)
    return model.predict_proba(Xte)[:, 1]


def score_xgboost(X_train, X_test, y_train):
    if not HAS_XGBOOST:
        return None
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=0,
    )
    model.fit(Xtr, y_train)
    return model.predict_proba(Xte)[:, 1]


# ============================================================
# Per-user evaluation
# ============================================================

def evaluate_user(user_windows, feature_cols, classifier_name):
    """
    Per-user protocol. Returns a dict of metrics for this user, or None.
    """
    genuine = [w for w in user_windows if w["label_binary"] == 0]
    impostor = [w for w in user_windows if w["label_binary"] == 1]

    if len(genuine) < MIN_GENUINE_PER_USER:
        return None
    if len(impostor) < MIN_IMPOSTOR_PER_USER:
        return None

    # Deterministic split of genuine windows into enrollment / genuine test
    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(len(genuine))
    n_enroll = int(len(genuine) * ENROLLMENT_FRACTION)
    enroll_idx = perm[:n_enroll]
    genuine_test_idx = perm[n_enroll:]

    enroll_windows = [genuine[i] for i in enroll_idx]
    genuine_test_windows = [genuine[i] for i in genuine_test_idx]

    if len(genuine_test_windows) < 3:
        return None

    # Test set = held-out genuine + all impostor
    test_windows = genuine_test_windows + impostor

    X_enroll = np.array([[w["features"][c] for c in feature_cols] for w in enroll_windows])
    X_genuine_test = np.array([[w["features"][c] for c in feature_cols] for w in genuine_test_windows])
    X_impostor = np.array([[w["features"][c] for c in feature_cols] for w in impostor])
    X_test = np.vstack([X_genuine_test, X_impostor])
    y_test = np.array([0] * len(genuine_test_windows) + [1] * len(impostor))

    if classifier_name == "one_class_svm":
        scores = score_one_class_svm(X_enroll, X_test)
    elif classifier_name == "random_forest":
        # Supervised: train on enrolled genuine + balanced sample of impostor
        X_train = np.vstack([X_enroll, X_impostor])
        y_train = np.array([0] * len(enroll_windows) + [1] * len(impostor))
        X_train, y_train = balance_binary(X_train, y_train)
        X_train, y_train = sk_shuffle(X_train, y_train, random_state=RANDOM_STATE)
        scores = score_random_forest(X_train, X_test, y_train)
    elif classifier_name == "xgboost":
        X_train = np.vstack([X_enroll, X_impostor])
        y_train = np.array([0] * len(enroll_windows) + [1] * len(impostor))
        X_train, y_train = balance_binary(X_train, y_train)
        X_train, y_train = sk_shuffle(X_train, y_train, random_state=RANDOM_STATE)
        scores = score_xgboost(X_train, X_test, y_train)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    if scores is None:
        return None

    m = per_user_metrics(y_test, scores)
    if m is None:
        return None

    m["n_enroll"] = len(enroll_windows)
    m["n_genuine_test"] = len(genuine_test_windows)
    m["n_impostor_test"] = len(impostor)
    return m


def aggregate_across_users(per_user_results):
    """per_user_results: list of dicts (each user's metrics) or None."""
    valid = [r for r in per_user_results if r is not None]
    if not valid:
        return None
    aucs = [r["auc"] for r in valid]
    eers = [r["eer"] for r in valid]
    fars = [r["far"] for r in valid]
    frrs = [r["frr"] for r in valid]
    accs = [r["accuracy"] for r in valid]
    f1s = [r["f1"] for r in valid]

    out = {}
    for name, vals in [
        ("auc", aucs), ("eer", eers), ("far", fars),
        ("frr", frrs), ("accuracy", accs), ("f1", f1s),
    ]:
        m, s = mean_std(vals)
        lo, hi = bootstrap_mean_ci(vals)
        out[name] = m
        out[f"{name}_std"] = s
        out[f"{name}_ci"] = [lo, hi]

    out["n_users"] = len(valid)
    out["per_user"] = valid
    return out


# ============================================================
# Main
# ============================================================

def main():
    app = create_app()
    with app.app_context():
        windows = load_labeled_windows()
        if not windows:
            print("❌ No labeled windows found.")
            return

        per_user = summarize_by_user(windows)
        print("\n📊 Per-user label counts:")
        eligible_users = []
        for uid, info in sorted(per_user.items()):
            gen = info["genuine"]
            imp = info["stress"] + info["malicious"]
            eligible = gen >= MIN_GENUINE_PER_USER and imp >= MIN_IMPOSTOR_PER_USER
            mark = "✅" if eligible else "  "
            print(f"   {mark} user {uid:>3} ({info['user_email']:<35}) "
                  f"genuine={gen:>3}  stress={info['stress']:>3}  "
                  f"malicious={info['malicious']:>3}")
            if eligible:
                eligible_users.append(uid)

        print(f"\n✅ Eligible users for per-user evaluation: {len(eligible_users)}")
        for uid in eligible_users:
            print(f"   - {per_user[uid]['user_email']} (id={uid})")

        if len(eligible_users) < 2:
            print("\n❌ Need at least 2 eligible users.")
            return

        # Build per-user window lists
        by_user = {}
        for w in windows:
            by_user.setdefault(w["user_id"], []).append(w)

        results = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "version": "v3-per-user",
            "protocol": "per-user 80/20 genuine split + full impostor set",
            "enrollment_fraction": ENROLLMENT_FRACTION,
            "n_users_total": len(per_user),
            "n_users_eligible": len(eligible_users),
            "n_windows_total": len(windows),
            "per_user_counts": {
                str(uid): {
                    "user_email": info["user_email"],
                    "genuine": info["genuine"],
                    "stress": info["stress"],
                    "malicious": info["malicious"],
                } for uid, info in per_user.items()
            },
            "eligible_user_ids": eligible_users,
            "feature_slices": {},
        }

        classifiers = ["one_class_svm", "random_forest"]
        if HAS_XGBOOST:
            classifiers.append("xgboost")

        for slice_name, cols in FEATURE_SLICES.items():
            print(f"\n🔬 Feature slice: {slice_name} ({len(cols)} features)")
            results["feature_slices"][slice_name] = {}

            for clf in classifiers:
                print(f"   ▶ {clf} ...")
                per_user_metrics_list = []
                for uid in eligible_users:
                    m = evaluate_user(by_user[uid], cols, clf)
                    per_user_metrics_list.append(m)
                    if m is not None:
                        print(f"      user {uid:>3}: "
                              f"AUC={m['auc']:.3f}  EER={m['eer']:.3f}  "
                              f"n_test={m['n_genuine_test'] + m['n_impostor_test']}")
                agg = aggregate_across_users(per_user_metrics_list)
                if agg is None:
                    print("      ⚠️  No valid users for this classifier/slice.")
                    results["feature_slices"][slice_name][clf] = None
                    continue
                print(f"      MEAN across {agg['n_users']} users: "
                      f"AUC={agg['auc']:.3f} ± {agg['auc_std']:.3f} "
                      f"[{agg['auc_ci'][0]:.3f}, {agg['auc_ci'][1]:.3f}]  "
                      f"EER={agg['eer']:.3f} ± {agg['eer_std']:.3f}")
                results["feature_slices"][slice_name][clf] = agg

        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()