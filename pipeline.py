# pipeline.py
# Βοηθητικές συναρτήσεις για preprocess, φόρτωμα μοντέλων, inference (ένα/πολλά),
# σύνθεση alerts/recs, timelines και debouncing.

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

# ==============================
# 1) PREPROCESS (ίδιο με training)
# ==============================

def preprocess_for_inference(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Φτιάχνω ΑΚΡΙΒΩΣ τα features που είχες στο training:
    - hour, minute, sin_time, cos_time (από sec_of_day ή timestamp)
    - rolling mean/std 5' (300 δείγματα, 1Hz)
    - lags 60s & 300s για HR/EDA/TEMP
    - drops 60s: HR_drop, TEMP_drop
    - interactions: HRxTemp (=HR*Temperature), EDAxHum (=EDA*Humidity)
    - πετάω γραμμές με NaN λόγω rolling/lag/diff (ίδια λογική με training)
    """
    df = df_raw.copy()

    # Έλεγξε ότι έχεις χρόνο + βασικά σήματα
    need_time = ('sec_of_day' in df.columns) or ('timestamp' in df.columns)
    if not need_time:
        raise ValueError("Χρειάζομαι 'timestamp' ή 'sec_of_day'.")
    for col in ['HR', 'EDA', 'TEMP', 'Temperature', 'Humidity']:
        if col not in df.columns:
            raise ValueError(f"Λείπει στήλη: {col}")

    # Αν δεν έχεις sec_of_day, το υπολογίζω από timestamp
    if 'sec_of_day' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['sec_of_day'] = (
            df['timestamp'].dt.hour * 3600
            + df['timestamp'].dt.minute * 60
            + df['timestamp'].dt.second
        )

    # Κυκλικά χρονικά features
    df['hour']     = df['sec_of_day'] // 3600
    df['minute']   = (df['sec_of_day'] % 3600) // 60
    df['sin_time'] = np.sin(2 * np.pi * df['sec_of_day'] / 86400)
    df['cos_time'] = np.cos(2 * np.pi * df['sec_of_day'] / 86400)

    # Rolling 5' (300 δείγματα, 1Hz)
    for var in ['HR', 'EDA', 'TEMP']:
        df[f'{var}_rollmean'] = df[var].rolling(window=300, min_periods=1).mean()
        df[f'{var}_rollstd']  = df[var].rolling(window=300, min_periods=1).std()

    # Lags 60s & 300s
    for lag in [60, 300]:
        for var in ['HR', 'EDA', 'TEMP']:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)

    # Drops 60s (χρησιμοποιήθηκαν σε κάποια μοντέλα σου)
    df['HR_drop']   = df['HR'].diff(60)
    df['TEMP_drop'] = df['TEMP'].diff(60)

    # Interactions
    df['HRxTemp'] = df['HR'] * df['Temperature']
    df['EDAxHum'] = df['EDA'] * df['Humidity']

    # Πετάω γραμμές με NaN λόγω rolling/lag/diff (ίδια λογική με training)
    lag_roll_cols = [c for c in df.columns if ('_lag' in c) or ('_roll' in c)]
    lag_roll_cols += ['HR_drop', 'TEMP_drop']
    df = df.dropna(subset=lag_roll_cols).reset_index(drop=True)

    return df


# ==========================================
# 2) LOAD MODEL + ALIGN FEATURES (helpers)
# ==========================================

def load_model_and_features(model_path: str, features_path: str):
    """
    Φορτώνω το εκπαιδευμένο μοντέλο (.pkl) και τη λίστα ονομάτων features (.pkl).
    Επιστρέφω (model, feature_list).
    """
    model = joblib.load(model_path)
    feature_list = joblib.load(features_path)
    if not isinstance(feature_list, (list, tuple)):
        raise TypeError("Το features_pkl δεν είναι λίστα/tuple.")
    return model, list(feature_list)

def align_features(df_engineered: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """
    Ευθυγραμμίζω τα engineered δεδομένα με τα features που περιμένει το μοντέλο:
    - Αν λείπει κάποια στήλη, τη δημιουργώ με 0.0 (ασφαλής default).
    - Κρατώ ΜΟΝΟ τις στήλες του feature_list και με την ίδια σειρά.
    - Μετατρέπω σε αριθμητικούς τύπους όπου χρειάζεται.
    """
    X = df_engineered.copy()

    for col in feature_list:
        if col not in X.columns:
            X[col] = 0.0

    X = X[feature_list]

    for c in X.columns:
        if X[c].dtype == 'bool':
            X[c] = X[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)

    return X


# ==========================================
# 3) INFERENCE (ένα μοντέλο & πολλά μοντέλα)
# ==========================================

def predict_single_model(
    df_engineered: pd.DataFrame,
    model_path: str,
    features_path: str,
    threshold: float = 0.5,
    return_frame: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Κάνω inference για ΕΝΑ μοντέλο πάνω σε ΗΔΗ προεπεξεργασμένα δεδομένα (df_engineered).
    - Φορτώνω μοντέλο + feature list, ευθυγραμμίζω στήλες.
    - Αν υπάρχει predict_proba → επιστρέφω proba + pred με threshold.
    - Αλλιώς → επιστρέφω μόνο pred.
    """
    model, feature_list = load_model_and_features(model_path, features_path)
    X = align_features(df_engineered, feature_list)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        y_pred  = (y_proba >= threshold).astype(int)
    else:
        y_proba = None
        y_pred  = model.predict(X).astype(int)

    out_df = None
    if return_frame:
        out_df = pd.DataFrame(index=df_engineered.index)
        if y_proba is not None:
            out_df["prob"] = y_proba
        out_df["pred"] = y_pred

    return y_pred, y_proba, out_df


def predict_all_models(
    df_engineered: pd.DataFrame,
    model_specs: List[Dict[str, Any]],
    default_threshold: float = 0.5,
    keep_proba: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Τρέχω ΠΟΛΛΑ μοντέλα (NOW + FUTURE) πάνω σε ΗΔΗ προεπεξεργασμένα δεδομένα.
    - Περιμένω model_specs με κλειδιά: name, model_path, features_path, (προαιρετικά threshold, kind)
    - Επιστρέφω DataFrame με <name>_pred (+ <name>_prob αν υπάρχει) και meta dict.
    """
    out_df = pd.DataFrame(index=df_engineered.index)
    meta: Dict[str, Dict[str, Any]] = {}

    for spec in model_specs:
        name = spec["name"]
        model_path = spec["model_path"]
        features_path = spec["features_path"]
        threshold = spec.get("threshold", default_threshold)
        kind = spec.get("kind", "")

        if verbose:
            print(f"🔮 Τρέχω: {name} (thr={threshold})")

        try:
            y_pred, y_proba, _ = predict_single_model(
                df_engineered=df_engineered,
                model_path=model_path,
                features_path=features_path,
                threshold=threshold,
                return_frame=False
            )

            out_df[f"{name}_pred"] = y_pred.astype(int)
            if keep_proba and (y_proba is not None):
                out_df[f"{name}_prob"] = y_proba.astype(float)

            meta[name] = {
                "kind": kind,
                "threshold": threshold,
                "model_path": model_path,
                "features_path": features_path,
                "has_proba": y_proba is not None,
                "n_positives": int(np.sum(y_pred)),
                "positives_pct": float(np.mean(y_pred))
            }

        except Exception as e:
            if verbose:
                print(f"⚠️ Σφάλμα στο '{name}': {e}")
            out_df[f"{name}_pred"] = 0
            meta[name] = {
                "kind": kind,
                "threshold": threshold,
                "model_path": model_path,
                "features_path": features_path,
                "has_proba": False,
                "error": str(e)
            }

    return out_df, meta


# ==========================================
# 4) ALERTS & RECOMMENDATIONS
# ==========================================

# Σοβαρότητες ανά event (προσαρμόζεις αν θες)
EVENT_SEVERITY = {
    "stress":       2,  # MED
    "fatigue":      2,  # MED
    "dehydration":  2,  # MED
    "envrisk":      2,  # MED
    "arrhythmia":   3,  # HIGH
    "hypothermia":  3,  # HIGH
    "syncope":      4,  # CRITICAL (αν το προσθέσεις)
}
SEVERITY_LABEL = {0: "NONE", 1: "LOW", 2: "MED", 3: "HIGH", 4: "CRITICAL"}

# Συστάσεις (NOW/FUTURE)
EVENT_RECS_NOW = {
    "stress":      "Αναπνοές/διάλειμμα, ενυδάτωση, χαμήλωσε ρυθμό.",
    "fatigue":     "Σύντομη ανάπαυση, ανακατανομή φόρτου, καφεΐνη αν επιτρέπεται.",
    "dehydration": "Ενυδάτωση, ηλεκτρολύτες, παρακολούθηση συγχύσεων.",
    "envrisk":     "Περιορισμός έκθεσης, καταφύγιο, τήρηση πρωτοκόλλου.",
    "arrhythmia":  "Ακινησία, καταγραφή (αν υπάρχει), ειδοποίησε ιατρική ομάδα.",
    "hypothermia": "Θέρμανση, ξηρά ρούχα, ζεστά υγρά όπου επιτρέπεται.",
    "syncope":     "Κάθισε/ύπτια θέση, αξιολόγηση ζωτικών, άμεση ιατρική εκτίμηση.",
}
EVENT_RECS_FUTURE = {
    "stress":      "Πρόληψη: ήπια αναπνοή, ρύθμισε ρυθμό, προληπτική ενυδάτωση.",
    "fatigue":     "Πρόληψη: μικρό διάλειμμα/rotation, προετοιμασία καφεΐνης.",
    "dehydration": "Πρόληψη: υγρά/ηλεκτρολύτες πριν τα συμπτώματα.",
    "envrisk":     "Πρόληψη: σκιά/καταφύγιο, ενημέρωσε την ομάδα.",
    "arrhythmia":  "Πρόληψη: μείωσε έντονη δραστηριότητα, εξοπλισμός standby.",
    "hypothermia": "Πρόληψη: πρόσθετη μόνωση/ρούχα, θερμαντικά μέτρα.",
    "syncope":     "Πρόληψη: στήριξη, απόφυγε απότομες μεταβάσεις στάσης.",
}

def _event_key_from_name(model_name: str) -> str:
    """
    Εξάγω βασικό event από το 'name' του μοντέλου (π.χ. 'stress_future' → 'stress').
    """
    for key in EVENT_SEVERITY.keys():
        if model_name.startswith(key):
            return key
    return "unknown"

def compose_alerts_and_recs(pred_df: pd.DataFrame, model_specs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Συνθέτω ανά γραμμή:
    - alert_score: μέγιστη σοβαρότητα από τα ενεργά events
    - alert_level: ετικέτα σοβαρότητας
    - recommendations: ενιαίο κείμενο με συστάσεις (NOW/FUTURE)
    Βασίζομαι στις στήλες <name>_pred (+ προαιρετικά <name>_prob) και στο kind (NOW/FUTURE).
    """
    out = pred_df.copy()
    name_to_kind = {spec["name"]: spec.get("kind", "") for spec in model_specs}

    alert_scores: List[int] = []
    alert_levels: List[str] = []
    rec_texts: List[str] = []

    for _, row in out.iterrows():
        active_sev = 0
        recs: List[str] = []

        for spec in model_specs:
            name = spec["name"]
            kind = name_to_kind.get(name, "")
            pred_col = f"{name}_pred"
            if pred_col in row.index and int(row[pred_col]) == 1:
                ev = _event_key_from_name(name)
                sev = EVENT_SEVERITY.get(ev, 1)
                active_sev = max(active_sev, sev)

                # Επιλογή κειμένου συστάσεων
                tip = EVENT_RECS_NOW.get(ev, "") if kind == "NOW" else EVENT_RECS_FUTURE.get(ev, "")
                tag = "ΤΩΡΑ" if kind == "NOW" else "ΠΡΟΒΛΕΨΗ 5′"
                if tip:
                    recs.append(f"[{tag} • {ev}] {tip}")

        alert_scores.append(active_sev)
        alert_levels.append(SEVERITY_LABEL.get(active_sev, "NONE"))
        rec_texts.append(" | ".join(recs) if recs else "Κανονική παρακολούθηση.")

    out["alert_score"] = alert_scores
    out["alert_level"] = alert_levels
    out["recommendations"] = rec_texts
    return out


# ==========================================
# 5) TIMELINE (rolling) & DEBOUNCING
# ==========================================

def make_alert_timeline(
    pred_df: pd.DataFrame,
    model_name: str,
    timestamps: Optional[pd.Series] = None,
    window_seconds: int = 300
) -> pd.DataFrame:
    """
    Υπολογίζω κυλιόμενο άθροισμα alerts για ένα μοντέλο (π.χ. 'stress_future') σε παράθυρο N δευτερολέπτων.
    - Με timestamps: resample σε 1s και rolling 'Ns' (πιο σωστή χρονοκλίμακα).
    - Χωρίς timestamps: υποθέτω 1Hz και rolling με window=N δείγματα.
    Επιστρέφω:
      - με timestamps: ['ts', '<name>_pred', 'rolling_count']
      - χωρίς timestamps: ['<name>_pred', 'rolling_count']
    """
    pred_col = f"{model_name}_pred"
    if pred_col not in pred_df.columns:
        raise KeyError(f"Δεν βρήκα στήλη '{pred_col}' στο pred_df.")

    s = pred_df[pred_col].astype(int)

    if timestamps is not None:
        if len(timestamps) != len(s):
            raise ValueError("Το μήκος του timestamps δεν ταιριάζει με το pred_df.")
        ts = pd.to_datetime(timestamps, errors='coerce')

        tmp = pd.DataFrame({pred_col: s.values}, index=ts).sort_index()

        # Resample σε 1s (πεζό 's' για αποφυγή FutureWarning)
        tmp_1s = tmp.resample("1s").max().fillna(0).astype(int)

        # Rolling άθροισμα με παράθυρο χρόνου
        window = pd.Timedelta(seconds=window_seconds)
        roll_df = tmp_1s.rolling(window, min_periods=1).sum()

        out = pd.DataFrame({
            "ts": tmp_1s.index,
            pred_col: tmp_1s[pred_col].values,
            "rolling_count": roll_df[pred_col].values
        })
        return out[["ts", pred_col, "rolling_count"]]

    # Χωρίς timestamps → 1Hz index
    rolling_count = s.rolling(window=window_seconds, min_periods=1).sum()
    out = pd.DataFrame({pred_col: s.values, "rolling_count": rolling_count.values})
    return out


def apply_alert_rule(
    timeline_df: pd.DataFrame,
    model_name: str,
    count_threshold: int = 3,
    min_persist_seconds: int = 5
) -> pd.DataFrame:
    """
    Εφαρμόζω κανόνα ενεργοποίησης (debouncing) στο rolling timeline:
    - active_raw: 1 όταν rolling_count ≥ count_threshold
    - active_debounced: 1 όταν κρατά τουλ. min_persist_seconds συνεχόμενα
    - edge_on: 1 στο δευτερόλεπτο που «ανάβει» το debounced alert (rising edge)
    """
    pred_col = f"{model_name}_pred"
    if "rolling_count" not in timeline_df.columns:
        raise KeyError("Λείπει η στήλη 'rolling_count' στο timeline_df.")
    if pred_col not in timeline_df.columns:
        raise KeyError(f"Λείπει η στήλη '{pred_col}' στο timeline_df.")

    df = timeline_df.copy()
    df["active_raw"] = (df["rolling_count"] >= count_threshold).astype(int)

    run_sum = df["active_raw"].rolling(window=min_persist_seconds, min_periods=1).sum()
    df["active_debounced"] = (run_sum >= min_persist_seconds).astype(int)

    prev = df["active_debounced"].shift(1, fill_value=0)
    df["edge_on"] = ((df["active_debounced"] == 1) & (prev == 0)).astype(int)

    keep_cols: List[str] = []
    if "ts" in df.columns:
        keep_cols.append("ts")
    keep_cols += [pred_col, "rolling_count", "active_raw", "active_debounced", "edge_on"]
    return df[keep_cols]


# ==========================================
# 6) (Προαιρετικό) Συνοπτικά για dashboard
# ==========================================

def summarize_for_dashboard(
    pred_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    model_specs: List[Dict[str, Any]],
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Ετοιμάζω συνοπτικά στοιχεία για dashboard:
    - counts & rates ανά event και ανά τύπο (NOW/FUTURE)
    - σύνοψη ανά alert_level
    - τα τελευταία N ενεργά alerts (με timestamp αν υπάρχει)
    """
    name_to_kind = {spec["name"]: spec.get("kind", "") for spec in model_specs}

    # events summary
    events_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for spec in model_specs:
        name = spec["name"]
        kind = name_to_kind.get(name, "")
        ev   = _event_key_from_name(name)
        col  = f"{name}_pred"
        if col in pred_df.columns:
            cnt = int(pred_df[col].sum())
            rate = float(pred_df[col].mean())
            events_summary.setdefault(ev, {})
            events_summary[ev][kind] = {"count": cnt, "rate": rate}

    # levels summary
    level_counts = alerts_df["alert_level"].value_counts(dropna=False).to_dict()

    # recent active
    active = alerts_df[alerts_df["alert_score"] > 0].copy()
    if "timestamp" in active.columns:
        active = active.sort_values("timestamp")
        recent_cols = ["timestamp", "alert_level", "recommendations"]
    else:
        recent_cols = ["alert_level", "recommendations"]
    recent = active.tail(top_n)[recent_cols]

    return {"events": events_summary, "levels": level_counts, "recent": recent}
