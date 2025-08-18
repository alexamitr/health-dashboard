# app.py — Multi-Model Health Dashboard (NOW + FUTURE)
# Τρέχω από τερματικό: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
from io import StringIO
#  HTTP live:
try:
    import requests
except Exception:
    requests = None  


# Φέρνω τις συναρτήσεις από το pipeline.py (πρέπει να είναι στον ίδιο φάκελο)
from pipeline import (
    preprocess_for_inference,
    predict_all_models,
    compose_alerts_and_recs,
    make_alert_timeline,
    apply_alert_rule,
    summarize_for_dashboard,
)

# ==========================
# Ρύθμιση σελίδας / Τίτλος
# ==========================
st.set_page_config(page_title="Health Forecast Dashboard", layout="wide")
st.title("🛰️ Health Forecast Dashboard — NOW & 5′ FUTURE")

st.caption("Φόρτωσε CSV με raw μετρήσεις → preprocess (ίδιο με training) → τρέχω όλα τα μοντέλα → KPIs, γραφήματα, alerts & προτάσεις.")

# ==========================================================
# DEFAULT MODEL SPECS (paths/thresholds — προσαρμόζεις αν θες)
# ==========================================================
DEFAULT_MODEL_SPECS = [
    # FUTURE (5')
    {"name": "stress_future",      "model_path": "model_stress_event_future.pkl",
     "features_path": "features_model_stress_event_future.pkl", "threshold": 0.50, "kind": "FUTURE"},
    {"name": "fatigue_future",     "model_path": "model_fatigue_event_future.pkl",
     "features_path": "features_model_fatigue_event_future.pkl", "threshold": 0.50, "kind": "FUTURE"},
    {"name": "dehydration_future", "model_path": "model_dehydration_event_future.pkl",
     "features_path": "features_model_dehydration_event_future.pkl", "threshold": 0.50, "kind": "FUTURE"},
    {"name": "arrhythmia_future",  "model_path": "model_arrhythmia_event_future.pkl",
     "features_path": "features_model_arrhythmia_event_future.pkl", "threshold": 0.45, "kind": "FUTURE"},
    {"name": "hypothermia_future", "model_path": "model_hypothermia_event_future.pkl",
     "features_path": "features_model_hypothermia_event_future.pkl", "threshold": 0.45, "kind": "FUTURE"},
    {"name": "envrisk_future",     "model_path": "model_env_risk_event_future.pkl",
     "features_path": "features_model_env_risk_event_future.pkl", "threshold": 0.45, "kind": "FUTURE"},

    # NOW
    {"name": "stress_now", "model_path": "model_stress_event_now.pkl",
     "features_path": "features_stress_event_now.pkl", "threshold": 0.50, "kind": "NOW"},
    {"name": "envrisk_now", "model_path": "model_env_risk_event_now.pkl",
     "features_path": "features_env_risk_event_now.pkl", "threshold": 0.50, "kind": "NOW"},
    {"name": "arrhythmia_now", "model_path": "model_arrhythmia_event_now.pkl",
     "features_path": "features_arrhythmia_event_now.pkl", "threshold": 0.40, "kind": "NOW"},
    {"name": "hypothermia_now", "model_path": "model_hypothermia_event.pkl",
     "features_path": "features_hypothermia_event_now.pkl", "threshold": 0.40, "kind": "NOW"},
    {"name": "fatigue_now", "model_path": "model_fatigue_event_now.pkl",
     "features_path": "features_fatigue_event_now.pkl", "threshold": 0.50, "kind": "NOW"},
    {"name": "dehydration_now", "model_path": "model_dehydration_event_now.pkl",
     "features_path": "features_model_dehydration_event_now.pkl", "threshold": 0.50, "kind": "NOW"}

]

# ==========================
# SIDEBAR — Ρυθμίσεις / Input
# ==========================
st.sidebar.header("Ρυθμίσεις")

st.sidebar.markdown("---")
live_mode = st.sidebar.toggle("🔴 Live mode", value=False, help="Αυτόματη ανανέωση ανά Ν δευτερόλεπτα")

source_kind = st.sidebar.selectbox(
    "Live source",
    options=["CSV (append)", "HTTP JSON (GET)"],
    index=0,
    help="Διάλεξε πηγή live δεδομένων"
)

refresh_sec = st.sidebar.slider("Refresh (sec)", 1, 30, 5, 1)
buffer_minutes = st.sidebar.slider("Buffer (minutes)", 5, 120, 30, 5, help="Πόσο ιστορικό κρατάω για rolling/lag")


uploaded = st.sidebar.file_uploader(
    "Φόρτωσε CSV (raw μετρήσεις)",
    type=["csv"],
    help="Χρειάζομαι: timestamp ή sec_of_day, και HR, EDA, TEMP, Temperature, Humidity"
)

all_names = [m["name"] for m in DEFAULT_MODEL_SPECS]
selected = st.sidebar.multiselect(
    "Επέλεξε ποια μοντέλα να τρέξω",
    options=all_names,
    default=all_names
)
model_specs = [m for m in DEFAULT_MODEL_SPECS if m["name"] in selected]

thr_offset = st.sidebar.slider(
    "Global threshold offset",
    -0.20, 0.20, 0.00, 0.01,
    help="Προσθέτω/αφαιρώ από τα per-model thresholds (για γρήγορο fine-tuning)"
)

# ==========================
# 1) Δεδομένα (raw)
# ==========================

def read_live_csv(path: str) -> pd.DataFrame:
    """Διαβάζω ΟΛΟ το CSV που ενημερώνεται (append) και επιστρέφω DataFrame."""
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def read_live_http(url: str) -> pd.DataFrame:
    """
    Κάνω GET σε endpoint που γυρνά JSON (λίστα από records) και φτιάχνω DataFrame.
    Περιμένω πεδία: timestamp/sec_of_day, HR, EDA, TEMP, Temperature, Humidity.
    """
    if requests is None:
        raise RuntimeError("Λείπει το package 'requests' (pip install requests).")
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def append_new_rows(buffer: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ενώνω buffer με νέα δεδομένα. Αν έχω timestamp, κρατάω μόνο όσα είναι νεότερα.
    Αν δεν έχω, κάνω απλό concat.
    """
    if buffer is None or buffer.empty:
        return new_df.copy()

    if "timestamp" in new_df.columns and "timestamp" in buffer.columns:
        last_ts = buffer["timestamp"].max()
        fresh = new_df[new_df["timestamp"] > last_ts]
        return pd.concat([buffer, fresh], ignore_index=True)
    else:
        # fallback: σκέτο concat
        return pd.concat([buffer, new_df.iloc[len(buffer):]], ignore_index=True)

def trim_buffer_to_minutes(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Κρατάω μόνο τα τελευταία Χ λεπτά, αν υπάρχει timestamp. Αλλιώς επιστρέφω όπως είναι."""
    if "timestamp" not in df.columns or df.empty:
        return df
    cutoff = df["timestamp"].max() - pd.Timedelta(minutes=minutes)
    return df[df["timestamp"] >= cutoff].reset_index(drop=True)




st.subheader("1) Δεδομένα (raw)")

df_raw = None

if live_mode:
    st.info("Live mode ενεργό — θα ανανεώνω αυτόματα.")
    if source_kind == "CSV (append)":
        live_path = st.sidebar.text_input("CSV path", "live_feed.csv", help="Διαδρομή αρχείου που ενημερώνεται")
        if live_path:
            try:
                # 1) φόρτωσε πλήρη τρέχουσα εικόνα
                new_df = read_live_csv(live_path)
                # 2) ενημέρωσε buffer στο session_state
                buf = st.session_state.get("raw_buffer", pd.DataFrame())
                df_raw = append_new_rows(buf, new_df)
                # 3) κάνε trim στο παράθυρο Χ λεπτών (για rolling/lag)
                df_raw = trim_buffer_to_minutes(df_raw, buffer_minutes)
                st.session_state["raw_buffer"] = df_raw.copy()
            except Exception as e:
                st.error(f"Σφάλμα ανάγνωσης live CSV: {e}")
                st.stop()
        else:
            st.warning("Δώσε έγκυρο CSV path.")
            st.stop()

    else:  # HTTP JSON (GET)
        live_url = st.sidebar.text_input("HTTP URL", "http://127.0.0.1:8000/latest")
        if live_url:
            try:
                new_df = read_live_http(live_url)
                buf = st.session_state.get("raw_buffer", pd.DataFrame())
                df_raw = append_new_rows(buf, new_df)
                df_raw = trim_buffer_to_minutes(df_raw, buffer_minutes)
                st.session_state["raw_buffer"] = df_raw.copy()
            except Exception as e:
                st.error(f"Σφάλμα HTTP: {e}")
                st.stop()
        else:
            st.warning("Δώσε έγκυρο URL.")
            st.stop()

else:
    # classic: upload ενός CSV & στατική ανάλυση (όπως είχες)
    uploaded = st.sidebar.file_uploader(
        "Φόρτωσε CSV (raw μετρήσεις)",
        type=["csv"],
        help="Χρειάζομαι: timestamp ή sec_of_day, HR, EDA, TEMP, Temperature, Humidity"
    )
    if uploaded is None:
        st.info("Φόρτωσε ένα CSV από το sidebar για να συνεχίσω.")
        st.stop()
    try:
        df_raw = pd.read_csv(uploaded)
        if "timestamp" in df_raw.columns:
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
    except Exception as e:
        st.error(f"Σφάλμα ανάγνωσης CSV: {e}")
        st.stop()

# Κοινή προεπισκόπηση
st.write("Δείγμα raw:")
st.dataframe(df_raw.tail(20))  # στο live δείχνω τα τελευταία


# ==========================
# 2) Preprocess (ίδιο με training)
# ==========================
st.subheader("2) Preprocess (ίδιο με training)")

try:
    df_eng = preprocess_for_inference(df_raw)  # ίδια features/ονόματα/NA drop με training
    st.success(f"OK — engineered σχήμα: {df_eng.shape[0]:,} γραμμές × {df_eng.shape[1]:,} στήλες")
    st.dataframe(df_eng.head(20))
except Exception as e:
    st.error(f"Σφάλμα preprocess: {e}")
    st.stop()

# ==========================
# 3) Πρόβλεψη (πολλαπλά μοντέλα)
# ==========================
st.subheader("3) Πρόβλεψη (πολλαπλά μοντέλα)")

# εφαρμόζω offset στα thresholds
specs_adj = []
for spec in model_specs:
    s = spec.copy()
    s["threshold"] = max(0.0, min(1.0, s.get("threshold", 0.5) + thr_offset))
    specs_adj.append(s)

try:
    pred_df, meta = predict_all_models(
        df_engineered=df_eng,
        model_specs=specs_adj,
        default_threshold=0.5,
        keep_proba=True,
        verbose=False
    )
    st.success("OK — ολοκληρώθηκε το inference για τα επιλεγμένα μοντέλα.")
except Exception as e:
    st.error(f"Σφάλμα στο inference: {e}")
    st.stop()

# Γρήγορη σύνοψη: πόσα pred=1 ανά μοντέλο & πιθανά errors
st.write("Σύνοψη θετικών (pred=1) ανά μοντέλο:")
st.dataframe(pred_df.filter(regex="_pred$").sum().to_frame("positives"))

errors = {k: v.get("error") for k, v in meta.items() if v.get("error")}
if errors:
    st.warning(f"Μοντέλα με σφάλμα φόρτωσης: {errors}")

# ==========================
# 4) Alerts & Recommendations
# ==========================
st.subheader("4) Alerts & Recommendations")

alerts_df = compose_alerts_and_recs(pred_df, specs_adj)

# πρόσθεσε timestamp αν υπάρχει στα engineered δεδομένα
if "timestamp" in df_eng.columns:
    alerts_df = alerts_df.copy()
    alerts_df["timestamp"] = df_eng["timestamp"].values

cols_show = ["timestamp", "alert_level", "alert_score", "recommendations"] if "timestamp" in alerts_df.columns \
            else ["alert_level", "alert_score", "recommendations"]
st.dataframe(alerts_df[cols_show].head(20))

# ==========================
# 5) KPIs / Σύνοψη
# ==========================
st.subheader("5) KPIs / Σύνοψη")

total = len(alerts_df)
active = int((alerts_df["alert_score"] > 0).sum())
rate = active / total if total else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Σύνολο δειγμάτων", f"{total:,}")
c2.metric("Ενεργά alerts", f"{active:,}")
c3.metric("Rate", f"{rate:.2%}")

# Κατανομή επιπέδων
level_counts = alerts_df["alert_level"].value_counts().to_dict()
st.write("Κατανομή επιπέδων:", level_counts)

# Σύνοψη ανά event/kind + πρόσφατα ενεργά
summary = summarize_for_dashboard(pred_df, alerts_df, specs_adj, top_n=10)
st.write("Σύνοψη ανά event/kind:", summary["events"])
st.write("Πρόσφατα ενεργά:")
st.dataframe(summary["recent"])

# ==========================
# 6) Γραφήματα (ανά μοντέλο)
# ==========================
st.subheader("6) Γραφήματα")

plot_model = st.selectbox("Διάλεξε μοντέλο για προβολή", options=[m["name"] for m in specs_adj])
prob_col = f"{plot_model}_prob"
pred_col = f"{plot_model}_pred"

# Γράφημα πιθανότητας
if prob_col in pred_df.columns:
    chart_df = pred_df[[prob_col]].copy()
    if "timestamp" in alerts_df.columns:
        chart_df["timestamp"] = alerts_df["timestamp"].values
        st.line_chart(chart_df.set_index("timestamp")[prob_col], height=220)
    else:
        st.line_chart(chart_df[prob_col], height=220)
else:
    st.info("Το επιλεγμένο μοντέλο δεν παρέχει predict_proba.")

# Πίνακας ενεργών για το επιλεγμένο μοντέλο
st.subheader("7) Ενεργά alerts για το επιλεγμένο μοντέλο")
mask = pred_df[pred_col] == 1
table_cols = (["timestamp"] if "timestamp" in alerts_df.columns else []) + [pred_col, "alert_level", "recommendations"]
st.dataframe(alerts_df.loc[mask, table_cols].head(200))

# ==========================
# 7) (Προαιρετικό) Rolling timeline & debouncing
# ==========================
with st.expander("🔧 Προαιρετικά: Rolling timeline & debouncing"):
    if "timestamp" in alerts_df.columns:
        tl = make_alert_timeline(
            pred_df,
            model_name=plot_model,
            timestamps=alerts_df["timestamp"],
            window_seconds=300  # 5'
        )
        st.write("Δείγμα timeline (rolling_count):")
        st.dataframe(tl.head())

        thr_count = st.slider("Ελάχιστα hits στο παράθυρο (rolling_count ≥)", 1, 30, 3, 1)
        persist_s = st.slider("Ελάχιστη διάρκεια (δευτ.) για debouncing", 1, 60, 5, 1)

        tl2 = apply_alert_rule(
            tl, model_name=plot_model,
            count_threshold=thr_count,
            min_persist_seconds=persist_s
        )
        st.write("Debounced timeline (edge_on = στιγμές έναρξης επεισοδίου):")
        st.dataframe(tl2.loc[tl2["edge_on"] == 1].head(30))

        if "ts" in tl2.columns:
            st.line_chart(tl2.set_index("ts")["rolling_count"], height=180)
    else:
        st.info("Δεν υπάρχει timestamp — το timeline με βάση χρόνο είναι απενεργοποιημένο.")

# ==========================
# 8) Εξαγωγή αποτελεσμάτων
# ==========================
st.subheader("8) Εξαγωγή")

export_df = alerts_df.copy()
if "timestamp" in export_df.columns:
    export_df = export_df.sort_values("timestamp")

st.download_button(
    "⬇️ Κατέβασε τα αποτελέσματα (CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="alerts_results.csv",
    mime="text/csv"
)


# Auto-refresh στο live
if live_mode:
    st.caption(f"Auto-refresh σε {refresh_sec}s…")
    time.sleep(refresh_sec)
    st.experimental_rerun()

