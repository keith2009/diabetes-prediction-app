"""
HbA1c Prediction Clinical Decision Support System (CDSS)
app.py — Streamlit front-end

[v2] Changes:
  - Loads MAPIE model dict {'model', 'feature_names'}
  - Displays 90% prediction interval alongside point estimate
  - Expanded inputs: SBP, key medication/comorbidity baseline flags
  - Fixed med_insulin_baseline conflation (now independent input)
  - Input range warnings (FPG, HbA1c, SBP, Age)
  - Permanent disclaimer banner
  - Audit log → prediction_log.csv
"""

import csv
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ระบบพยากรณ์ค่าน้ำตาลสะสม HbA1c",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #f0f4f8; }

/* Disclaimer banner */
.disclaimer-banner {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border-left: 5px solid #f59e0b;
    border-radius: 8px;
    padding: 14px 20px;
    margin-bottom: 20px;
    color: #78350f;
    font-size: 14px;
    font-weight: 500;
}

/* Page header */
.page-header {
    background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 50%, #0d9488 100%);
    padding: 36px 32px 28px;
    border-radius: 16px;
    margin-bottom: 24px;
    color: white;
    text-align: center;
}
.page-header h1 { font-size: 2rem; font-weight: 700; margin: 0 0 8px; }
.page-header p  { font-size: 1rem; opacity: 0.88; margin: 0; }

/* Section cards */
.section-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 16px;
}
.section-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
}

/* Result card */
.result-card {
    background: white;
    border-radius: 16px;
    padding: 28px 32px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin-top: 8px;
}
.result-value {
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
}
.result-range {
    font-size: 1.1rem;
    color: #64748b;
    margin-top: 4px;
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0f766e, #0d9488);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1rem;
    padding: 14px 20px;
    letter-spacing: 0.02em;
    transition: all 0.25s ease;
    cursor: pointer;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0d9488, #14b8a6);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(13,148,136,0.35);
}

/* Range warning pill */
.range-pill {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fcd34d;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 500;
    margin-top: 4px;
}

/* Metric badge */
.metric-row {
    display: flex;
    gap: 16px;
    margin-top: 12px;
    flex-wrap: wrap;
}
.metric-badge {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 16px;
    text-align: center;
    flex: 1;
    min-width: 100px;
}
.metric-badge .label { font-size: 11px; color: #94a3b8; font-weight: 500; }
.metric-badge .value { font-size: 1.1rem; font-weight: 700; color: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load model artifact
# ──────────────────────────────────────────────
@st.cache_resource
def load_model_artifact():
    artifact = joblib.load('diabetes_rf_model.pkl')
    # Support both old (bare model) and new (dict) formats
    if isinstance(artifact, dict):
        return artifact['model'], artifact['feature_names']
    else:
        # Legacy bare model — derive feature names from model
        return artifact, list(artifact.feature_names_in_)

try:
    model, feature_names = load_model_artifact()
    MODEL_SUPPORTS_CI = hasattr(model, 'predict_interval')
    # MAPIE 1.4 uses CrossConformalRegressor
    IS_MAPIE = type(model).__name__ == 'CrossConformalRegressor'
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลดไฟล์โมเดล 'diabetes_rf_model.pkl' ได้: {str(e)}")
    st.stop()

# ──────────────────────────────────────────────
# Median fill-values (for features not in the UI)
# ──────────────────────────────────────────────
MEDIANS = {
    "age": 62.0, "type1": 0.0, "type2": 1.0, "gdm": 0.0, "sex": 0.0, "dm_onset": 1.0,
    "co_atrial_fibrillation_baseline": 0.0, "co_hf_baseline": 0.0, "co_ckd_baseline": 1.0,
    "co_stroke_baseline": 0.0, "co_dementia_baseline": 0.0, "co_ht_baseline": 1.0,
    "co_arrhythmias_baseline": 0.0, "co_cad_baseline": 0.0, "med_gliclazide_baseline": 0.0,
    "med_gemigliptin_baseline": 0.0, "med_pioglitazone_baseline": 0.0, "med_acarbose_baseline": 0.0,
    "med_linagliptin_baseline": 0.0, "med_empagliflozin_baseline": 0.0, "med_insulin_baseline": 0.0,
    "med_glipizide_baseline": 1.0, "med_dulaglutide_baseline": 0.0, "med_semaglutide_baseline": 0.0,
    "med_dapagliflozin_baseline": 0.0, "med_metformin_baseline": 1.0, "med_sitagliptin_baseline": 0.0,
    "med_trelagliptin_baseline": 0.0, "med_liraglutide_baseline": 0.0, "med_glibenclamide_baseline": 0.0,
    "med_glimepiride_baseline": 0.0, "Period": 22.0, "vitalsign_resp": 19.0, "vitalsign_o2sat": 99.0,
    "vitalsign_temp": 36.5, "vitalsign_hr": 82.0, "vitalsign_ht": 157.0, "vitalsign_wt": 64.975,
    "vitalsign_bmi": 25.98, "vitalsign_sbp": 136.0, "vitalsign_dbp": 72.0, "lab_fpg": 124.0,
    "lab_hba1c": 7.0, "lab_wbc": 7.1, "lab_platelet": 260.5, "lab_hemoglobin": 12.4,
    "lab_hematocrit": 38.4, "lab_tg": 120.0, "lab_chol": 156.0, "lab_hdl": 53.0, "lab_ldl": 88.0,
    "lab_uric": 5.9, "lab_t3": 85.92, "lab_t4": 6.885, "lab_probnp": 1290.0, "lab_calcium": 9.3,
    "lab_co2": 25.6, "lab_cl": 102.0, "lab_po4": 3.6, "lab_potassium": 4.4, "lab_sodium": 140.0,
    "lab_uacr": 98.83, "co_arrhythmias": 1.0, "co_atrial_fibrillation": 1.0, "co_cad": 1.0,
    "co_dementia": 1.0, "co_hf": 1.0, "co_stroke": 1.0, "co_ht": 1.0, "co_ckd": 1.0,
    "med_acarbose": 1.0, "med_dapagliflozin": 1.0, "med_dulaglutide": 1.0, "med_empagliflozin": 1.0,
    "med_gemigliptin": 1.0, "med_glibenclamide": 1.0, "med_gliclazide": 1.0, "med_glimepiride": 1.0,
    "med_glipizide": 1.0, "med_insulin": 1.0, "med_linagliptin": 1.0, "med_metformin": 1.0,
    "med_pioglitazone": 1.0, "med_semaglutide": 1.0, "med_sitagliptin": 1.0, "med_trelagliptin": 1.0,
    "identify_by_lab": 0.0, "identify_by_medication": 0.0,
}

# ──────────────────────────────────────────────
# Audit logging
# ──────────────────────────────────────────────
LOG_FILE = 'prediction_log.csv'
LOG_HEADERS = [
    'timestamp', 'age', 'sex', 'hba1c', 'fpg', 'sbp',
    'insulin_current', 'insulin_baseline', 'glipizide_baseline',
    'metformin_baseline', 'dementia_baseline', 'period',
    'predicted_hba1c', 'ci_lower', 'ci_upper',
]

def append_audit_log(row: dict):
    """Append one prediction record to the audit log CSV."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ──────────────────────────────────────────────
# DISCLAIMER BANNER (always visible at top)
# ──────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-banner">
  ⚠️ <strong>คำเตือนสำคัญ:</strong>
  ระบบนี้เป็นเครื่องมือสนับสนุนการตัดสินใจทางคลินิก (CDSS) เท่านั้น
  ไม่ใช่การวินิจฉัยโรคหรือการสั่งการรักษา
  กรุณาพิจารณาผลลัพธ์ร่วมกับดุลยพินิจทางคลินิกและข้อมูลผู้ป่วยในบริบทที่ครบถ้วนเสมอ
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Page header
# ──────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>🩺 ระบบพยากรณ์ค่าน้ำตาลสะสม HbA1c</h1>
  <p>ปัญญาประดิษฐ์สำหรับประเมินระดับ HbA1c ในรอบการนัดถัดไป (≈ 60 วัน) — พร้อมช่วงความเชื่อมั่น 90%</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# INPUT FORM — 3 sections
# ──────────────────────────────────────────────
st.subheader("📋 กรอกข้อมูลผู้ป่วย")

# ── Section 1: Current Labs & Vitals ──────────
st.markdown("<div class='section-title'>🔬 ผลตรวจและสัญญาณชีพวันนี้</div>", unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)

with col_a:
    age = st.number_input(
        "อายุ (ปี)", min_value=1, max_value=110, value=62,
        help="อายุปัจจุบันของผู้ป่วย",
        key="input_age"
    )
    if age < 18:
        st.markdown("<span class='range-pill'>⚠️ โมเดลนี้เทรนด้วยข้อมูลผู้ใหญ่ — ผลอาจไม่แม่นยำ</span>", unsafe_allow_html=True)

    sex_label = st.selectbox("เพศ", options=["ชาย", "หญิง"], index=0, key="input_sex")
    sex = 1 if sex_label == "ชาย" else 0

with col_b:
    hba1c = st.number_input(
        "HbA1c วันนี้ (%)", min_value=3.0, max_value=20.0, value=7.0, step=0.1,
        help="ระดับน้ำตาลสะสมเฉลี่ย 2–3 เดือน",
        key="input_hba1c"
    )
    if hba1c > 14.0:
        st.markdown("<span class='range-pill'>⚠️ HbA1c > 14% — กรุณาตรวจสอบความถูกต้อง</span>", unsafe_allow_html=True)

    fpg = st.number_input(
        "FPG (mg/dL)", min_value=50, max_value=600, value=124, step=1,
        help="ค่าน้ำตาลในเลือดหลังงดอาหาร ≥ 8 ชั่วโมง",
        key="input_fpg"
    )
    if fpg > 400:
        st.markdown("<span class='range-pill'>⚠️ FPG > 400 mg/dL — ค่าสูงผิดปกติ</span>", unsafe_allow_html=True)

with col_c:
    sbp = st.number_input(
        "ความดันโลหิตตัวบน SBP (mmHg)", min_value=60, max_value=260, value=136, step=1,
        help="Systolic Blood Pressure — หนึ่งในปัจจัยสำคัญของโมเดล",
        key="input_sbp"
    )
    if sbp > 200 or sbp < 80:
        st.markdown("<span class='range-pill'>⚠️ SBP อยู่นอกช่วงปกติ</span>", unsafe_allow_html=True)

    period = st.number_input(
        "ระยะเวลาติดตาม (วัน)", min_value=0, max_value=730, value=60,
        help="จำนวนวันที่ผู้ป่วยอยู่ในแผนการติดตามครั้งนี้",
        key="input_period"
    )

st.divider()

# ── Section 2: Current Medications ────────────
st.markdown("<div class='section-title'>💊 ยาที่ใช้อยู่ในปัจจุบัน</div>", unsafe_allow_html=True)
med_col1, med_col2, med_col3, med_col4 = st.columns(4)

with med_col1:
    insulin_current_label = st.selectbox(
        "อินซูลิน (ปัจจุบัน)", ["ไม่ได้ใช้", "ใช้อยู่"], index=0, key="input_insulin_current"
    )
    insulin_current = 1 if insulin_current_label == "ใช้อยู่" else 0

with med_col2:
    glipizide_current_label = st.selectbox(
        "Glipizide (ปัจจุบัน)", ["ไม่ได้ใช้", "ใช้อยู่"], index=1, key="input_glipizide_current"
    )
    glipizide_current = 1 if glipizide_current_label == "ใช้อยู่" else 0

with med_col3:
    metformin_current_label = st.selectbox(
        "Metformin (ปัจจุบัน)", ["ไม่ได้ใช้", "ใช้อยู่"], index=1, key="input_metformin_current"
    )
    metformin_current = 1 if metformin_current_label == "ใช้อยู่" else 0

with med_col4:
    st.markdown("<small style='color:#94a3b8'>ยาอื่นๆ ใช้ค่า median จากข้อมูลเทรน</small>", unsafe_allow_html=True)

st.divider()

# ── Section 3: Baseline (at-diagnosis) History ──
with st.expander("📁 ประวัติ ณ วันวินิจฉัยโรค DM (Baseline History) — คลิกเพื่อขยาย", expanded=False):
    st.markdown(
        "<small style='color:#64748b'>ข้อมูลสถานะของผู้ป่วย <b>ณ วันแรกที่วินิจฉัยเป็นเบาหวาน</b> "
        "— ใช้เป็นปัจจัยสำคัญในโมเดล และแยกจากสถานะปัจจุบัน</small>",
        unsafe_allow_html=True
    )
    bl_col1, bl_col2, bl_col3, bl_col4 = st.columns(4)

    with bl_col1:
        insulin_baseline_label = st.selectbox(
            "อินซูลิน (ณ วินิจฉัย)", ["ไม่ได้ใช้", "ใช้อยู่"], index=0, key="input_insulin_baseline"
        )
        insulin_baseline = 1 if insulin_baseline_label == "ใช้อยู่" else 0

    with bl_col2:
        glipizide_baseline_label = st.selectbox(
            "Glipizide (ณ วินิจฉัย)", ["ไม่ได้ใช้", "ใช้อยู่"], index=1, key="input_glipizide_baseline"
        )
        glipizide_baseline = 1 if glipizide_baseline_label == "ใช้อยู่" else 0

    with bl_col3:
        metformin_baseline_label = st.selectbox(
            "Metformin (ณ วินิจฉัย)", ["ไม่ได้ใช้", "ใช้อยู่"], index=1, key="input_metformin_baseline"
        )
        metformin_baseline = 1 if metformin_baseline_label == "ใช้อยู่" else 0

    with bl_col4:
        dementia_baseline_label = st.selectbox(
            "ภาวะสมองเสื่อม (ณ วินิจฉัย)", ["ไม่มี", "มี"], index=0, key="input_dementia_baseline"
        )
        dementia_baseline = 1 if dementia_baseline_label == "มี" else 0

st.divider()

# ──────────────────────────────────────────────
# PREDICT BUTTON
# ──────────────────────────────────────────────
predict_clicked = st.button(
    "🔮 พยากรณ์ค่า HbA1c ในรอบถัดไป (~60 วัน)",
    type="primary",
    key="predict_btn"
)

if predict_clicked:
    with st.spinner("🧠 AI กำลังประมวลผล..."):

        # Build input row from medians, then override with user inputs
        input_data = pd.DataFrame([{col: MEDIANS.get(col, 0.0) for col in feature_names}])

        # ── Current labs & vitals ──
        input_data.at[0, 'age']            = float(age)
        input_data.at[0, 'sex']            = float(sex)
        input_data.at[0, 'lab_hba1c']      = float(hba1c)
        input_data.at[0, 'lab_fpg']        = float(fpg)
        input_data.at[0, 'vitalsign_sbp']  = float(sbp)
        input_data.at[0, 'Period']         = float(period)

        # ── Current medications ──
        input_data.at[0, 'med_insulin']    = float(insulin_current)
        input_data.at[0, 'med_glipizide']  = float(glipizide_current)
        input_data.at[0, 'med_metformin']  = float(metformin_current)

        # ── Baseline (at-diagnosis) history — independent of current meds ──
        input_data.at[0, 'med_insulin_baseline']   = float(insulin_baseline)
        input_data.at[0, 'med_glipizide_baseline']  = float(glipizide_baseline)
        input_data.at[0, 'med_metformin_baseline']  = float(metformin_baseline)
        input_data.at[0, 'co_dementia_baseline']    = float(dementia_baseline)

        input_data = input_data.astype(float)

        # ── Run prediction ──
        if IS_MAPIE:
            # MAPIE 1.4: predict() for point, predict_interval() for CI
            y_pred_arr = model.predict(input_data)
            y_pred_pts, y_pis_arr = model.predict_interval(input_data)
            prediction = float(y_pred_arr[0])
            ci_lower   = float(y_pis_arr[0, 0, 0])
            ci_upper   = float(y_pis_arr[0, 1, 0])
            has_ci = True
        else:
            prediction = float(model.predict(input_data)[0])
            ci_lower = ci_upper = None
            has_ci = False

        # ── Audit log ──
        try:
            append_audit_log({
                'timestamp':          datetime.now().isoformat(timespec='seconds'),
                'age':                age,
                'sex':                sex_label,
                'hba1c':              hba1c,
                'fpg':                fpg,
                'sbp':                sbp,
                'insulin_current':    insulin_current,
                'insulin_baseline':   insulin_baseline,
                'glipizide_baseline': glipizide_baseline,
                'metformin_baseline': metformin_baseline,
                'dementia_baseline':  dementia_baseline,
                'period':             period,
                'predicted_hba1c':    round(prediction, 3),
                'ci_lower':           round(ci_lower, 3) if ci_lower is not None else '',
                'ci_upper':           round(ci_upper, 3) if ci_upper is not None else '',
            })
        except Exception:
            pass  # Logging failure must never block the clinical display

    # ──────────────────────────────────────────
    # RESULTS DISPLAY
    # ──────────────────────────────────────────

    # Traffic-light colour scheme
    if prediction < 6.5:
        color     = "#059669"   # green
        bg_color  = "#ecfdf5"
        border    = "#a7f3d0"
        level     = "🟢 ควบคุมได้ดีมาก"
        level_en  = "Well Controlled"
        tips = [
            "รักษาพฤติกรรมการกินและการออกกำลังกายที่ดีต่อไป",
            "วัดน้ำตาลด้วยตนเองตามคำแนะนำของแพทย์อย่างสม่ำเสมอ",
        ]
    elif prediction < 8.0:
        color     = "#d97706"   # amber
        bg_color  = "#fffbeb"
        border    = "#fde68a"
        level     = "🟡 เฝ้าระวัง"
        level_en  = "Monitor Closely"
        tips = [
            "ลดการบริโภคน้ำตาลและแป้งขัดขาว",
            "เพิ่มการออกกำลังกายอย่างน้อย 150 นาที/สัปดาห์",
            "พบแพทย์ตามนัดและปรึกษาเรื่องการปรับยา",
        ]
    else:
        color     = "#dc2626"   # red
        bg_color  = "#fef2f2"
        border    = "#fca5a5"
        level     = "🔴 ความเสี่ยงสูง"
        level_en  = "High Risk — Urgent Review"
        tips = [
            "นัดพบแพทย์ก่อนกำหนดเพื่อพิจารณาปรับแผนการรักษาทันที",
            "หลีกเลี่ยงอาหารที่มีน้ำตาลสูงทุกประเภท",
            "ติดตามค่าน้ำตาลในเลือดอย่างใกล้ชิดและปฏิบัติตามแผนยาอย่างเคร่งครัด",
        ]

    # Point estimate + CI display
    ci_text = (
        f"ช่วงความเชื่อมั่น 90%: {ci_lower:.1f}% – {ci_upper:.1f}%"
        if has_ci else "ไม่มีข้อมูลช่วงความเชื่อมั่น (โมเดลเก่า)"
    )

    st.markdown(f"""
    <div class="result-card" style="border-left: 6px solid {color}; background: {bg_color};">
      <div style="color:{color}; font-size:0.9rem; font-weight:600; margin-bottom:4px;">{level} ({level_en})</div>
      <div class="result-value" style="color:{color};">{prediction:.2f}%</div>
      <div class="result-range">{ci_text}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**💡 คำแนะนำ:**")
    for tip in tips:
        st.markdown(f"• {tip}")

    # Input summary metrics
    st.markdown("""
    <div class="metric-row">
      <div class="metric-badge"><div class="label">HbA1c วันนี้</div><div class="value">{:.1f}%</div></div>
      <div class="metric-badge"><div class="label">FPG</div><div class="value">{} mg/dL</div></div>
      <div class="metric-badge"><div class="label">SBP</div><div class="value">{} mmHg</div></div>
      <div class="metric-badge"><div class="label">Period</div><div class="value">{} วัน</div></div>
    </div>
    """.format(hba1c, fpg, sbp, period), unsafe_allow_html=True)

    if has_ci:
        st.caption(
            "ℹ️ ช่วงความเชื่อมั่น 90% คำนวณโดย Conformal Prediction (MAPIE Jackknife+) "
            "หมายความว่าในระยะยาว ค่า HbA1c จริงจะอยู่ในช่วงนี้ประมาณ 90% ของเวลา"
        )

    # Audit log notification
    if os.path.exists(LOG_FILE):
        st.caption(f"📁 บันทึกการพยากรณ์นี้ลงใน `{LOG_FILE}` เรียบร้อยแล้ว")
