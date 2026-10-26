"""
نظام التنبؤ بالكوليرا في اليمن - معالجة البيانات وهندسة الميزات
pipeline.py · v2.0 (2026-2027)

الإصلاحات الجوهرية:
✅ لا تسرب: الهدف = New_Cases الأسبوع القادم (shift=-1)
✅ New_Deaths محذوفة من الميزات (غير معروفة مسبقاً)
✅ الأمطار المتأخرة 2-3 أسابيع (فترة حضانة الكوليرا)
✅ ميزات WASH وكثافة المخيمات (2026 additions)
✅ مؤشر الشدة التاريخي كميزة سياقية
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cholera  = pd.read_csv(DATA_DIR / "cholera_yemen_2026_2027.csv",  parse_dates=['Date'])
    rainfall = pd.read_csv(DATA_DIR / "rainfall_yemen_2026_2027.csv", parse_dates=['Date'])
    idps     = pd.read_csv(DATA_DIR / "idps_yemen_2026_2027.csv",     parse_dates=['Date'])
    return cholera, rainfall, idps


def build_features(cholera, rainfall, idps) -> pd.DataFrame:
    # دمج
    df = cholera.copy().sort_values('Date')
    df = pd.merge_asof(df, rainfall.sort_values('Date'),
                       on='Date', direction='nearest', tolerance=pd.Timedelta('7d'))
    df = pd.merge_asof(df, idps.sort_values('Date'),
                       on='Date', direction='nearest', tolerance=pd.Timedelta('14d'))

    df = df.sort_values('Date').reset_index(drop=True)

    nc = df['New_Cases']
    rm = df['rainfall_mm']
    idp = df['total_idps']

    # ── Lag Cases ──────────────────────────────────────
    for lag in [1, 2, 3, 4, 8]:
        df[f'cases_lag_{lag}'] = nc.shift(lag)

    # ── Rolling statistics ─────────────────────────────
    df['cases_roll4_mean'] = nc.shift(1).rolling(4).mean()
    df['cases_roll8_mean'] = nc.shift(1).rolling(8).mean()
    df['cases_roll4_std']  = nc.shift(1).rolling(4).std()
    df['cases_roll4_max']  = nc.shift(1).rolling(4).max()

    # ── Momentum & trend ──────────────────────────────
    df['cases_delta1']     = nc.shift(1) - nc.shift(2)
    df['cases_delta2']     = nc.shift(1) - nc.shift(3)
    df['cases_pct_change'] = nc.shift(1).pct_change(fill_method=None).fillna(0)
    df['cases_accel']      = df['cases_delta1'] - df['cases_delta1'].shift(1)  # تسارع

    # ── Lag Rainfall (2-4 أسابيع) ─────────────────────
    for lag in [1, 2, 3, 4]:
        df[f'rain_lag_{lag}'] = rm.shift(lag)
    df['rain_roll4_mean']  = rm.shift(1).rolling(4).mean()
    df['rain_anomaly_lag2'] = df['rainfall_anomaly'].shift(2)

    # ── IDP features ──────────────────────────────────
    df['idps_lag1']        = idp.shift(1)
    df['idps_pct_change']  = idp.pct_change(fill_method=None).fillna(0)
    df['idps_x_rain']      = df['idps_lag1'] * df['rain_lag_2'] / 1e10  # تفاعل

    # ── WASH & density (2026 additions) ───────────────
    if 'wash_access_%' in df.columns:
        df['wash_lag1']     = df['wash_access_%'].shift(1)
        df['camp_density_lag1'] = df['camp_density'].shift(1) if 'camp_density' in df.columns else 0

    # ── Contextual ────────────────────────────────────
    if 'Severity_Index' in df.columns:
        df['severity_lag1'] = df['Severity_Index'].shift(1)
    if 'Governorate_Count' in df.columns:
        df['gov_count_lag1'] = df['Governorate_Count'].shift(1)

    # ── Time features ─────────────────────────────────
    df['week_of_year']     = df['Date'].dt.isocalendar().week.astype(int)
    df['month']            = df['Date'].dt.month
    df['quarter']          = df['Date'].dt.quarter
    df['weeks_elapsed']    = (df['Date'] - df['Date'].min()).dt.days // 7

    # ── CFR lag (بدون تسرب) ───────────────────────────
    df['cfr_lag1']         = df['CFR (%)'].shift(1)

    # ── الهدف الصحيح (الأسبوع القادم) ────────────────
    df['target'] = df['New_Cases'].shift(-1)

    # إزالة صفوف بلا هدف وملء باقي القيم
    df = df.dropna(subset=['target']).fillna(0).reset_index(drop=True)

    print(f"✅ هندسة الميزات: {df.shape[0]} أسبوع × {df.shape[1]} عمود")
    df.to_csv(OUT_DIR / "final_features.csv", index=False)
    return df


EXCLUDE = {'Date', 'New_Cases', 'New_Deaths', 'Cases', 'Deaths',
           'CFR (%)', 'Attack Rate (per 1000)', 'target',
           'Severity_Index', 'Governorate_Count',
           'wash_access_%', 'camp_density', 'rainfall_mm',
           'rainfall_anomaly', 'rain_7d_accum', 'total_idps'}

def get_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE]


def run():
    cholera, rainfall, idps = load_all()
    df = build_features(cholera, rainfall, idps)
    return df


if __name__ == "__main__":
    run()
