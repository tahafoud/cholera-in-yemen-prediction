"""
توليد بيانات اصطناعية واقعية لوباء الكوليرا في اليمن 2026-2027
مبنية على:
- نمط وباء 2017-2018 (الأكبر في التاريخ الحديث)
- إضافة موجة ثانية محتملة (2026) بسبب استمرار الأزمة الإنسانية
- بيانات أمطار موسمية واقعية (موسم الخريف + الصيف)
- بيانات نازحين محدّثة (OCHA 2025 estimates)
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(2026)


# ─────────────────────────────────────────────────────────────
# 1. بيانات الكوليرا 2026-01-01 → 2027-03-31 (67 أسبوع)
# ─────────────────────────────────────────────────────────────
def gen_cholera():
    dates = pd.date_range("2026-01-05", periods=67, freq="W-MON")
    n = len(dates)
    t = np.arange(n)

    # موجة أولى: يناير-أبريل 2026 (ناجمة عن أمطار غير موسمية + تدهور WASH)
    wave1 = 18000 * np.exp(-((t - 10) ** 2) / (2 * 8**2))

    # هدوء صيفي نسبي
    calm = 4000 * np.ones(n)

    # موجة ثانية: أكتوبر-ديسمبر 2026 (موسم الأمطار الخريفي + تدفق نازحين)
    wave2 = 25000 * np.exp(-((t - 42) ** 2) / (2 * 10**2))

    # موجة ثالثة أخف: فبراير 2027
    wave3 = 12000 * np.exp(-((t - 58) ** 2) / (2 * 6**2))

    base = calm + wave1 + wave2 + wave3
    noise = np.random.normal(0, 1200, n)
    new_cases = np.clip(base + noise, 500, None)

    # معدل الوفيات يرتفع في الموجات
    cfr_base = 0.006 + 0.004 * (base / base.max())
    new_deaths = new_cases * cfr_base * np.random.uniform(0.85, 1.15, n)

    cases  = new_cases.cumsum()
    deaths = new_deaths.cumsum()
    cfr    = deaths / cases * 100
    attack = cases / 34_000_000 * 1000  # تعداد يمن 2026 تقريباً

    df = pd.DataFrame({
        'Date':                    dates,
        'New_Cases':               new_cases.round(0),
        'New_Deaths':              new_deaths.round(1),
        'Cases':                   cases.round(0),
        'Deaths':                  deaths.round(1),
        'CFR (%)':                 cfr.round(3),
        'Attack Rate (per 1000)':  attack.round(4),
        'Governorate_Count':       np.random.randint(12, 22, n),   # عدد المحافظات المتأثرة
        'Severity_Index':          (base / base.max()).round(3),    # مؤشر الشدة 0-1
    })

    path = OUT_DIR / "cholera_yemen_2026_2027.csv"
    df.to_csv(path, index=False)
    print(f"✅ بيانات الكوليرا: {n} أسبوع → {path.name}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. بيانات الأمطار (CHIRPS-like) 2026-2027
# ─────────────────────────────────────────────────────────────
def gen_rainfall(dates):
    n = len(dates)
    months = dates.dt.month

    # نمط موسمي اليمن: ذروة مارس-مايو وأغسطس-سبتمبر
    seasonal = (
        20 * np.sin(2 * np.pi * (months - 3) / 12) +
        15 * np.sin(2 * np.pi * (months - 8) / 6)
    ).clip(0)

    # حدث مطري شاذ في مارس 2026 (El Niño influence)
    elnino_boost = np.zeros(n)
    elnino_idx = (dates >= "2026-03-01") & (dates <= "2026-04-30")
    elnino_boost[elnino_idx] = 35

    mm = seasonal + elnino_boost + np.random.gamma(1.5, 3, n)
    mm = mm.clip(0)

    # شذوذ بالنسبة للمتوسط التاريخي (1990-2020)
    hist_mean = seasonal.mean()
    anomaly = ((mm - hist_mean) / (hist_mean + 1) * 100).round(1)

    df = pd.DataFrame({
        'Date':              dates,
        'rainfall_mm':       mm.round(2),
        'rainfall_anomaly':  anomaly,
        'rain_7d_accum':     mm.rolling(1).sum().round(2),   # تراكم أسبوعي
    })

    path = OUT_DIR / "rainfall_yemen_2026_2027.csv"
    df.to_csv(path, index=False)
    print(f"✅ بيانات الأمطار: {n} سجل → {path.name}")
    return df


# ─────────────────────────────────────────────────────────────
# 3. بيانات النازحين الداخليين (IDP) 2026-2027
# ─────────────────────────────────────────────────────────────
def gen_idps(dates):
    n = len(dates)
    t = np.arange(n)

    # قاعدة: 4.5 مليون نازح (OCHA 2025 baseline)
    base = 4_500_000
    # ارتفاع تدريجي خلال 2026 بسبب التصعيد
    trend = 50_000 * t / n
    # تذبذب موسمي (حركة أكثر في أوقات الصراع الشتوي)
    seasonal = 120_000 * np.sin(2 * np.pi * t / 52)
    noise = np.random.normal(0, 30_000, n)

    idps = (base + trend + seasonal + noise).clip(3_500_000, 6_000_000)

    # بيانات WASH: نسبة الوصول إلى مياه آمنة (%)
    wash_access = np.clip(55 - 8 * np.sin(2 * np.pi * t / 52) + np.random.normal(0, 2, n), 40, 70)

    # كثافة الازدحام في المخيمات (أشخاص/مخيم تقريباً)
    camp_density = idps / np.random.randint(180, 220, n)

    df = pd.DataFrame({
        'Date':          dates,
        'total_idps':    idps.astype(int),
        'wash_access_%': wash_access.round(1),
        'camp_density':  camp_density.astype(int),
    })

    path = OUT_DIR / "idps_yemen_2026_2027.csv"
    df.to_csv(path, index=False)
    print(f"✅ بيانات النازحين: {n} سجل → {path.name}")
    return df


# ─────────────────────────────────────────────────────────────
# 4. بيانات جغرافية: محافظات اليمن (توزيع الحالات)
# ─────────────────────────────────────────────────────────────
GOVERNORATES = {
    'صنعاء':     {'lat': 15.35, 'lon': 44.21, 'pop': 2_600_000, 'risk': 0.90},
    'عدن':       {'lat': 12.78, 'lon': 45.03, 'pop': 900_000,   'risk': 0.75},
    'تعز':       {'lat': 13.58, 'lon': 44.02, 'pop': 3_000_000, 'risk': 0.95},
    'حضرموت':   {'lat': 14.93, 'lon': 48.52, 'pop': 1_200_000, 'risk': 0.55},
    'الحديدة':  {'lat': 14.80, 'lon': 42.95, 'pop': 2_800_000, 'risk': 0.98},
    'إب':        {'lat': 13.97, 'lon': 44.18, 'pop': 2_400_000, 'risk': 0.88},
    'مأرب':      {'lat': 15.47, 'lon': 45.32, 'pop': 700_000,   'risk': 0.80},
    'ذمار':      {'lat': 14.55, 'lon': 44.40, 'pop': 1_500_000, 'risk': 0.70},
    'حجة':       {'lat': 15.70, 'lon': 43.60, 'pop': 1_700_000, 'risk': 0.85},
    'الضالع':   {'lat': 13.69, 'lon': 44.73, 'pop': 600_000,   'risk': 0.72},
    'أبين':      {'lat': 13.15, 'lon': 45.23, 'pop': 500_000,   'risk': 0.65},
    'البيضاء':  {'lat': 14.00, 'lon': 45.57, 'pop': 600_000,   'risk': 0.68},
    'المهرة':   {'lat': 16.52, 'lon': 51.58, 'pop': 130_000,   'risk': 0.30},
    'لحج':       {'lat': 13.05, 'lon': 44.88, 'pop': 900_000,   'risk': 0.78},
    'شبوة':      {'lat': 15.35, 'lon': 47.01, 'pop': 600_000,   'risk': 0.60},
    'صعدة':      {'lat': 16.94, 'lon': 43.77, 'pop': 900_000,   'risk': 0.82},
    'عمران':     {'lat': 15.66, 'lon': 43.94, 'pop': 1_100_000, 'risk': 0.76},
    'ريمة':      {'lat': 14.78, 'lon': 43.53, 'pop': 500_000,   'risk': 0.74},
    'المحويت':  {'lat': 15.46, 'lon': 43.54, 'pop': 500_000,   'risk': 0.71},
    'الجوف':    {'lat': 16.22, 'lon': 45.48, 'pop': 500_000,   'risk': 0.65},
    'سقطرى':    {'lat': 12.46, 'lon': 53.82, 'pop': 60_000,    'risk': 0.25},
}

def gen_governorate_data(dates, cholera_df):
    """توزيع الحالات على المحافظات بشكل واقعي."""
    rows = []
    for _, wrow in cholera_df.iterrows():
        total_cases = wrow['New_Cases']
        risks = np.array([v['risk'] for v in GOVERNORATES.values()])
        risks = risks / risks.sum()  # normalize
        case_dist = np.random.dirichlet(risks * 10) * total_cases

        for i, (gov, info) in enumerate(GOVERNORATES.items()):
            gov_cases = int(case_dist[i])
            rows.append({
                'Date':         wrow['Date'],
                'Governorate':  gov,
                'Lat':          info['lat'],
                'Lon':          info['lon'],
                'Population':   info['pop'],
                'Risk_Score':   info['risk'],
                'New_Cases':    gov_cases,
                'Attack_Rate':  gov_cases / info['pop'] * 1000,
            })

    df = pd.DataFrame(rows)
    path = OUT_DIR / "cholera_by_governorate.csv"
    df.to_csv(path, index=False)
    print(f"✅ بيانات المحافظات: {len(df)} سجل → {path.name}")
    return df


# ─────────────────────────────────────────────────────────────
# 5. تشغيل
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 توليد بيانات نظام الكوليرا 2026-2027...\n")
    cholera  = gen_cholera()
    rainfall = gen_rainfall(cholera['Date'])
    idps     = gen_idps(cholera['Date'])
    gov_df   = gen_governorate_data(cholera['Date'], cholera)
    print(f"\n✅ جميع الملفات محفوظة في: {OUT_DIR.resolve()}")
