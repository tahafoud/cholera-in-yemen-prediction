"""
تشغيل كامل: توليد البيانات → pipeline → تدريب النموذج
python src/train.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from generate_data import gen_cholera, gen_rainfall, gen_idps, gen_governorate_data
from pipeline      import build_features, get_features
from model         import train_evaluate, forecast_weeks

def main():
    print("=" * 60)
    print("  نظام الإنذار المبكر للكوليرا – اليمن 2026-2027")
    print("=" * 60)

    # 1. توليد البيانات
    print("\n[1/3] توليد البيانات الاصطناعية...")
    cholera  = gen_cholera()
    rainfall = gen_rainfall(cholera['Date'])
    idps     = gen_idps(cholera['Date'])
    gen_governorate_data(cholera['Date'], cholera)

    # 2. هندسة الميزات
    print("\n[2/3] هندسة الميزات...")
    df       = build_features(cholera, rainfall, idps)
    features = get_features(df)
    print(f"  عدد الميزات: {len(features)}")

    # 3. تدريب النموذج
    print("\n[3/3] تدريب النموذج...")
    results = train_evaluate(df, features)
    m = results['metrics']

    # 4. ملخص
    print("\n" + "=" * 60)
    print("  ✅ الملخص النهائي")
    print("=" * 60)
    print(f"  الدقة      : {m['accuracy']:.1f}%")
    print(f"  R² Score   : {m['R2']:.4f}")
    print(f"  MAPE       : {m['MAPE']:.2f}%")
    print(f"  WF R² avg  : {m['wf_r2_mean']:.4f}")
    print("=" * 60)

    # 5. تنبؤ نموذجي
    print("\n🔮 تنبؤ الأسابيع الـ 8 القادمة:")
    future = forecast_weeks(results['model'], df, features, 8)
    print(future[['Date','Predicted_Cases','CI_Lower','CI_Upper']].to_string(index=False))

    print(f"\n✅ شغّل الواجهة بالأمر:\n   streamlit run app.py\n")

if __name__ == "__main__":
    main()
