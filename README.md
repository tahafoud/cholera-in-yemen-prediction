Eng:Taha alshawesh

# 🏥 نظام الإنذار المبكر للكوليرا – اليمن 2026-2027

## هيكل المشروع

```
cholera_system/
├── app.py                  ← واجهة Streamlit الرئيسية
├── requirements.txt        ← المكتبات المطلوبة
├── src/
│   ├── generate_data.py   ← توليد البيانات (2026-2027)
│   ├── pipeline.py        ← معالجة البيانات + هندسة الميزات
│   ├── model.py           ← النموذج (Stacking Ensemble)
│   └── train.py           ← سكريبت التدريب الكامل
├── data/                  ← البيانات (تُولَّد تلقائياً)
└── outputs/               ← النموذج المحفوظ (تُولَّد تلقائياً)
```

## التشغيل السريع

```bash
# 1. تثبيت المكتبات
pip install -r requirements.txt

# 2. تدريب النموذج (يولّد البيانات تلقائياً)
python src/train.py

# 3. تشغيل الواجهة
streamlit run app.py
```

## الإصلاحات الجوهرية (v2.0)

| المشكلة القديمة          | الحل الجديد                         |
| ------------------------ | ----------------------------------- |
| تسرب: `New_Deaths` كميزة | محذوفة تماماً                       |
| الهدف = نفس الأسبوع      | الهدف = الأسبوع القادم (`shift=-1`) |
| أمطار = صفر              | أنماط موسمية واقعية                 |
| نموذج واحد (XGBoost)     | Stacking Ensemble (5 نماذج)         |
| حلقة تنبؤ خاطئة          | تحديث كامل لكل الميزات في كل خطوة   |
| واجهة LTR                | RTL كاملة + Cairo font              |
| بلا خريطة                | خريطة تفاعلية 21 محافظة             |

## مكونات النموذج

- **XGBoost** (500 estimators, lr=0.03)
- **LightGBM** (500 estimators, lr=0.03)
- **GradientBoosting** (300 estimators)
- **RandomForest** (300 estimators)
- **ExtraTrees** (200 estimators)
- **Meta-Learner**: HuberRegressor (مقاوم للقيم الشاذة)
- **التقييم**: Walk-Forward TimeSeriesSplit (5 folds)

## المطوّر

طه فؤاد عبدالرحمن الشاوش  
معيد ومحاضر – قسم علوم الحاسوب – جامعة ذمار  
github.com/tahafoud
