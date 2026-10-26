"""
نظام التنبؤ بالكوليرا في اليمن - النموذج
model.py · v2.0 (2026-2027)

الاستراتيجية:
  • Stacking Ensemble: XGBoost + LightGBM + GradientBoosting + RandomForest
  • Meta-learner: HuberRegressor (مقاوم للقيم الشاذة)
  • Walk-Forward Validation (TimeSeriesSplit)
  • تنبؤ مستقبلي مع فترات ثقة
  • SHAP لتفسير النموذج
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
class StackingEnsemble:
    def __init__(self):
        self.base_models  = self._build_base()
        self.meta_learner = HuberRegressor(epsilon=1.35, max_iter=500)
        self.scaler       = RobustScaler()
        self.feature_importances_ = None
        self.feature_names_       = None

    def _build_base(self):
        models = {}
        if HAS_XGB:
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.03, max_depth=5,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                reg_alpha=0.2, reg_lambda=1.5, gamma=0.1,
                random_state=42, verbosity=0
            )
        if HAS_LGB:
            models['lgb'] = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.03, max_depth=6,
                num_leaves=31, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.2, reg_lambda=1.5, random_state=42, verbose=-1
            )
        models['rf']  = RandomForestRegressor(
            n_estimators=300, max_depth=7, min_samples_leaf=2,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        models['gbm'] = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.04, max_depth=4,
            subsample=0.8, min_samples_leaf=3, random_state=42
        )
        models['et']  = ExtraTreesRegressor(
            n_estimators=200, max_depth=7, random_state=42, n_jobs=-1
        )
        return models

    def _oof(self, X, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof  = np.zeros((len(y), len(self.base_models)))
        for fold, (tr, va) in enumerate(tscv.split(X)):
            for j, m in enumerate(self.base_models.values()):
                m.fit(X[tr], y[tr])
                oof[va, j] = m.predict(X[va])
        return oof

    def fit(self, X_df, y):
        self.feature_names_ = list(X_df.columns)
        X = self.scaler.fit_transform(X_df)
        oof = self._oof(X, y.values)
        for m in self.base_models.values():
            m.fit(X, y.values)
        self.meta_learner.fit(oof, y.values)
        imps = [m.feature_importances_
                for m in self.base_models.values()
                if hasattr(m, 'feature_importances_')]
        if imps:
            self.feature_importances_ = np.mean(imps, axis=0)
        return self

    def predict(self, X_df):
        X     = self.scaler.transform(X_df)
        preds = np.column_stack([m.predict(X) for m in self.base_models.values()])
        return self.meta_learner.predict(preds).clip(0)

    def predict_interval(self, X_df, z=1.645):
        X     = self.scaler.transform(X_df)
        preds = np.column_stack([m.predict(X) for m in self.base_models.values()])
        point = self.meta_learner.predict(preds).clip(0)
        spread = preds.std(axis=1)
        return point, (point - z*spread).clip(0), (point + z*spread).clip(0)


# ──────────────────────────────────────────────────────────────
def train_evaluate(df: pd.DataFrame, feature_cols: list) -> dict:
    X = df[feature_cols]
    y = df['target']

    split = int(len(X) * 0.80)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    print(f"\n📊 تدريب: {len(X_tr)} أسبوع | اختبار: {len(X_te)} أسبوع")

    model = StackingEnsemble()
    print("⚙️  تدريب Stacking Ensemble ...")
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    mae    = mean_absolute_error(y_te, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    r2     = r2_score(y_te, y_pred)
    mape   = np.mean(np.abs((y_te.values - y_pred) / (y_te.values + 1))) * 100
    acc    = max(0.0, 100 - mape)

    # Walk-forward
    tscv = TimeSeriesSplit(n_splits=5)
    wf_r2 = []
    for tr_i, va_i in tscv.split(X):
        m = StackingEnsemble()
        m.fit(X.iloc[tr_i], y.iloc[tr_i])
        wf_r2.append(r2_score(y.iloc[va_i], m.predict(X.iloc[va_i])))

    print(f"\n{'='*55}")
    print(f"  MAE       : {mae:>10,.0f} حالة")
    print(f"  RMSE      : {rmse:>10,.0f} حالة")
    print(f"  R²        : {r2:>10.4f}")
    print(f"  MAPE      : {mape:>10.2f}%")
    print(f"  الدقة     : {acc:>10.1f}%")
    print(f"  WF R² avg : {np.mean(wf_r2):>10.4f} ± {np.std(wf_r2):.4f}")
    print(f"{'='*55}")

    joblib.dump(model,        OUT_DIR / "cholera_model.pkl")
    joblib.dump(feature_cols, OUT_DIR / "features_list.pkl")

    # SHAP
    shap_df = _shap_importance(model, X)

    return {
        'model': model,
        'features': feature_cols,
        'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2,
                    'MAPE': mape, 'accuracy': acc,
                    'wf_r2_mean': np.mean(wf_r2),
                    'wf_r2_std':  np.std(wf_r2)},
        'test': {'dates':     df['Date'].iloc[split:].values,
                 'actual':    y_te.values,
                 'predicted': y_pred},
        'shap_df': shap_df,
        'df': df,
    }


def _shap_importance(model, X_df) -> pd.DataFrame | None:
    try:
        import shap
        best = (model.base_models.get('xgb')
                or model.base_models.get('lgb')
                or model.base_models.get('rf'))
        if best is None:
            return None
        Xs = pd.DataFrame(model.scaler.transform(X_df), columns=X_df.columns)
        exp = shap.TreeExplainer(best)
        sv  = exp.shap_values(Xs)
        return (pd.DataFrame({'feature': X_df.columns, 'shap': np.abs(sv).mean(0)})
                  .sort_values('shap', ascending=False).reset_index(drop=True))
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# عتبات ديناميكية
# ──────────────────────────────────────────────────────────────
def dynamic_thresholds(series: pd.Series) -> dict:
    return {
        'green':  float(series.quantile(0.25)),
        'yellow': float(series.quantile(0.60)),
        'orange': float(series.quantile(0.80)),
        'red':    float(series.quantile(0.92)),
    }

ALERT_INFO = {
    'green':  {'label': '🟢 وضع طبيعي',            'color': '#28a745', 'bg': '#d4edda',
               'action': 'مراقبة روتينية ورصد جودة المياه.'},
    'yellow': {'label': '🟡 تأهب – مراقبة مكثفة',  'color': '#856404', 'bg': '#fff3cd',
               'action': 'تعزيز كلور المياه، توزيع ORS، تفعيل فرق الرصد.'},
    'orange': {'label': '🟠 إنذار متقدم',           'color': '#8B4513', 'bg': '#FFE4B5',
               'action': 'نشر عيادات متنقلة، تنسيق مع WHO/UNICEF، فتح خط ساخن صحي.'},
    'red':    {'label': '🔴 تفشٍ حاد – تدخل فوري', 'color': '#721c24', 'bg': '#f8d7da',
               'action': 'إعلان طوارئ صحية، فتح مراكز CTCs، تنسيق MSF/ICRC.'},
}

def classify(cases: float, thr: dict) -> str:
    if cases < thr['green']:  return 'green'
    if cases < thr['yellow']: return 'yellow'
    if cases < thr['orange']: return 'orange'
    return 'red'


# ──────────────────────────────────────────────────────────────
# تنبؤ مستقبلي صحيح
# ──────────────────────────────────────────────────────────────
def forecast_weeks(model: StackingEnsemble, df: pd.DataFrame,
                   feature_cols: list, n: int = 12) -> pd.DataFrame:
    hist = df.copy()
    last_date = hist['Date'].max()
    rows = []

    for i in range(1, n + 1):
        nc  = hist['New_Cases']
        rm  = hist['rainfall_mm'] if 'rainfall_mm' in hist else pd.Series([0]*len(hist))
        idp = hist['total_idps']  if 'total_idps'  in hist else pd.Series([0]*len(hist))

        r = hist.iloc[-1].copy()
        for lag in [1, 2, 3, 4, 8]:
            col = f'cases_lag_{lag}'
            if col in feature_cols:
                r[col] = nc.iloc[-lag] if len(nc) >= lag else 0

        r['cases_roll4_mean'] = nc.tail(4).mean()
        r['cases_roll8_mean'] = nc.tail(8).mean()
        r['cases_roll4_std']  = nc.tail(4).std() if len(nc) >= 2 else 0
        r['cases_roll4_max']  = nc.tail(4).max()
        r['cases_delta1']     = float(nc.iloc[-1]) - float(nc.iloc[-2] if len(nc)>=2 else 0)
        r['cases_delta2']     = float(nc.iloc[-1]) - float(nc.iloc[-3] if len(nc)>=3 else 0)
        r['cases_pct_change'] = float(nc.iloc[-1]) / (float(nc.iloc[-2])+1) - 1 if len(nc)>=2 else 0
        r['cases_accel']      = r['cases_delta1'] - (float(nc.iloc[-2]) - float(nc.iloc[-3] if len(nc)>=3 else 0))

        for lag in [1, 2, 3, 4]:
            col = f'rain_lag_{lag}'
            if col in feature_cols:
                r[col] = float(rm.iloc[-lag]) if len(rm) >= lag else 0
        if 'rain_roll4_mean' in feature_cols:
            r['rain_roll4_mean'] = rm.tail(4).mean()

        r['idps_lag1']       = float(idp.iloc[-1]) if len(idp) >= 1 else 0
        r['idps_pct_change'] = float(idp.iloc[-1]) / (float(idp.iloc[-2])+1) - 1 if len(idp)>=2 else 0

        future_date = last_date + pd.Timedelta(weeks=i)
        r['week_of_year']  = future_date.isocalendar()[1]
        r['month']         = future_date.month
        r['quarter']       = (future_date.month - 1) // 3 + 1
        r['weeks_elapsed'] += i

        X_row = pd.DataFrame([r[feature_cols]])
        pt, lo, hi = model.predict_interval(X_row)

        rows.append({
            'Date':            future_date,
            'Predicted_Cases': max(0, float(pt[0])),
            'CI_Lower':        max(0, float(lo[0])),
            'CI_Upper':        max(0, float(hi[0])),
            'week':            i,
        })

        nr = r.copy()
        nr['Date']      = future_date
        nr['New_Cases'] = max(0, float(pt[0]))
        hist = pd.concat([hist, pd.DataFrame([nr])], ignore_index=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pipeline import run, get_features
    df = run()
    fc = get_features(df)
    results = train_evaluate(df, fc)
    m = results['metrics']
    print(f"\n🎯 الدقة: {m['accuracy']:.1f}%  R²={m['R2']:.4f}")
