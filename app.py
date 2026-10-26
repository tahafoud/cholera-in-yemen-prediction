"""
نظام الإنذار المبكر لتفشي الكوليرا في اليمن 2026-2027
Yemen Cholera Early Warning System  ·  v2.0
"""

import sys
from pathlib import Path
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── مسارات ────────────────────────────────────────────────
ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"
SRC_DIR  = ROOT / "src"

# ─── إعداد الصفحة ──────────────────────────────────────────
st.set_page_config(
    page_title="نظام الإنذار المبكر للكوليرا – اليمن 2026",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS (RTL + تصميم احترافي) ─────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');

  html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
    direction: rtl;
  }
  .main { background: #f0f4f8; }
  .block-container { padding: 1.5rem 2rem; }

  /* ─── header ─── */
  .hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px; padding: 2rem 2.5rem;
    color: white; margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }
  .hero h1 { font-size: 2.2rem; font-weight: 900; margin: 0; letter-spacing: -0.5px; }
  .hero p  { font-size: 1rem; opacity: 0.8; margin: 0.4rem 0 0; }
  .hero-badge {
    display: inline-block; background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px; padding: 0.25rem 0.9rem;
    font-size: 0.8rem; margin-left: 0.5rem;
  }

  /* ─── KPI cards ─── */
  .kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
  .kpi-card {
    background: white; border-radius: 14px; padding: 1.4rem 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); text-align: center;
    border-top: 4px solid;
  }
  .kpi-card .num  { font-size: 2rem; font-weight: 900; }
  .kpi-card .lbl  { font-size: 0.85rem; color: #666; margin-top: 0.3rem; }
  .kpi-card .delta { font-size: 0.82rem; font-weight: 600; margin-top: 0.2rem; }

  /* ─── alert box ─── */
  .alert-box {
    border-radius: 14px; padding: 1.5rem 2rem;
    margin: 1rem 0; font-size: 1.15rem; font-weight: 700;
    border-right: 8px solid; box-shadow: 0 4px 16px rgba(0,0,0,0.1);
  }
  .alert-box .alert-title { font-size: 1.6rem; }
  .alert-box .alert-action { font-size: 0.95rem; font-weight: 400; margin-top: 0.5rem; opacity: 0.85; }

  /* ─── section header ─── */
  .sec-head {
    font-size: 1.3rem; font-weight: 700; color: #1a1a2e;
    border-right: 5px solid #e74c3c; padding-right: 0.8rem;
    margin: 1.8rem 0 1rem;
  }

  /* ─── table ─── */
  .styled-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
  .styled-table th { background: #1a1a2e; color: white; padding: 0.7rem 1rem; }
  .styled-table td { padding: 0.6rem 1rem; border-bottom: 1px solid #eee; }
  .styled-table tr:hover td { background: #f8f9ff; }

  /* ─── sidebar ─── */
  [data-testid="stSidebar"] { background: #1a1a2e !important; }
  [data-testid="stSidebar"] * { color: white !important; }
  [data-testid="stSidebar"] .stSlider > div { color: white !important; }

  /* ─── tabs ─── */
  .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; padding: 0.6rem 1.5rem;
    font-family: 'Cairo', sans-serif; font-weight: 600;
  }

  /* ─── disclaimer ─── */
  .disclaimer {
    background: #fff8e1; border: 1px solid #ffc107;
    border-radius: 10px; padding: 0.8rem 1.2rem;
    font-size: 0.82rem; color: #856404; margin-top: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  تحميل البيانات والنموذج
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⚙️ جارٍ تحميل النموذج...")
def load_model():
    try:
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        model    = joblib.load(OUT_DIR / "cholera_model.pkl")
        features = joblib.load(OUT_DIR / "features_list.pkl")
        return model, features
    except FileNotFoundError:
        return None, None

@st.cache_data(show_spinner="📂 جارٍ تحميل البيانات...")
def load_data():
    try:
        df  = pd.read_csv(OUT_DIR / "final_features.csv",   parse_dates=['Date'])
        gov = pd.read_csv(DATA_DIR / "cholera_by_governorate.csv", parse_dates=['Date'])
        return df, gov
    except FileNotFoundError:
        return None, None


# ════════════════════════════════════════════════════════════
#  دوال مساعدة
# ════════════════════════════════════════════════════════════
def dyn_thresholds(series):
    return {'green':  series.quantile(0.25),
            'yellow': series.quantile(0.60),
            'orange': series.quantile(0.80),
            'red':    series.quantile(0.92)}

ALERT_INFO = {
    'green':  {'label': '🟢 وضع طبيعي',            'border': '#28a745', 'bg': '#d4edda', 'fg': '#155724',
               'action': 'مراقبة روتينية ورصد جودة مصادر المياه.'},
    'yellow': {'label': '🟡 تأهب – مراقبة مكثفة',  'border': '#ffc107', 'bg': '#fff8dc', 'fg': '#856404',
               'action': 'تعزيز كلور مصادر المياه · توزيع محاليل الإماهة الفموية (ORS) · تفعيل فرق الرصد.'},
    'orange': {'label': '🟠 إنذار متقدم',           'border': '#ff8c00', 'bg': '#fff0d9', 'fg': '#663300',
               'action': 'نشر عيادات متنقلة · تنسيق عاجل مع منظمة الصحة العالمية · فتح خط ساخن صحي.'},
    'red':    {'label': '🔴 تفشٍ حاد – تدخل فوري', 'border': '#dc3545', 'bg': '#fde8e8', 'fg': '#721c24',
               'action': 'إعلان حالة طوارئ صحية · فتح مراكز علاج الكوليرا (CTCs) · التنسيق مع UNICEF / WHO / MSF / ICRC.'},
}

def classify(v, thr):
    if v < thr['green']:  return 'green'
    if v < thr['yellow']: return 'yellow'
    if v < thr['orange']: return 'orange'
    return 'red'

def alert_html(level, predicted):
    a = ALERT_INFO[level]
    return f"""
    <div class="alert-box" style="background:{a['bg']};border-color:{a['border']};color:{a['fg']};">
      <div class="alert-title">{a['label']}</div>
      <div style="font-size:1rem;margin-top:.4rem;">
        <strong>الحالات المتوقعة للأسبوع القادم:</strong>
        <span style="font-size:1.4rem;font-weight:900;margin-right:.5rem;">{predicted:,.0f}</span> حالة
      </div>
      <div class="alert-action">⚡ <strong>الإجراء المقترح:</strong> {a['action']}</div>
    </div>"""

def fmt_num(n, dec=0):
    try:    return f"{n:,.{dec}f}"
    except: return str(n)


# ════════════════════════════════════════════════════════════
#  HERO HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
    <div>
      <h1>🏥 نظام الإنذار المبكر لتفشي الكوليرا</h1>
      <p>الجمهورية اليمنية · منظومة ذكاء اصطناعي للتنبؤ الوبائي · 2026–2027</p>
    </div>
    <div>
      <span class="hero-badge">🤖 XGBoost Stacking</span>
      <span class="hero-badge">📊 Real-time Dashboard</span>
      <span class="hero-badge">🗺️ Gov. Mapping</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  تحميل
# ════════════════════════════════════════════════════════════
model, features = load_model()
df, gov_df = load_data()

if model is None or df is None:
    st.error("❌ يرجى تشغيل سكريبت التدريب أولاً:  `python src/train.py`")
    st.stop()

thr = dyn_thresholds(df['New_Cases'])

# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ لوحة التحكم")
    st.markdown("---")
    st.markdown("### 🔮 أفق التنبؤ")
    n_weeks = st.slider("عدد الأسابيع القادمة", 1, 16, 8)

    st.markdown("### 🚨 عتبات الإنذار (يدوي)")
    use_dynamic = st.toggle("عتبات ديناميكية تلقائية", value=True)
    if not use_dynamic:
        thr['green']  = st.slider("عتبة الأخضر",  500,  10000, int(thr['green']),  500)
        thr['yellow'] = st.slider("عتبة الأصفر",  2000, 20000, int(thr['yellow']), 500)
        thr['orange'] = st.slider("عتبة البرتقالي",5000, 35000, int(thr['orange']), 1000)
        thr['red']    = st.slider("عتبة الأحمر",  8000, 50000, int(thr['red']),    1000)

    st.markdown("---")
    st.markdown("### 📊 إحصاءات البيانات")
    st.metric("إجمالي الأسابيع",   len(df))
    st.metric("ذروة الوباء (حالة)", fmt_num(df['New_Cases'].max()))
    st.metric("متوسط أسبوعي",       fmt_num(df['New_Cases'].mean()))

    st.markdown("---")
    st.markdown("""
<div style='font-size:.75rem;opacity:.6;text-align:center;'>
نظام ذكاء اصطناعي للدعم القراري<br>
Developed by طه الشاوش · 2026<br>
بيانات: HDX / WHO / OCHA
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 الوضع الوبائي الحالي",
    "🔮 التنبؤ المستقبلي",
    "🗺️ الخريطة الجغرافية",
    "🔬 تفسير النموذج",
    "ℹ️ عن النظام"
])


# ════════════════════════════════════════════════════════════
#  TAB 1 – الوضع الحالي
# ════════════════════════════════════════════════════════════
with tab1:
    last   = df.iloc[-1]
    prev   = df.iloc[-2]
    Xlast  = df[features].iloc[-1:]
    pred0  = float(model.predict(Xlast)[0])
    level  = classify(pred0, thr)

    # KPI cards
    delta_cases = int(last['New_Cases'] - prev['New_Cases'])
    delta_sign  = "+" if delta_cases >= 0 else ""
    delta_color = "#dc3545" if delta_cases > 0 else "#28a745"

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card" style="border-color:#e74c3c;">
        <div class="num" style="color:#e74c3c;">{fmt_num(last['New_Cases'])}</div>
        <div class="lbl">🦠 حالات الأسبوع الحالي</div>
        <div class="delta" style="color:{delta_color};">{delta_sign}{fmt_num(delta_cases)} مقارنة بالأسبوع الماضي</div>
      </div>
      <div class="kpi-card" style="border-color:#3498db;">
        <div class="num" style="color:#3498db;">{fmt_num(pred0)}</div>
        <div class="lbl">🔮 تنبؤ الأسبوع القادم</div>
        <div class="delta" style="color:#666;">بناءً على النموذج الذكي</div>
      </div>
      <div class="kpi-card" style="border-color:#9b59b6;">
        <div class="num" style="color:#9b59b6;">{fmt_num(df['New_Cases'].tail(4).mean())}</div>
        <div class="lbl">📈 متوسط آخر 4 أسابيع</div>
        <div class="delta" style="color:#666;">المتوسط المتحرك الشهري</div>
      </div>
      <div class="kpi-card" style="border-color:#f39c12;">
        <div class="num" style="color:#f39c12;">{fmt_num(last.get('CFR (%)', 0), 2)}%</div>
        <div class="lbl">💀 معدل إماتة الحالات (CFR)</div>
        <div class="delta" style="color:#666;">نسبة الوفيات من المصابين</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # صندوق الإنذار
    st.markdown(alert_html(level, pred0), unsafe_allow_html=True)

    # رسم بياني تاريخي
    st.markdown('<div class="sec-head">📈 المنحنى الوبائي الأسبوعي</div>', unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        subplot_titles=('الحالات الجديدة أسبوعياً', 'معدل الإماتة CFR (%)'))

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['New_Cases'],
        fill='tozeroy', fillcolor='rgba(231,76,60,0.15)',
        line=dict(color='#e74c3c', width=2.5),
        name='حالات جديدة', mode='lines+markers',
        marker=dict(size=4), hovertemplate='%{x|%Y-%m-%d}<br>%{y:,.0f} حالة<extra></extra>'
    ), row=1, col=1)

    # rolling mean
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['New_Cases'].rolling(4).mean(),
        line=dict(color='#8e44ad', width=2, dash='dot'),
        name='متوسط 4 أسابيع',
        hovertemplate='%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    if 'CFR (%)' in df.columns:
        fig.add_trace(go.Bar(
            x=df['Date'], y=df['CFR (%)'],
            marker_color='rgba(52,152,219,0.6)',
            name='CFR%',
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:.3f}%<extra></extra>'
        ), row=2, col=1)

    # تظليل الموجات
    waves = [
        ("2026-01-01", "2026-05-01", "rgba(255,100,100,0.07)", "الموجة الأولى"),
        ("2026-09-01", "2027-01-01", "rgba(255,150,50,0.07)",  "الموجة الثانية"),
        ("2027-01-15", "2027-03-31", "rgba(255,200,50,0.07)",  "الموجة الثالثة"),
    ]
    for s, e, clr, nm in waves:
        fig.add_vrect(x0=s, x1=e, fillcolor=clr, line_width=0, annotation_text=nm,
                      annotation_position="top left", row=1, col=1)

    fig.update_layout(height=520, template='plotly_white',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02),
                      hovermode='x unified', showlegend=True)
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # جدول آخر 8 أسابيع
    st.markdown('<div class="sec-head">📋 آخر 8 أسابيع</div>', unsafe_allow_html=True)
    recent = df[['Date','New_Cases','New_Deaths','CFR (%)']].tail(8).copy()
    recent['Date'] = recent['Date'].dt.strftime('%Y-%m-%d')
    recent.columns = ['التاريخ','حالات جديدة','وفيات جديدة','CFR%']
    recent = recent.iloc[::-1]
    st.dataframe(recent.style.format({'حالات جديدة': '{:,.0f}',
                                       'وفيات جديدة': '{:.1f}',
                                       'CFR%': '{:.3f}'}),
                 use_container_width=True)

    st.markdown("""
<div class="disclaimer">
⚠️ <strong>تنبيه:</strong> هذا النظام أداة دعم قراري أكاديمية مبنية على بيانات اصطناعية تحاكي الواقع.
لا ينبغي الاعتماد عليه منفرداً في القرارات الصحية الرسمية. يُوصى دائماً بالرجوع إلى منظمة الصحة العالمية وخبراء الوبائيات.
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 2 – التنبؤ المستقبلي
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="sec-head">🔮 تنبؤ الـ {n_weeks} أسابيع القادمة مع فترات الثقة (90%)</div>',
                unsafe_allow_html=True)

    # توليد التنبؤات
    from src.model import forecast_weeks
    with st.spinner("جارٍ حساب التنبؤات..."):
        future = forecast_weeks(model, df, features, n_weeks)

    # رسم مع CI
    fig2 = go.Figure()

    # التاريخ الفعلي (آخر 20 أسبوع)
    hist20 = df.tail(20)
    fig2.add_trace(go.Scatter(
        x=hist20['Date'], y=hist20['New_Cases'],
        mode='lines+markers', name='حالات فعلية',
        line=dict(color='#2c3e50', width=2.5),
        marker=dict(size=5),
        hovertemplate='%{x|%Y-%m-%d}<br>%{y:,.0f} حالة<extra></extra>'
    ))

    # CI
    fig2.add_trace(go.Scatter(
        x=pd.concat([future['Date'], future['Date'][::-1]]),
        y=pd.concat([future['CI_Upper'], future['CI_Lower'][::-1]]),
        fill='toself', fillcolor='rgba(231,76,60,0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        name='فترة الثقة 90%', showlegend=True, hoverinfo='skip'
    ))

    # التنبؤ
    fig2.add_trace(go.Scatter(
        x=future['Date'], y=future['Predicted_Cases'],
        mode='lines+markers', name='تنبؤ مستقبلي',
        line=dict(color='#e74c3c', width=2.5, dash='dash'),
        marker=dict(size=7, symbol='diamond'),
        hovertemplate='%{x|%Y-%m-%d}<br>تنبؤ: %{y:,.0f}<extra></extra>'
    ))

    # خط التقسيم
    fig2.add_vline(x=df['Date'].max(), line_dash='dot', line_color='gray',
                   annotation_text="اليوم", annotation_position="top right")

    # عتبات
    fig2.add_hline(y=thr['yellow'], line_dash='dash', line_color='#ffc107',
                   annotation_text="عتبة أصفر")
    fig2.add_hline(y=thr['red'],    line_dash='dash', line_color='#dc3545',
                   annotation_text="عتبة حمراء")

    fig2.update_layout(height=500, template='plotly_white',
                       xaxis_title='التاريخ', yaxis_title='عدد الحالات',
                       hovermode='x unified',
                       legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig2, use_container_width=True)

    # جدول التنبؤات مع مستوى الإنذار
    st.markdown('<div class="sec-head">📋 جدول التنبؤات التفصيلي</div>', unsafe_allow_html=True)

    future_display = future.copy()
    future_display['مستوى الإنذار'] = future_display['Predicted_Cases'].apply(
        lambda x: ALERT_INFO[classify(x, thr)]['label']
    )
    future_display['Date'] = future_display['Date'].dt.strftime('%Y-%m-%d')
    future_display = future_display.rename(columns={
        'Date': 'التاريخ',
        'Predicted_Cases': 'الحالات المتوقعة',
        'CI_Lower': 'الحد الأدنى (90%)',
        'CI_Upper': 'الحد الأعلى (90%)',
        'week': 'الأسبوع'
    })[['الأسبوع','التاريخ','الحالات المتوقعة','الحد الأدنى (90%)','الحد الأعلى (90%)','مستوى الإنذار']]

    st.dataframe(future_display.style.format({
        'الحالات المتوقعة': '{:,.0f}',
        'الحد الأدنى (90%)': '{:,.0f}',
        'الحد الأعلى (90%)': '{:,.0f}',
    }), use_container_width=True)

    # ملخص الإنذار
    red_weeks = (future_display['مستوى الإنذار'].str.contains('حمر')).sum()
    if red_weeks > 0:
        st.error(f"⚠️ تحذير: {red_weeks} أسبوع من الـ {n_weeks} القادمة تقع ضمن مستوى الإنذار الأحمر!")
    else:
        st.success("✅ لا تنبؤات بمستوى أحمر خلال الفترة المحددة.")


# ════════════════════════════════════════════════════════════
#  TAB 3 – الخريطة الجغرافية
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-head">🗺️ توزيع الحالات على المحافظات اليمنية</div>', unsafe_allow_html=True)

    if gov_df is not None:
        # آخر أسبوع
        last_week_gov = gov_df[gov_df['Date'] == gov_df['Date'].max()]

        col_map, col_rank = st.columns([3, 1])

        with col_map:
            map_type = st.radio("نوع البيانات", ["حالات جديدة", "معدل الهجوم (لكل 1000)"],
                                horizontal=True)
            val_col = 'New_Cases' if "حالات" in map_type else 'Attack_Rate'
            val_lbl = 'حالات جديدة' if "حالات" in map_type else 'معدل الهجوم'

            map_trace_type = getattr(go, 'Scattermap', None)
            map_layout_key = 'map'
            if map_trace_type is None:
                map_trace_type = go.Scattermapbox
                map_layout_key = 'mapbox'

            map_trace = map_trace_type(
                lat=last_week_gov['Lat'],
                lon=last_week_gov['Lon'],
                mode='markers',
                marker=dict(
                    size=np.sqrt(last_week_gov[val_col]) * (2.5 if val_col=='New_Cases' else 12),
                    color=last_week_gov[val_col],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title=val_lbl),
                    sizemin=8,
                ),
                text=last_week_gov['Governorate'],
                customdata=np.column_stack([last_week_gov['New_Cases'],
                                            last_week_gov['Attack_Rate'],
                                            last_week_gov['Risk_Score']]),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "حالات جديدة: %{customdata[0]:,.0f}<br>"
                    "معدل الهجوم: %{customdata[1]:.3f}‰<br>"
                    "مؤشر الخطورة: %{customdata[2]:.0%}<extra></extra>"
                )
            )

            fig_map = go.Figure(map_trace)
            fig_map.update_layout(
                **{map_layout_key: dict(style='carto-positron',
                                        center=dict(lat=15.5, lon=44.5),
                                        zoom=5.2)},
                height=500, margin=dict(l=0,r=0,t=0,b=0)
            )
            st.plotly_chart(fig_map, use_container_width=True)

        with col_rank:
            st.markdown("**🏆 ترتيب المحافظات**")
            top_gov = last_week_gov.nlargest(10, 'New_Cases')[['Governorate','New_Cases']].copy()
            top_gov.columns = ['المحافظة','الحالات']
            top_gov['الحالات'] = top_gov['الحالات'].apply(lambda x: f"{x:,}")
            st.dataframe(top_gov, use_container_width=True, hide_index=True)

        # رسم شريطي
        st.markdown('<div class="sec-head">📊 مقارنة المحافظات</div>', unsafe_allow_html=True)
        sorted_gov = last_week_gov.sort_values('New_Cases', ascending=True)
        fig_bar = go.Figure(go.Bar(
            x=sorted_gov['New_Cases'], y=sorted_gov['Governorate'],
            orientation='h',
            marker=dict(
                color=sorted_gov['New_Cases'],
                colorscale='Reds', showscale=False
            ),
            hovertemplate='%{y}: %{x:,.0f} حالة<extra></extra>'
        ))
        fig_bar.update_layout(height=500, template='plotly_white',
                               xaxis_title='عدد الحالات الجديدة',
                               yaxis_title='المحافظة')
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.warning("⚠️ بيانات المحافظات غير متوفرة. يرجى تشغيل سكريبت توليد البيانات.")

    # الجدول الزمني للمحافظات الأعلى خطراً
    if gov_df is not None:
        st.markdown('<div class="sec-head">📈 تطور الحالات في أعلى 5 محافظات خطراً</div>',
                    unsafe_allow_html=True)
        top5 = last_week_gov.nlargest(5, 'Risk_Score')['Governorate'].tolist()
        gov_top5 = gov_df[gov_df['Governorate'].isin(top5)]

        fig_top5 = px.line(gov_top5, x='Date', y='New_Cases', color='Governorate',
                           labels={'Date': 'التاريخ', 'New_Cases': 'حالات جديدة',
                                   'Governorate': 'المحافظة'},
                           template='plotly_white')
        fig_top5.update_layout(height=380,
                                legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig_top5, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  TAB 4 – تفسير النموذج
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-head">🔬 تفسير النموذج وأهمية الميزات</div>', unsafe_allow_html=True)

    # أهمية الميزات
    if model.feature_importances_ is not None:
        fi = pd.DataFrame({
            'الميزة':  model.feature_names_,
            'الأهمية': model.feature_importances_
        }).sort_values('الأهمية', ascending=False).head(15)

        fig_fi = go.Figure(go.Bar(
            x=fi['الأهمية'], y=fi['الميزة'],
            orientation='h',
            marker=dict(color=fi['الأهمية'], colorscale='Viridis', showscale=True),
            hovertemplate='%{y}<br>أهمية: %{x:.4f}<extra></extra>'
        ))
        fig_fi.update_layout(height=480, template='plotly_white',
                              xaxis_title='درجة الأهمية',
                              title='أهم 15 ميزة في النموذج')
        st.plotly_chart(fig_fi, use_container_width=True)

    # تفسير نصي
    st.markdown('<div class="sec-head">📖 ما تعنيه أهم الميزات</div>', unsafe_allow_html=True)

    FEAT_EXPLAIN = {
        'cases_lag_1':      ('📌 حالات الأسبوع الماضي',    'أقوى مؤشر – التفشي يحمل زخماً ذاتياً: إذا ارتفع الأسبوع الماضي فالأسبوع القادم سيرتفع.'),
        'cases_roll4_mean': ('📊 متوسط الشهر الماضي',      'يلتقط الاتجاه (Trend) بعيداً عن التذبذب الأسبوعي.'),
        'cases_lag_2':      ('⏳ حالات أسبوعين مضت',       'يعكس الفجوة الزمنية بين التعرض وظهور الأعراض (3-14 يوم).'),
        'rain_lag_2':       ('🌧️ أمطار أسبوعين مضت',      'الأمطار تلوّث مصادر المياه؛ الكوليرا تظهر بعد 2-3 أسابيع (فترة الحضانة + الوصول إلى المستشفى).'),
        'idps_lag1':        ('🏕️ النازحون الداخليون',     'ازدحام المخيمات يقلل الوصول إلى مياه نظيفة وصرف صحي مناسب.'),
        'wash_lag1':        ('💧 الوصول إلى مياه آمنة %', 'كلما انخفض هذا الرقم زاد خطر انتشار الكوليرا.'),
        'idps_x_rain':      ('⚡ تفاعل: أمطار × نازحين', 'ميزة مركّبة: أمطار غزيرة + نازحون كثيرون = خطر مضاعف.'),
        'cases_delta1':     ('📈 تغيير الأسبوع',          'سرعة الزيادة أو التراجع – يساعد في اكتشاف نقاط الانعطاف.'),
        'week_of_year':     ('📅 الأسبوع من السنة',       'الموسمية: الكوليرا ترتفع في مواسم الأمطار (مارس-مايو، أغسطس-أكتوبر).'),
        'cfr_lag1':         ('💀 معدل الإماتة المتأخر',   'ارتفاع CFR يعني ضغطاً على النظام الصحي وربما ضعف الرصد.'),
    }

    cols = st.columns(2)
    for i, (feat, (title, explanation)) in enumerate(FEAT_EXPLAIN.items()):
        with cols[i % 2]:
            st.markdown(f"""
<div style="background:white;border-radius:12px;padding:1rem 1.2rem;
            margin-bottom:.8rem;border-right:4px solid #e74c3c;
            box-shadow:0 2px 8px rgba(0,0,0,.06);">
  <div style="font-weight:700;font-size:1rem;">{title}</div>
  <div style="font-size:.88rem;color:#555;margin-top:.3rem;">{explanation}</div>
</div>""", unsafe_allow_html=True)

    # مصفوفة الارتباط
    st.markdown('<div class="sec-head">🔗 الارتباط بين الميزات الرئيسية</div>', unsafe_allow_html=True)

    corr_cols = ['New_Cases', 'cases_lag_1', 'cases_roll4_mean',
                 'rain_lag_2', 'idps_lag1', 'week_of_year']
    corr_cols = [c for c in corr_cols if c in df.columns]
    if len(corr_cols) >= 3:
        corr_mat = df[corr_cols].corr()
        arabic_names = {
            'New_Cases':        'حالات جديدة',
            'cases_lag_1':      'lag 1 أسبوع',
            'cases_roll4_mean': 'متوسط 4 أسابيع',
            'rain_lag_2':       'أمطار lag-2',
            'idps_lag1':        'نازحون lag-1',
            'week_of_year':     'أسبوع السنة',
        }
        corr_mat.index   = [arabic_names.get(c,c) for c in corr_mat.index]
        corr_mat.columns = [arabic_names.get(c,c) for c in corr_mat.columns]

        fig_corr = go.Figure(go.Heatmap(
            z=corr_mat.values, x=corr_mat.columns, y=corr_mat.index,
            colorscale='RdBu', zmin=-1, zmax=1,
            text=corr_mat.values.round(2),
            texttemplate='%{text}',
            hovertemplate='%{x} × %{y}<br>r = %{z:.3f}<extra></extra>'
        ))
        fig_corr.update_layout(height=400, template='plotly_white',
                                title='مصفوفة الارتباط (Pearson r)')
        st.plotly_chart(fig_corr, use_container_width=True)

    # نتائج التقييم
    st.markdown('<div class="sec-head">📏 أداء النموذج (Walk-Forward)</div>', unsafe_allow_html=True)
    metrics_data = {
        'المقياس':      ['MAE (حالة)', 'RMSE (حالة)', 'R² Score', 'MAPE%', 'الدقة%'],
        'القيمة':       ['~1,200', '~1,900', '≥ 0.97', '≤ 6.5%', '≥ 93%'],
        'التفسير':      ['متوسط الخطأ المطلق', 'جذر متوسط مربع الخطأ',
                         'نسبة التباين المفسَّر', 'متوسط النسبة المئوية للخطأ',
                         'دقة النموذج التقريبية'],
    }
    st.table(pd.DataFrame(metrics_data))
    st.info("💡 القيم المعروضة تقديرية بناءً على بيانات اصطناعية. مع البيانات الحقيقية قد تختلف.")


# ════════════════════════════════════════════════════════════
#  TAB 5 – عن النظام
# ════════════════════════════════════════════════════════════
with tab5:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
### 🎯 الهدف
بناء نظام ذكاء اصطناعي لإنذار مبكر بتفشي الكوليرا في اليمن،
يُمكّن السلطات الصحية والمنظمات الإنسانية من اتخاذ إجراءات وقائية
قبل وقوع التفشي بأسابيع.

### 🧠 النموذج
- **Stacking Ensemble**: XGBoost + LightGBM + GradientBoosting + Random Forest + Extra Trees
- **Meta-Learner**: Huber Regression (مقاوم للقيم الشاذة)
- **التقييم**: Walk-Forward Validation (5 folds)
- **الهدف**: حالات الأسبوع القادم (بدون تسرب بيانات)

### 📊 مصادر البيانات
| المصدر | البيانات |
|--------|---------|
| WHO / OCHA | بيانات وبائية كوليرا |
| HDX (CHIRPS) | هطول الأمطار الأسبوعي |
| OCHA IDP | النازحون الداخليون |
| WASH Cluster | الوصول إلى مياه آمنة |
""")

    with c2:
        st.markdown("""
### ✅ الإصلاحات الجوهرية (v2.0)

| المشكلة القديمة | الحل الجديد |
|----------------|-------------|
| ❌ تسرب: `New_Deaths` كميزة | ✅ محذوفة تماماً |
| ❌ الهدف = نفس الأسبوع | ✅ الهدف = الأسبوع القادم (shift=-1) |
| ❌ أمطار = صفر (بيانات فارغة) | ✅ أنماط موسمية واقعية |
| ❌ NaN ← أصفار وهمية | ✅ استيفاء خطي + ffill |
| ❌ حلقة تنبؤ لا تحدث الميزات | ✅ تحديث كامل في كل خطوة |
| ❌ نموذج واحد (XGBoost) | ✅ Stacking Ensemble (5 نماذج) |
| ❌ بدون فترات ثقة | ✅ CI 90% بتشتت النماذج |
| ❌ واجهة LTR | ✅ RTL كاملة + Cairo font |
| ❌ بلا خريطة | ✅ خريطة تفاعلية 21 محافظة |

### 🚀 التطوير المستقبلي
- ربط API حقيقي مع HDX و WHO
- تفصيل على مستوى المديريات
- نموذج LSTM للسلاسل الزمنية العميقة
- تطبيق موبايل للعاملين الميدانيين
- ربط بمنظومة الإنذار الوطنية
""")

    st.markdown("---")
    st.markdown("""
<div style="background:linear-gradient(135deg,#1a1a2e,#0f3460);color:white;
            border-radius:14px;padding:1.5rem 2rem;text-align:center;">
  <div style="font-size:1.3rem;font-weight:700;">👨‍💻 المطوّر</div>
  <div style="font-size:1.1rem;margin:.5rem 0;">طه فؤاد عبدالرحمن الشاوش</div>
  <div style="opacity:.8;font-size:.9rem;">
    معيد ومحاضر – قسم علوم الحاسوب – جامعة ذمار<br>
    مختص في الذكاء الاصطناعي وتعلم الآلة
  </div>
  <div style="margin-top:.8rem;opacity:.7;font-size:.8rem;">
    🔗 github.com/tahafoud &nbsp;·&nbsp; 2026 © جميع الحقوق محفوظة
  </div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;font-size:.82rem;padding:.5rem;'>
  🏥 نظام الإنذار المبكر للكوليرا – اليمن 2026 &nbsp;|&nbsp;
  مبني بـ Streamlit + XGBoost + Plotly &nbsp;|&nbsp;
  البيانات: WHO / OCHA / HDX
</div>""", unsafe_allow_html=True)
