# gdp_arimax_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="ARIMAX-прогноз ВВП", page_icon="📊")

# ─────────────────── 1. Исторические (условно-реалистичные) данные ──────────
hist = pd.DataFrame({
    "year":  np.arange(2014, 2024),
    "gdp":   [79, 83, 86, 92, 104, 111, 107, 130, 150, 160],   # трлн ₽
    "oil":   [99, 53, 43, 54,  71,  64,  41,  71, 100,  80],   # $/bbl
    "usd":   [37, 61, 67, 59,  62,  65,  73,  74,  68,  90],   # ₽/$
    "rate":  [9.0,12.0,10.0,8.5, 7.5, 6.3, 5.5, 8.5, 11.0,12.5]# %
})

y_train = hist["gdp"]
X_train = hist[["oil", "usd", "rate"]]

# ─────────────────── 2. Обучаем ARIMAX (SARIMAX с exog) ─────────────────────
# Для демонстрации берём порядок (1,1,0). В реальной задаче подбирают по AIC/BIC.
model = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(1, 1, 0),           # (p,d,q)
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# ─────────────────── 3. Ввод сценарных параметров ───────────────────────────
st.sidebar.header("Сценарные параметры")
oil_price = st.sidebar.slider("Цена нефти ($/барр.)", 30.0, 140.0, 80.0, 1.0)
usd_rate  = st.sidebar.slider("Курс USD/RUB (₽)",     50.0, 150.0, 90.0, 1.0)
key_rate  = st.sidebar.slider("Ключевая ставка ЦБ (%)", 2.0, 25.0, 12.0, 0.5)
horizon   = st.sidebar.slider("Горизонт прогноза, лет", 1, 15, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Модель**: SARIMAX(1,1,0) + exog (Oil, USD, Rate)")

# ─────────────────── 4. Формируем exog-матрицу для прогноза ─────────────────
future_years = np.arange(2024, 2024 + horizon)
X_future = np.column_stack([
    np.full(horizon, oil_price),
    np.full(horizon, usd_rate),
    np.full(horizon, key_rate)
])

# ─────────────────── 5. Прогноз ARIMAX ‒ точки и интервалы доверия ───────────
forecast_res = model.get_forecast(steps=horizon, exog=X_future)
gdp_pred = forecast_res.predicted_mean
conf_int = forecast_res.conf_int(alpha=0.2)   # 80 % ДИ

df_future = pd.DataFrame({
    "year": future_years,
    "gdp": gdp_pred,
    "low": conf_int["lower gdp"].values,
    "high": conf_int["upper gdp"].values
})

# ─────────────────── 6. Plotly-график истор. + прогноз ───────────────────────
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hist["year"], y=hist["gdp"],
    mode="lines+markers",
    name="История",
    line=dict(color="royalblue")
))

fig.add_trace(go.Scatter(
    x=df_future["year"], y=df_future["gdp"],
    mode="lines+markers",
    name="Прогноз (точка)",
    line=dict(color="firebrick", dash="dash")
))

fig.add_trace(go.Scatter(
    x=np.concatenate([df_future["year"], df_future["year"][::-1]]),
    y=np.concatenate([df_future["high"], df_future["low"][::-1]]),
    fill="toself",
    fillcolor="rgba(255,0,0,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip",
    name="80% ДИ"
))

fig.update_layout(
    title="Номинальный ВВП России: история и сценарный ARIMAX-прогноз",
    xaxis_title="Год",
    yaxis_title="ВВП, трлн ₽",
    template="plotly_white",
    hovermode="x unified"
)

st.title("📊 ARIMAX-прогноз ВВП России")
st.plotly_chart(fig, use_container_width=True)

# ─────────────────── 7. Метрики и таблица ────────────────────────────────────
delta = (df_future["gdp"].iloc[-1] - hist["gdp"].iloc[-1]) / hist["gdp"].iloc[-1] * 100
st.metric(
    label=f"ВВП в {future_years[-1]} г.",
    value=f"{df_future['gdp'].iloc[-1]:,.0f} трлн ₽",
    delta=f"{delta:+.1f}% к 2023"
)

with st.expander("Таблица значений прогноза"):
    full = pd.concat([hist[["year","gdp"]], df_future[["year","gdp"]]], ignore_index=True)
    st.dataframe(full.rename(columns={"year":"Год","gdp":"ВВП, трлн ₽"}).style.format("{:,.2f}"), height=300)

st.caption("⚠️ Демонстрационный пример: данные синтетические, параметры ARIMA не подбирались по AIC/BIC; результат не является официальным макроэкономическим прогнозом.")
