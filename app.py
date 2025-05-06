import streamlit as st

import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
import numpy as np

st.set_page_config(layout="wide")

# ── inject custom CSS ────────────────────────────────────────────────────
css = """
<style>
/* 1️⃣  reduce global page padding */
.stApp {
    padding: 0rem 1rem 2rem;     /* top right/left bottom */
}

/* 2️⃣  keep title from clipping under top browser bar */
h1 { 
    margin-top: 1.2rem;          /* push title a bit down */
}

/* 3️⃣  remove unnecessary side gutter (Streamlit adds it for sidebar) */
.css-18e3th9 { padding: 0; }     /* main block */
.css-1d391kg { padding: 0; }     /* wide-mode block */
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data loader (cached) 
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    df_raw = pd.read_csv(path)

    # Separate IVV and others
    df_ivv = df_raw[df_raw["ETF"] == "IVV"].copy()
    df_others = df_raw[df_raw["ETF"] != "IVV"].copy()

    # Try standard parsing for IVV
    df_ivv["Date"] = pd.to_datetime(df_ivv["Date"], errors="coerce", format="%Y-%m-%d")
    
    # If still all NaT, try Excel-style origin fallback
    if df_ivv["Date"].isna().sum() == len(df_ivv):
        df_ivv["Date"] = pd.to_datetime(df_ivv["Date"], errors="coerce", unit="D", origin="1899-12-30")

    # Parse rest normally
    df_others["Date"] = pd.to_datetime(df_others["Date"], errors="coerce")

    # Recombine
    df = pd.concat([df_ivv, df_others], ignore_index=True)
    df = df.sort_values(["ETF", "Date"])

    # Engineer fields
    df["Daily_Return"] = df.groupby("ETF")["Close"].pct_change()
    df["Cumulative_Return"] = df.groupby("ETF")["Daily_Return"].transform(lambda r: (1 + r).cumprod() - 1)
    df["MA_200"] = df.groupby("ETF")["Adj Close"].transform(lambda s: s.rolling(200, min_periods=1).mean())
    df["Growth10k"] = df.groupby("ETF")["Daily_Return"].transform(lambda r: (1 + r).cumprod() * 10_000)
    peak = df.groupby("ETF")["Adj Close"].transform("cummax")
    df["Drawdown"] = df["Adj Close"] / peak - 1
    df["Year"] = df["Date"].dt.year

    return df

csv_path = Path(__file__).parent / "combined_etf_data (1).csv"   # ensure file name

# manual refresh
if st.button("🔄 Refresh data"):
    load_data.clear()
data = load_data(csv_path)

st.title("ETF Observatory – ETF Performance Dashboard: Risk, Return, and Investment Insights Across Global Markets")

st.caption("A comprehensive interactive dashboard for analyzing historical performance, volatility, drawdowns, and risk-adjusted returns across major global ETFs.")


col1, col2 = st.columns(2)

with col1:
    st.subheader("1. ETF Universe")
    st.markdown(
        """
| Ticker | Fund Name                                  |
|:------:|:-------------------------------------------|
| **EEM** | iShares MSCI Emerging Markets             |
| **EFA** | iShares MSCI EAFE (Europe, Australasia & Far East) |
| **IVV** | iShares Core S&P 500                      |
| **IWM** | iShares Russell 2000                      |
| **TLT** | iShares 20+ Year Treasury Bond Fund       |
""",
        unsafe_allow_html=True,
    )

with col2:
    st.subheader("2. Core Metrics & Notation")
    st.markdown(
        """
| Symbol               | Meaning                                                                                                              |
|:--------------------:|:---------------------------------------------------------------------------------------------------------------------|
| `rₜ`                 | **Daily Return:** `(Closeₜ / Closeₜ₋₁) - 1`                                                                          |
| **Cumulative Return**| Compound growth over time: `(1+r₁) × (1+r₂) × … – 1`                                                                  |
| `μ` (mu)             | Average return over your selected period                                                                             |
| `σ` (sigma)          | Volatility (standard deviation of returns)                                                                           |
| **Sharpe Ratio**     | `(μ – r_f) / σ`, where `r_f` is the risk-free rate                                                                   |
| **Max Drawdown**     | Largest peak-to-trough percentage decline in **Adj Close**                                                            |
| **CAGR**             | Compound Annual Growth Rate – annualised return assuming reinvestment                                                 |
""",
        unsafe_allow_html=True,
    )

st.subheader("3. What we can do")
st.markdown(
    """
- Explore miniature price charts for each ETF and quickly spot trends.
- Filter by time window, sampling frequency (daily/monthly/yearly), and custom risk-free rate.
- Simulate the growth of any initial investment ($1 – $100) and see its current value.
- Compare ETFs on risk vs return with an interactive scatter and a dedicated volatility gauge.
- Dive into monthly performance with heatmaps and drill-down bar charts by year.
- Snapshot annual metrics in a radar chart and a clean, formatted table.
    """
)

st.subheader("ETF Dataset")
st.dataframe(data)
with st.expander("About the Dataset"):
    st.markdown("""
This dataset provides a daily historical record of five major **Exchange-Traded Funds (ETFs)** across global markets, capturing their performance from inception to the present. It serves as the foundational data for all visualizations and calculations in this dashboard.

### Key Features:
- **Date**: Trading date for each observation.
- **Adj Close / Close / Open / High / Low**: Standard price fields, where `Adj Close` accounts for splits and dividends.
- **Volume**: Total trading volume for the day.
- **ETF**: The ETF ticker symbol, identifying the fund:
    - `EEM`: iShares MSCI Emerging Markets
    - `EFA`: iShares MSCI EAFE (Europe, Australasia & Far East)
    - `IVV`: iShares Core S&P 500
    - `IWM`: iShares Russell 2000
    - `TLT`: iShares 20+ Year Treasury Bond Fund

### Engineered Fields:
- **Daily_Return**: Percentage change from previous close.
- **Cumulative_Return**: Compounded return from the start of the period.
- **Rolling_30d_Vol**: 30-day rolling volatility (standard deviation).
- **Month / Year**: Date-based partitioning for visual summaries.
- **MA_200**: 200-day moving average of the Adjusted Close.
- **Growth10k**: Simulated growth of a $10,000 investment over time.
- **Drawdown**: Maximum decline from previous price peaks.

These engineered metrics allow us to evaluate both **short-term dynamics** and **long-term investment performance**, enabling side-by-side risk-return comparisons, volatility analysis, and simulated portfolio growth.

This dataset has been cleaned, standardized, and transformed for interactive exploration within this dashboard.
    """)



# =============================================================================
# ROW 0 ▸ four mini closing‑price charts (one per ETF)
# =============================================================================
st.subheader("Mini Close‑Price Trends")

# choose which ETFs to show as minis (first four by default)
etfs_to_show = data["ETF"].unique()          #  ← show every ticker, not just [:4]
cols = st.columns(len(etfs_to_show), gap="medium")   # narrow gap (Streamlit ≥1.24)

for col, etf in zip(cols, etfs_to_show):
    mini_df = data[data["ETF"] == etf]
    fig = px.line(
        mini_df,
        x="Date",
        y="Close",
        title=etf,
        template="plotly_dark",
        height=250,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        xaxis=dict(title=None),
        yaxis=dict(title=None),
    )
    col.plotly_chart(fig, use_container_width=True)

    with col.expander(f" {etf} Insight"):
        if etf == "EEM":
            st.markdown("""
**EEM (iShares MSCI Emerging Markets)** tracks stocks from countries like China, Brazil, and India.  
This chart shows volatile performance with noticeable peaks around 2007 and 2021, reflecting the sensitivity of emerging markets to global risk sentiment.
            """)
        elif etf == "EFA":
            st.markdown("""
**EFA (iShares MSCI EAFE)** captures developed markets outside North America — Europe, Australasia, and the Far East.  
The chart shows long-term recovery post-2008, but relatively flatter growth compared to U.S.-based ETFs.
            """)
        elif etf == "IVV":
            st.markdown("""
**IVV (iShares Core S&P 500)** reflects the U.S. large-cap equity market.  
A consistent upward trend with strong growth after 2012, underlining the dominance of U.S. tech and large-cap sectors.
            """)
        elif etf == "IWM":
            st.markdown("""
**IWM (iShares Russell 2000)** covers small-cap U.S. stocks.  
More fluctuation than IVV, highlighting sensitivity to domestic economic conditions and investor sentiment.
            """)
        elif etf == "TLT":
            st.markdown("""
**TLT (iShares 20+ Year Treasury)** tracks long-term U.S. government bonds.  
You can observe spikes during periods of market stress (e.g., 2020) followed by declines as interest rates rise.
            """)

# =============================================================================
# ROW 1 ▸ full‑width Plotly overviews
# =============================================================================
st.subheader("Closing Prices (All ETFs)")
with st.expander("Interpretation: Price Trends and Window Metrics"):
    st.markdown("""
This section presents the historical **closing prices** of selected ETFs over a custom time range. You can choose from:

- **1Y** — captures recent market conditions and volatility
- **5Y** — balances short-term noise with longer trends
- **MAX** — shows long-term compounded growth and resilience

By adjusting this window, you can uncover how different ETFs behave across economic cycles — from recent turbulence to multi-decade trajectories.

### What We’re Comparing:
Each ETF line reflects raw market prices — not adjusted for dividends — which helps visualize:
- Relative growth rates
- Crash/recovery patterns
- Stability or cyclicality

### Understanding the Metrics Below:
We compute essential summary stats for the selected window:

- **Best Performer**: The ETF with the highest return over the window.
- **Avg Period Return**: Mean return across all ETFs during the selected range.
- **Avg Annual Volatility**: Standard deviation of returns, scaled to yearly — higher means more fluctuation.
- **Avg Max Drawdown**: The average peak-to-trough decline for each ETF — shows downside exposure during crashes.

These metrics help distinguish **return efficiency**, **risk behavior**, and **drawdown resilience**, providing a well-rounded snapshot of ETF performance over your chosen timeframe.
    """)


# ① dropdown for year‑span
window = st.selectbox("Show period:",
                      options=["1 Y", "5 Y", "MAX"],
                      index=2,            # default = MAX
                      key="price_range")

# ② compute slice
min_date = data["Date"].min()
if window == "1 Y":
    end_date = min_date + pd.DateOffset(years=1)
    subset = data[(data["Date"] >= min_date) & (data["Date"] <= end_date)]
elif window == "5 Y":
    end_date = min_date + pd.DateOffset(years=5)
    subset = data[(data["Date"] >= min_date) & (data["Date"] <= end_date)]
else:  # MAX
    subset = data.copy()

# ③ plot
st.plotly_chart(
    px.line(
        subset,
        x="Date",
        y="Close",
        color="ETF",
        title=f"ETF Closing Prices — {window}",
        template="plotly_white",
    ),
    use_container_width=True,
)

summary = (
    subset.groupby("ETF")
    .agg(
        first_close=("Close", "first"),
        last_close=("Close", "last"),
        daily_std=("Daily_Return", "std"),
    )
)
summary["Period_Return"] = summary["last_close"] / summary["first_close"] - 1
summary["Ann_Vol"] = summary["daily_std"] * (252**0.5)

# Max‑drawdown requires rolling cumulative max
def max_dd(group):
    peak = group["Adj Close"].cummax()
    dd = group["Adj Close"] / peak - 1
    return dd.min()

maxdd = subset.groupby("ETF").apply(max_dd).rename("Max_Drawdown")
summary = summary.join(maxdd)

# Which ETF won?
winner = summary["Period_Return"].idxmax()
winner_ret = summary.loc[winner, "Period_Return"]

# Show metrics side‑by‑side
st.subheader("Key Metrics for Selected Window")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Best Performer", winner, f"{winner_ret:.2%}")
m2.metric("Avg Period Return", f"{summary['Period_Return'].mean():.2%}")
m3.metric("Avg Ann. Volatility", f"{summary['Ann_Vol'].mean():.2%}")
m4.metric("Avg Max Drawdown", f"{summary['Max_Drawdown'].mean():.2%}")


st.subheader("Cumulative Returns (All ETFs)")
with st.expander("Interpretation: Cumulative Returns and Long-Term Growth"):
    st.markdown("""
This section visualizes the **cumulative return** for each ETF from 2000 up to the year selected in the slider. It highlights how an investment compounds over time, assuming all returns are reinvested.

### What We’re Seeing:
- **Cumulative Return** tracks total growth:  
  \[(1 + r₁) × (1 + r₂) × ... × (1 + rₙ) - 1\]  
  It's a strong signal of long-term performance, ignoring short-term noise.
  
- **Best Performing ETF**: The asset that achieved the highest return over the selected range.
- **Underperforming ETF**: The one with the lowest return.
- **Average Cumulative Return**: Mean return across all ETFs for the selected period.
- **CAGR (Compound Annual Growth Rate)**: Indicates average yearly return assuming reinvestment. Smooths out volatility and makes comparisons fair across timeframes.

### Why This Matters:
Cumulative returns reveal which ETFs have sustained growth and resilience over multiple market cycles. For example:
- **IWM** (U.S. small-cap) shows strong compounding despite volatility.
- **TLT** (U.S. Treasury bonds) may lag in returns, but often acts defensively during market stress.

You can move the slider to examine how different years (e.g., pre-2008, post-COVID) affect overall ETF rankings and long-term wealth accumulation.
    """)

# ① End‑year slider: from first year to last year in the dataset
yr_min, yr_max = int(data["Year"].min()), int(data["Year"].max())
end_year = st.slider(
    "Display data up to year:",
    yr_min,
    yr_max,
    yr_max,
    key="cum_slider",
)

# ② Subset the data: keep rows up to 31‑Dec of the chosen year
window_mask = data["Date"] <= pd.Timestamp(year=end_year, month=12, day=31)
cur = data[window_mask]

# ③ Plot cumulative returns
fig_cum = px.line(
    cur,
    x="Date",
    y="Cumulative_Return",
    color="ETF",
    title=f"Cumulative Returns (up to {end_year})",
    template="plotly_white",
)
st.plotly_chart(fig_cum, use_container_width=True)

# ④ Live metrics ----------------------------------------------------------
last_day = cur["Date"].max()
last_vals = (
    cur[cur["Date"] == last_day]
    .set_index("ETF")["Cumulative_Return"]
)

best_etf = last_vals.idxmax()
worst_etf = last_vals.idxmin()
best_ret = last_vals.max()
worst_ret = last_vals.min()
avg_ret = last_vals.mean()

years_span = (last_day - cur["Date"].min()).days / 365.25
cagr_best = (1 + best_ret) ** (1 / years_span) - 1

m1, m2, m3, m4 = st.columns(4)
m1.metric("Best Performing ETF", best_etf, f"{best_ret:.2%}")
m2.metric("Underperforming ETF", worst_etf, f"{worst_ret:.2%}")
m3.metric("Avg Cum Return", f"{avg_ret:.2%}")
m4.metric("CAGR (Best)", f"{cagr_best:.2%}")

# =============================================================================
# ROW 2 ▸ Growth of $10 000
# =============================================================================
st.header("Growth of an Investment")
with st.expander("About Growth of an Investment"):
    st.markdown("""
This section visualizes how a one-time investment would have grown over time in a selected ETF. Use the dropdown to choose among **EEM**, **EFA**, **IVV**, **IWM**, or **TLT**, and set your initial investment using the slider.

The chart reflects the compounding effect of returns, based on daily price changes. It offers a visual and quantitative sense of how each fund performs over the long term—highlighting both growth potential and volatility.

Key points:
- The graph tracks **portfolio value** over time, simulating reinvested returns.
- The final metric below the chart displays the **current value** of your selected investment.
- Ideal for comparing historical performance and assessing how different asset classes react to market cycles.

This tool is especially useful for exploring how different ETFs align with your long-term financial goals and risk tolerance.
    """)

# ① choose ETF
choice = st.selectbox("Choose ETF", data["ETF"].unique())

# ② choose starting amount (slider $1 – $100)
start_amt = st.slider("Initial Amount ($)", 1, 100, 10, step=1)

# ③ build dataframe for the chosen ETF
gdf = data[data["ETF"] == choice].copy()
gdf["Wealth"] = (1 + gdf["Cumulative_Return"]) * start_amt

# ④ plot with Altair
growth_chart = (
    alt.Chart(gdf)
    .mark_line(strokeWidth=2, color="#1f77b4")
    .encode(
        x="Date:T",
        y=alt.Y("Wealth:Q", title="Portfolio Value ($)"),
        tooltip=["Date:T", alt.Tooltip("Wealth:Q", format="$.2f")],
    )
    .properties(height=350, width=900)
)
st.altair_chart(growth_chart, use_container_width=True)

# ⑤ show current value as a metric
final_val = gdf["Wealth"].iloc[-1]
st.metric(
    label=f"Value today of ${start_amt} invested in {choice}",
    value=f"${final_val:,.2f}",
)

# =============================================================================
# ROW 3 ▸ Risk vs Return scatter
# =============================================================================
# =============================================================================
# ROW 3 ▸ Risk vs Return
# =============================================================================
import numpy as np
import altair as alt
import plotly.graph_objects as go

st.header("Risk vs Return")
with st.expander("About Risk vs Return Analysis"):
    st.markdown("""
This section provides a detailed comparison of **risk-adjusted performance** for various ETFs using standard financial metrics like **Volatility**, **Return**, and the **Sharpe Ratio**. Let’s break it down:

---

### Controls (Left Panel)

- **Year Window**:  
  Adjust the date range to select the historical time period for analysis. This helps to study how ETFs performed in different market cycles (e.g., bull markets, crashes, recoveries).

- **Frequency (Daily / Monthly / Yearly)**:  
  Choose the time scale for return and volatility calculation.  
  - **Daily**: Most granular but noisier.
  - **Monthly**: Good balance between detail and smoothness.
  - **Yearly**: Smoothest, best for long-term performance view.

- **Risk-Free Rate (%)**:  
  Used in Sharpe Ratio calculations. This represents the return from a "riskless" investment like U.S. Treasuries. Adjust this slider based on macroeconomic assumptions or personal benchmarks.

- **Highlight ETFs**:  
  Optionally emphasize specific ETFs in the scatter plot. Helpful for visual isolation of desired funds.

- **ETF to Gauge**:  
  Select an ETF to inspect its **Volatility Gauge**, Sharpe Ratio, and annualized return in more detail.

---

### Scatter Plot (Center)

Each dot represents an ETF, plotted using:
- **X-Axis (Volatility σ)**:  
  Annualized standard deviation of returns — a measure of risk or price fluctuation.
- **Y-Axis (Return μ)**:  
  Annualized average return during the selected period.

This visual shows the risk-return tradeoff. Ideally, we want funds in the **top-left quadrant** — high return, low volatility.

---

### Volatility Gauge (Right Panel)

- This semi-circular gauge shows the **annualized volatility** of the ETF selected in the dropdown.
- Color zones:
  - Green = Low volatility (more stable)
  - Yellow = Medium volatility
  - Red = High volatility (riskier)
- The black marker shows where your selected ETF stands.

This helps investors **visually assess the ETF’s stability** and compare it to historical volatility thresholds.

---

### Risk–Return Snapshot (Below Gauge)

This dynamic text highlights:
- **Risk Category** (Low/Med/High) based on volatility
- Arrow indicating return direction (↑ for positive, ↓ for negative)
- **Annualized Return** (percentage return normalized to yearly scale)

Together, this gives a clean, instant overview of risk vs reward for a specific ETF.

---

### Sharpe Ratio Metrics (Bottom Panel)

- **Best Sharpe**: ETF with the highest Sharpe Ratio — offers the most efficient return per unit of risk.
- **Worst Sharpe**: ETF with the lowest Sharpe — least efficient.
- **Avg Sharpe**: Mean Sharpe Ratio across all ETFs.
- **Quadrant Metrics**:
  - ⬆ Return / ⬇ Vol: ETFs with high return and low volatility — desirable zone.
  - ⬇ Return / ⬆ Vol: Undesirable zone — low returns with high risk.

---

### Why this matters?

This section lets users:
- Explore **risk-adjusted return** in a quantitative and visual way.
- Compare ETFs not just on returns, but on **efficiency**.
- Make informed decisions aligned with their **risk tolerance** and **investment horizon**.

Use this to evaluate if a higher return is worth the additional volatility, and discover which ETFs strike the best balance.

    """)


# 3‑column layout: controls | scatter | gauge
ctrl_col, scat_col, gauge_col = st.columns([1, 2.5, 1.2])

# ─────────────────────────  controls  ──────────────────────────
with ctrl_col:
    start_year, end_year = st.slider(
        "Year window:",
        int(data.Year.min()), int(data.Year.max()),
        (int(data.Year.min()), int(data.Year.max())),
        step=1, key="risk_years"
    )

    freq = st.radio(
        "Frequency:", ["Daily", "Monthly", "Yearly"],
        index=0, horizontal=True
    )

    rf = st.slider(
        "Risk‑free rate (%)", 0.0, 5.0, 0.0, 0.25,
        key="rf"
    )

    hi_etfs = st.multiselect(
        "Highlight ETFs (optional)",
        options=data["ETF"].unique(),
        default=[]
    )

    # NEW — gauge ETF selector lives with the other controls
    g_etf = st.selectbox(
        "ETF to gauge:",
        options=data["ETF"].unique(),
        key="gauge_etf"
    )

# ────────────────────  data window & stats  ───────────────────
mask = (
    (data["Date"] >= f"{start_year}-01-01") &
    (data["Date"] <= f"{end_year}-12-31")
)
win = data[mask].copy()

if freq == "Monthly":
    win["Period_Return"] = win.groupby("ETF")["Close"].pct_change()
    win = (
        win.set_index("Date")
           .groupby("ETF")["Period_Return"]
           .resample("M").sum()
           .unstack("ETF").stack()
           .reset_index()
           .rename(columns={0: "Period_Return"})
    )
elif freq == "Yearly":
    win["Period_Return"] = win.groupby("ETF")["Close"].pct_change()
    win = (
        win.set_index("Date")
           .groupby("ETF")["Period_Return"]
           .resample("Y").sum()
           .unstack("ETF").stack()
           .reset_index()
           .rename(columns={0: "Period_Return"})
    )
else:                                # Daily
    win["Period_Return"] = win["Daily_Return"]

stats = (
    win.groupby("ETF")["Period_Return"]
       .agg(mu="mean", sigma="std")
       .assign(
           ann_mu    = lambda d: d.mu    * (252 if freq=="Daily" else 12 if freq=="Monthly" else 1),
           ann_sigma = lambda d: d.sigma * (np.sqrt(252) if freq=="Daily" else np.sqrt(12) if freq=="Monthly" else 1),
       )
)
stats["Sharpe"] = (stats.ann_mu - rf/100) / stats.ann_sigma

# quadrant counts
median_sigma = stats.ann_sigma.median()
quad = (stats.ann_mu > 0).astype(str) + "-" + (stats.ann_sigma > median_sigma).astype(str)
qcounts = quad.value_counts().reindex(
    ["True-False","True-True","False-False","False-True"], fill_value=0
)
names = {
    "True-False": "⬆ Return / ⬇ Vol",
    "True-True" : "⬆ Return / ⬆ Vol",
    "False-False":"⬇ Return / ⬇ Vol",
    "False-True" : "⬇ Return / ⬆ Vol",
}

plot_df = stats.reset_index()
if hi_etfs:
    plot_df = plot_df[plot_df.ETF.isin(hi_etfs)]

# ─────────────────────  scatter plot  ──────────────────────
with scat_col:
    scatter = (
        alt.Chart(plot_df)
           .mark_circle(size=200)
           .encode(
               x=alt.X("ann_sigma:Q", title="Volatility σ (annualised)"),
               y=alt.Y("ann_mu:Q",    title="Return μ (annualised)"),
               color="ETF",
               tooltip=[
                   "ETF",
                   alt.Tooltip("ann_mu:Q",    title="Ann Return", format=".2%"),
                   alt.Tooltip("ann_sigma:Q", title="Ann Vol",    format=".2%"),
                   alt.Tooltip("Sharpe:Q",    format=".2f"),
               ],
           )
           .properties(height=420)
    )
    st.altair_chart(scatter, use_container_width=True)

# ─────────────────────  gauge only  ───────────────────────
with gauge_col:
    st.subheader("Volatility Gauge")

    # thresholds
    vals        = stats.ann_sigma.sort_values()
    low, high   = np.percentile(vals, [33, 66])
    val         = stats.loc[g_etf, "ann_sigma"]

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"valueformat": ".3f"},
        title={"text": f"Ann σ for {g_etf}"},
        gauge={
            "axis":   {"range": [0, vals.max()*1.1]},
            "steps": [
                {"range":[0,    low],  "color":"#4CAF50"},
                {"range":[low, high],  "color":"#FFC107"},
                {"range":[high, vals.max()*1.1], "color":"#F44336"},
            ],
            "threshold": {
                "value":     val,
                "line":      {"color":"black","width":4},
                "thickness": 0.75,
            },
        },
    ))
    fig_g.update_layout(margin=dict(l=0,r=0,t=25,b=0), height=420)
    st.plotly_chart(fig_g, use_container_width=True)

    level     = "Low Risk" if val < low else "Med Risk" if val < high else "High Risk"
    arrow     = "⬆️" if stats.loc[g_etf,"ann_mu"] > 0 else "⬇️"
    ann_ret   = stats.loc[g_etf,"ann_mu"]
    st.metric(
        "Risk–Return Snapshot",
        f"{level} / {arrow} {ann_ret:.2%} ann return"
    )

# ────────────────────  Sharpe metric strip  ───────────────────
best  = stats.Sharpe.idxmax()
worst = stats.Sharpe.idxmin()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Best Sharpe",  best,  f"{stats.loc[best,'Sharpe']:.2f}")
m2.metric("Worst Sharpe", worst, f"{stats.loc[worst,'Sharpe']:.2f}")
m3.metric("Avg Sharpe",          f"{stats.Sharpe.mean():.2f}")
m4.metric(names["True-False"], int(qcounts["True-False"]))
m5.metric(names["False-True"], int(qcounts["False-True"]))


# =============================================================================
# ROW 4 ▸ Calendar heatmap
# =============================================================================
import calendar

# ── Monthly x Year Heatmap ────────────────────────────────────────────────
st.header("Monthly Returns Heatmap")
with st.expander("About Monthly Returns Visualization"):
    st.markdown("""
This section provides a detailed month-by-month breakdown of ETF performance over multiple years using two complementary visualizations.

---

### Monthly Returns Heatmap (Top Chart)

- **What It Shows**:  
  A heatmap matrix where:
  - **Rows** represent years (within the selected range)
  - **Columns** represent months (January to December)
  - **Cell colors** represent the average daily return for each ETF in that month

- **Color Coding**:
  - Blue tones indicate positive average returns (darker blues signal stronger gains)
  - Red tones indicate negative returns
  - Near-white means close to zero or neutral returns

- **Purpose**:  
  Enables quick visual detection of:
  - Seasonal return trends (e.g., consistent December rallies)
  - Crisis years or prolonged underperformance
  - Monthly volatility patterns across the years

- **User Controls**:
  - ETF Selector: Choose the fund to analyze
  - Year Range Slider: Narrow or expand the time horizon

---

### Monthly Returns Breakdown (Bottom Bar Chart)

- **What It Shows**:  
  A bar chart for a selected year from the heatmap:
  - **X-axis** shows months
  - **Y-axis** displays average daily returns in each month

- **Purpose**:
  - Offers a zoomed-in view of monthly behavior within a specific year
  - Clarifies which months drove annual performance
  - Highlights inconsistencies or standout months in otherwise stable years

---

### How the Two Charts Work Together

- The bar chart is linked to the heatmap:
  - Selecting a year updates the bar chart to show monthly detail
  - Provides seamless transition from a multi-year overview to annual resolution

- **Example**:
  - A dark red October cell in the heatmap for 2008 suggests a drawdown
  - Select 2008 in the bar chart to confirm that October was indeed a steep decline

---

### Significance

Understanding ETF performance at the monthly level is useful for:
- Timing market entries and exits more effectively
- Detecting seasonal or cyclical behavior in ETF returns
- Adjusting portfolio rebalancing strategies
- Enhancing historical awareness of volatility and market anomalies

This dual-chart setup enables both high-level pattern recognition and granular month-by-month analysis—supporting more informed, data-driven decisions.
    """)


# choose ETF & year range
etf_sel = st.selectbox("ETF:", data.ETF.unique())
yr_min, yr_max = int(data.Date.dt.year.min()), int(data.Date.dt.year.max())
year_range = st.slider("Year range:", yr_min, yr_max, (yr_min, yr_max), step=1)

# prepare monthly avg returns
df_month = (
    data[(data.ETF==etf_sel)]
      .assign(Year=lambda d: d.Date.dt.year,
              Month=lambda d: d.Date.dt.month)
      .query("Year >= @year_range[0] and Year <= @year_range[1]")
      .groupby(["Year","Month"])["Daily_Return"]
      .mean()
      .reset_index()
)

# pivot into heatmap-friendly
heat_df = df_month.pivot(index="Year", columns="Month", values="Daily_Return").fillna(0)

# altair heatmap
heatmap = (
    alt.Chart(df_month)
       .mark_rect()
       .encode(
           x=alt.X("Month:O", title="Month",
                   axis=alt.Axis(labelFlush=True, labelAngle=0, tickCount=12,
                                 labelExpr="datum.value>0 ? datum.value : ''",
                                 labelFontSize=10)),
           y=alt.Y("Year:O", title="Year",
                   axis=alt.Axis(labelFontSize=10)),
           color=alt.Color("Daily_Return:Q",
                           title="Avg Daily Return",
                           scale=alt.Scale(scheme="redblue", domainMid=0)),
           tooltip=[
             alt.Tooltip("Year:O"), 
             alt.Tooltip("Month:O", title="Mon"),
             alt.Tooltip("Daily_Return:Q", format=".2%")
           ]
       )
       .properties(width=700, height=300)
)
st.altair_chart(heatmap, use_container_width=True)


# ── Drill-down: Monthly Bar Chart ─────────────────────────────────────────
st.subheader(f"{etf_sel} Monthly Returns Breakdown")

# pick a single year for bar chart
bar_year = st.selectbox("Select year:", list(range(year_range[0], year_range[1]+1)))
df_bar = (
    df_month[df_month.Year==bar_year]
      .sort_values("Month")
)

bar = (
    alt.Chart(df_bar)
       .mark_bar()
       .encode(
           x=alt.X("Month:O", title="Month",
                   axis=alt.Axis(labelAngle=0, tickCount=12)),
           y=alt.Y("Daily_Return:Q", title="Avg Daily Return", axis=alt.Axis(format=".2%")),
           tooltip=[alt.Tooltip("Daily_Return:Q", format=".2%"),
                    "Month:O"]
       )
       .properties(width=700, height=200)
)
st.altair_chart(bar, use_container_width=True)

# =============================================================================
# NEW SECTION ▸ ETF Radar Comparison
# =============================================================================
import plotly.express as px

st.header("ETF Radar Comparison")
with st.expander("About ETF Radar Comparison"):
    st.markdown("""
The **ETF Radar Comparison** module enables side-by-side analysis of two selected ETFs using both a normalized radar chart and actual metric values.

---

### Radar Chart (Top)

- **Purpose**:  
  Provides a visual snapshot of how two ETFs stack up across key performance and risk metrics for a selected year.

- **Metrics Compared**:
  - **Annual Return**: Total percentage gain over the selected year.
  - **Annual Volatility**: Standard deviation of daily returns annualized—higher values imply greater risk.
  - **Max Drawdown**: Worst peak-to-trough decline in adjusted closing price—used to assess downside risk.
  - **Sharpe Ratio**: Risk-adjusted return; calculated as \((\mu - r_f) / \sigma\), where:
    - \( \mu \) = average return
    - \( r_f \) = risk-free rate (e.g., treasury yield)
    - \( \sigma \) = volatility
  - **CAGR (Compound Annual Growth Rate)**: Measures the geometric growth rate assuming reinvestment and smooth compounding.

- **Normalization**:
  Each metric is scaled from 0 to 1 across the two ETFs to fit the radar format, allowing quick visual comparison even if the raw values differ in magnitude.

- **Visual Interpretation**:
  - A larger, more filled-in polygon suggests stronger performance across the board.
  - For instance, if one ETF has higher annual return, better Sharpe, and lower drawdown, it will dominate the radar plot.

---

### Metric Table (Bottom)

- **Purpose**:  
  Complements the radar chart by showing actual, unscaled values for precision.

- **Columns**:
  - Each row corresponds to an ETF
  - Columns include Annual Return, Volatility, Max Drawdown, Sharpe Ratio, and CAGR
  - Values are formatted clearly with percent signs and decimals for easy reference

---

### Interactivity

- **ETF Selector**: Choose any two ETFs from the dropdown
- **Year Slider**: Adjust the year to compare ETF performance in different market regimes (e.g., crisis years, bull runs)

---

### Use Cases

- Compare growth-oriented vs. defensive ETFs in volatile years
- Understand which ETF offers better risk-adjusted returns
- Make informed allocation decisions based on historical behavior
- Spot years when one ETF clearly outperformed across most dimensions

This comparison tool allows both quick pattern recognition (radar chart) and precision evaluation (data table), helping users make nuanced judgments about ETF performance.
    """)


# ── Controls ────────────────────────────────────────────────────────────────
etfs_radar = st.multiselect(
    "Select up to 2 ETFs to compare:",
    options=data.ETF.unique(),
    default=["EEM", "IVV"],
    max_selections=2
)
radar_year = st.slider(
    "Choose year for snapshot:",
    int(data.Date.dt.year.min()),
    int(data.Date.dt.year.max()),
    int(data.Date.dt.year.max())
)

# ── Compute snapshot stats at annual frequency ───────────────────────────────
df_year = (
    data.assign(Year=data.Date.dt.year)
        .query("Year == @radar_year")
        .groupby("ETF")
        .agg(
            Annual_Return=("Daily_Return", lambda x: (1 + x).prod() - 1),
            Annual_Volatility=("Daily_Return", "std"),
            Max_Drawdown=("Drawdown", "min"),
            Sharpe=("Daily_Return", lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252))),
            CAGR=("Daily_Return", lambda x: (1 + x).prod() ** (1 / 1) - 1)
        )
        .reset_index()
        .query("ETF in @etfs_radar")
)

# ── Melt for polar plot ───────────────────────────────────────────────────────
radar_df = df_year.melt(
    id_vars="ETF",
    var_name="Metric",
    value_name="Value"
)

# ── Normalize metrics to [0,1] for display ───────────────────────────────────
# (so that different scales can be seen relative to each other)
norms = {}
for m in radar_df.Metric.unique():
    mn, mx = radar_df.query("Metric == @m").Value.min(), radar_df.query("Metric == @m").Value.max()
    norms[m] = (mn, mx)
radar_df["NormValue"] = radar_df.apply(
    lambda row: (row.Value - norms[row.Metric][0]) / (norms[row.Metric][1] - norms[row.Metric][0] + 1e-9),
    axis=1
)

# ── Build radar chart ─────────────────────────────────────────────────────────
fig_radar = px.line_polar(
    radar_df,
    r="NormValue",
    theta="Metric",
    color="ETF",
    line_close=True,
    template="plotly_dark",
    title=f"ETF Metrics Radar — {radar_year}"
)
fig_radar.update_traces(fill="toself", opacity=0.6)
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(range=[0,1], visible=True, tickvals=[0,0.5,1], ticktext=["Low","Med","High"])
    ),
    legend=dict(title=None),
    margin=dict(l=20,r=20,t=40,b=20),
    height=500
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── Show actual numbers in table ─────────────────────────────────────────────
st.subheader("Actual Metric Values")
st.dataframe(
    df_year.set_index("ETF").style.format({
        "Annual_Return": "{:.1%}",
        "Annual_Volatility": "{:.2%}",
        "Max_Drawdown": "{:.1%}",
        "Sharpe": "{:.2f}",
        "CAGR": "{:.1%}",
    }),
    use_container_width=True
)
