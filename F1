import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------
# Driver colors (edit here)
# -----------------------
DRIVER_COLORS = {
    "NOR": "#FF8700",  # McLaren orange
    "VER": "#1E41FF",  # Red Bull-ish blue
    "LEC": "#DC0000",  # Ferrari red
    "HAM": "#00D2BE",  # Mercedes teal
    "ALO": "#006F62",  # Aston-ish green
}

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="F1 Pit Wall (MVP)", layout="wide")
st.title("F1 Pit Wall Strategy Dashboard — MVP")

# -----------------------
# 0. Data (synthetic so it runs from zero)
# -----------------------
np.random.seed(7)
drivers = ["NOR", "VER", "LEC", "HAM", "ALO"]
laps_total = 58

rows = []
for d in drivers:
    base = 93.0 + np.random.uniform(-0.6, 0.6)
    pit_laps = sorted(np.random.choice(range(10, 45), size=2, replace=False))
    stint = 1
    compounds = ["SOFT", "MEDIUM", "HARD"]
    comp = compounds[np.random.randint(0, 3)]

    for lap in range(1, laps_total + 1):
        # simple degradation + noise
        deg = 0.02 * lap
        noise = np.random.normal(0, 0.25)
        lap_time = base + deg + noise

        pit = lap in pit_laps
        if pit:
            lap_time += 20.5  # pit loss
            stint += 1
            comp = compounds[(compounds.index(comp) + 1) % 3]

        rows.append(
            {
                "race": "SAMPLE_RACE",
                "driver": d,
                "lap": lap,
                "lap_time_s": round(float(lap_time), 3),
                "stint": stint,
                "compound": comp,
                "pit_stop": pit,
            }
        )

df = pd.DataFrame(rows)

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Controls")
selected_drivers = st.sidebar.multiselect(
    "Drivers", sorted(df["driver"].unique()), default=["NOR", "VER", "LEC"]
)
show_pit_markers = st.sidebar.checkbox("Show pit markers", value=True)

if not selected_drivers:
    st.warning("Please select at least one driver.")
    st.stop()

baseline_driver = st.sidebar.selectbox(
    "Delta baseline driver",
    options=selected_drivers,
    index=0,
    help="Delta plot shows each driver's clean-lap time minus baseline driver's clean-lap time (same lap).",
)

# Delta display controls (1)
st.sidebar.subheader("Delta View")
delta_last_n = st.sidebar.slider("Show last N laps", 5, int(df["lap"].max()), 15)
delta_roll = st.sidebar.slider("Rolling average (laps)", 1, 10, 3)
show_raw_delta = st.sidebar.checkbox("Show raw delta (no smoothing)", value=False)

# Ensure plot includes baseline
plot_drivers = sorted(set(selected_drivers + [baseline_driver]))
plot_df = df[df["driver"].isin(plot_drivers)].copy()

# color map for plotly
color_map = {d: DRIVER_COLORS.get(d, "#999999") for d in plot_drivers}

# -----------------------
# Layout
# -----------------------
left, right = st.columns([2, 1], gap="large")

# -----------------------
# Right column (what-if) first so left can shade by its parameters
# -----------------------
with right:
    st.subheader("What-if: Undercut (Toy Model)")

    driver_focus = st.selectbox("Focus driver", sorted(df["driver"].unique()), index=0)
    ddf = df[df["driver"] == driver_focus].sort_values("lap")

    target_lap = st.slider("Hypothetical pit lap", 1, int(ddf["lap"].max()), 20)
    last_n = st.slider("Use last N clean laps", 3, 15, 8)
    pit_loss = st.number_input("Assumed pit loss (s)", 10.0, 35.0, 20.5, 0.5)
    tire_gain = st.number_input("Assumed tire gain (s/lap)", 0.0, 3.0, 0.8, 0.1)
    window_laps = st.slider("Evaluation window (laps)", 1, 15, 5)

    hist = ddf[(ddf["lap"] < target_lap) & (~ddf["pit_stop"])].tail(last_n)

    if len(hist) < 3:
        st.warning("Not enough clean laps before the selected pit lap.")
    else:
        baseline = hist["lap_time_s"].mean()
        net = pit_loss - tire_gain * window_laps

        st.metric("Estimated net delta after window (s)", f"{net:.2f}")
        st.caption(
            f"Baseline pace (avg last {len(hist)} clean laps): {baseline:.3f}s. "
            f"Net = pit_loss ({pit_loss:.1f}) - tire_gain ({tire_gain:.1f}) × window ({window_laps})."
        )
        if net <= 0:
            st.success("Toy model suggests the undercut window is favorable (net gain).")
        else:
            st.info("Toy model suggests the undercut may not be worth it (net loss).")

# -----------------------
# Helper: clean laps
# -----------------------
def clean_laps(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["lap_time_clean"] = np.where(out["pit_stop"], np.nan, out["lap_time_s"])
    return out

clean_df = clean_laps(plot_df)

# -----------------------
# (2) KPI Header row (pit wall feel)
# -----------------------
# fastest clean lap among currently selected drivers
kpi_pool = clean_df[clean_df["driver"].isin(selected_drivers)].dropna(subset=["lap_time_clean"])
if kpi_pool.empty:
    fastest_driver = "-"
    fastest_time = np.nan
else:
    idx = kpi_pool["lap_time_clean"].idxmin()
    fastest_driver = str(kpi_pool.loc[idx, "driver"])
    fastest_time = float(kpi_pool.loc[idx, "lap_time_clean"])

# focus driver KPIs
focus_clean = clean_df[clean_df["driver"] == driver_focus].dropna(subset=["lap_time_clean"]).sort_values("lap")
if focus_clean.empty:
    focus_last = np.nan
    focus_best = np.nan
    focus_last_lap = None
else:
    focus_last = float(focus_clean.iloc[-1]["lap_time_clean"])
    focus_last_lap = int(focus_clean.iloc[-1]["lap"])
    focus_best = float(focus_clean["lap_time_clean"].min())

focus_pit_count = int(df[df["driver"] == driver_focus]["pit_stop"].sum())

# baseline recent pace (last delta_last_n clean laps)
base_clean = clean_df[clean_df["driver"] == baseline_driver].dropna(subset=["lap_time_clean"]).sort_values("lap")
base_recent = base_clean.tail(delta_last_n)["lap_time_clean"].mean() if not base_clean.empty else np.nan

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Race", "SAMPLE_RACE")
m2.metric("Fastest clean lap", f"{fastest_time:.3f}s" if np.isfinite(fastest_time) else "-", fastest_driver)
m3.metric(
    f"{driver_focus} last clean",
    f"{focus_last:.3f}s" if np.isfinite(focus_last) else "-",
    f"Lap {focus_last_lap}" if focus_last_lap is not None else "-",
)
m4.metric(f"{driver_focus} pit stops", f"{focus_pit_count:d}")
m5.metric(
    f"{baseline_driver} avg (last {delta_last_n})",
    f"{base_recent:.3f}s" if np.isfinite(base_recent) else "-",
)

# -----------------------
# Left column (plots + tables)
# -----------------------
with left:
    # ---------
    # Lap Time Trace + Pit window shading
    # ---------
    st.subheader("Lap Time Trace")
    fig = px.line(
        plot_df,
        x="lap",
        y="lap_time_s",
        color="driver",
        hover_data=["stint", "compound", "pit_stop"],
        color_discrete_map=color_map,
    )

    # Pit window shading (use focus driver what-if controls)
    x0 = target_lap
    x1 = min(target_lap + window_laps, int(df["lap"].max()))
    fig.add_vline(
        x=x0,
        line_width=2,
        line_dash="dash",
        annotation_text=f"Pit lap (focus): {driver_focus} L{x0}",
        annotation_position="top left",
    )
    fig.add_vrect(
        x0=x0,
        x1=x1,
        opacity=0.12,
        line_width=0,
        annotation_text=f"Eval window: {window_laps} laps",
        annotation_position="top right",
    )

    if show_pit_markers:
        pit_df = plot_df[plot_df["pit_stop"]]
        if not pit_df.empty:
            pit_scatter = px.scatter(
                pit_df,
                x="lap",
                y="lap_time_s",
                color="driver",
                hover_data=["stint", "compound"],
                color_discrete_map=color_map,
            )
            for tr in pit_scatter.data:
                tr.update(marker=dict(size=9, symbol="x"))
                fig.add_trace(tr)

    fig.update_layout(xaxis_title="Lap", yaxis_title="Lap time (s)", height=520)
    st.plotly_chart(fig, use_container_width=True)

    # ---------
    # (1) Delta plot: last N + rolling
    # ---------
    st.subheader(f"Delta to {baseline_driver} (clean laps)")

    pivot = clean_df.pivot_table(
        index="lap",
        columns="driver",
        values="lap_time_clean",
        aggfunc="mean",
    ).sort_index()

    if baseline_driver not in pivot.columns:
        st.warning("Baseline driver not available in the current selection.")
    else:
        # restrict to last N laps in the session (engineering-style view)
        max_lap = int(pivot.index.max())
        min_lap = max(1, max_lap - delta_last_n + 1)
        pivot_last = pivot.loc[(pivot.index >= min_lap) & (pivot.index <= max_lap)].copy()

        delta_wide = pivot_last.sub(pivot_last[baseline_driver], axis=0)

        # drop baseline series for display (optional)
        if baseline_driver in delta_wide.columns:
            delta_wide_disp = delta_wide.drop(columns=[baseline_driver])
        else:
            delta_wide_disp = delta_wide

        # rolling average
        rolled = delta_wide_disp.rolling(window=delta_roll, min_periods=1).mean()

        def wide_to_long(w: pd.DataFrame, value_name: str) -> pd.DataFrame:
            return (
                w.reset_index()
                .melt(id_vars="lap", var_name="driver", value_name=value_name)
                .dropna()
            )

        # smoothed line
        rolled_long = wide_to_long(rolled, "delta_s")

        if rolled_long.empty:
            st.info("No clean-lap overlap to compute delta (try changing drivers).")
        else:
            fig2 = px.line(
                rolled_long,
                x="lap",
                y="delta_s",
                color="driver",
                color_discrete_map=color_map,
            )
            fig2.add_hline(y=0, line_width=1, line_dash="dot")
            fig2.update_layout(
                xaxis_title=f"Lap (last {delta_last_n})",
                yaxis_title=f"Delta vs {baseline_driver} (s) — rolling {delta_roll}",
                height=360,
            )
            st.plotly_chart(fig2, use_container_width=True)

            # optional raw overlay
            if show_raw_delta:
                raw_long = wide_to_long(delta_wide_disp, "delta_s_raw")
                if not raw_long.empty:
                    fig3 = px.line(
                        raw_long,
                        x="lap",
                        y="delta_s_raw",
                        color="driver",
                        color_discrete_map=color_map,
                    )
                    fig3.add_hline(y=0, line_width=1, line_dash="dot")
                    fig3.update_layout(
                        xaxis_title=f"Lap (last {delta_last_n})",
                        yaxis_title=f"Delta vs {baseline_driver} (s) — raw",
                        height=300,
                    )
                    st.plotly_chart(fig3, use_container_width=True)

    # ---------
    # (4) Stints table upgrade: mean/best clean pace + degradation slope
    # ---------
    st.subheader("Stints (Summary Table — Pace & Degradation)")

    def slope_per_lap(group: pd.DataFrame) -> float:
        g = group.dropna(subset=["lap_time_clean"])
        if len(g) < 3:
            return np.nan
        # linear fit: lap_time_clean = a*lap + b
        a = np.polyfit(g["lap"].values.astype(float), g["lap_time_clean"].values.astype(float), 1)[0]
        return float(a)

    stint_table = (
        clean_df.groupby(["driver", "stint", "compound"], as_index=False)
        .agg(
            lap_start=("lap", "min"),
            lap_end=("lap", "max"),
            laps=("lap", "count"),
            mean_clean=("lap_time_clean", "mean"),
            best_clean=("lap_time_clean", "min"),
        )
        .sort_values(["driver", "stint"])
    )

    # add slope by group (degradation proxy)
    slopes = (
        clean_df.groupby(["driver", "stint", "compound"], as_index=False)
        .apply(lambda g: pd.Series({"deg_s_per_lap": slope_per_lap(g)}))
        .reset_index(drop=True)
    )

    stint_table = stint_table.merge(slopes, on=["driver", "stint", "compound"], how="left")

    # nicer display
    stint_table["mean_clean"] = stint_table["mean_clean"].round(3)
    stint_table["best_clean"] = stint_table["best_clean"].round(3)
    stint_table["deg_s_per_lap"] = stint_table["deg_s_per_lap"].round(4)

    st.dataframe(stint_table, use_container_width=True, hide_index=True)

st.caption("MVP uses synthetic data. Next step: replace with real race lap data (CSV).")
