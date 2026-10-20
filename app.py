import random
from datetime import date, timedelta
from math import ceil, log2
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from benchmark import run_benchmark
from bplus_index import SEASONS, BPlusTree, build_index, city_labels, date_ints, fmt_key
from heatwave_analysis import (
    ADVISORIES, THRESHOLDS, add_normals, assess, backtest_forecast, build_events, classify,
    criteria_text, data_quality, outlook, region_summary, regional_daily, trend_per_decade,
    yearly_stats,
)

st.set_page_config(page_title="Heatwave Watch", page_icon="🌡️", layout="wide")
DATA_FILE = Path(__file__).parent / "heatwave_data.csv"

st.title("🌡️ Heatwave Watch")
st.caption("Climate Intelligence for Heatwave Monitoring and Early Warning")

@st.cache_data
def load_data(file_bytes=None):
    if file_bytes is None:
        data = pd.read_csv(DATA_FILE)
    else:
        data = pd.read_csv(file_bytes)

    # Match the uploaded dataset's original column names to the app's standard names.
    data = data.rename(columns={
        "Date": "date",
        "Temp Max": "tmax",
        "Temp Min": "tmin",
    })

    required = {"region", "date", "lat", "lon", "tmax"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
            + ". Required columns are region, Date/date, lat, lon, and Temp Max/tmax."
        )

    # Dates may be ISO (2024-05-21) or day-month-year (21-05-2024). Decide by the text itself:
    # dayfirst=True on ISO text swaps day and month or drops rows, so ISO is parsed strictly.
    text = data["date"].astype("string").str.strip()
    iso = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False)
    parsed = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    parsed[iso] = pd.to_datetime(text[iso], format="%Y-%m-%d", errors="coerce")
    parsed[~iso] = pd.to_datetime(text[~iso], dayfirst=True, errors="coerce")
    data["date"] = parsed
    for col in ("tmax", "lat", "lon"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if "tmin" in data.columns:
        data["tmin"] = pd.to_numeric(data["tmin"], errors="coerce")
    if "Rain" in data.columns:
        data["Rain"] = pd.to_numeric(data["Rain"], errors="coerce")

    data["region"] = data["region"].astype("string").str.strip()
    data = data.dropna(subset=["region", "date", "lat", "lon", "tmax"])
    data = data[data["region"] != ""]
    return data.sort_values("date")

try:
    df = load_data()
except FileNotFoundError:
    st.error("heatwave_data.csv was not found. Keep it in the same folder as app.py.")
    st.stop()
except (ValueError, pd.errors.ParserError) as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if df.empty:
    st.error("No valid observations were found in the dataset.")
    st.stop()

with st.sidebar:
    st.header("Search observations")
    uploaded_file = st.file_uploader(
        "Upload another CSV (optional)",
        type=["csv"],
        help="Use columns region, Date, lat, lon, and Temp Max, or their lowercase equivalents.",
    )

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except (ValueError, pd.errors.ParserError) as exc:
        st.sidebar.error(f"Could not load uploaded CSV: {exc}")
        st.stop()
    if df.empty:
        st.warning("The uploaded CSV contains no valid observations.")
        st.stop()

# Identifies the current dataset for the cached index and analysis below.
data_key = (len(df), str(df["date"].min()), str(df["date"].max()), round(float(df["tmax"].sum()), 2))

regions = sorted(df["region"].dropna().unique().tolist())
with st.sidebar:
    region = st.selectbox("Select region", regions)
    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    start = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
    end = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)
    threshold = st.slider("Hot-day threshold (°C)", 30.0, 50.0, 40.0, 0.5)
    st.divider()
    terrain_choice = st.selectbox(
        "Heat-wave criteria (terrain)",
        ["Auto", "Plains", "Coastal", "Hilly"],
        help="Auto picks coastal, hilly or plains from the region name. "
             "IMD thresholds: " + ", ".join(f"{k} {v:.0f} °C" for k, v in THRESHOLDS.items()) + ".",
    )
    st.caption("The warning status is assessed as of the End date, so you can replay any past heat wave.")

if start > end:
    st.error("Start date must be on or before the end date.")
    st.stop()

result = df[
    (df["region"] == region)
    & (df["date"].dt.date >= start)
    & (df["date"].dt.date <= end)
].copy()

if result.empty:
    st.warning("No observations found for this region and date range. Try another selection.")
    st.stop()


# ----------------------------------------------------------------------
# Heat-wave analysis (IMD-style criteria), cached per dataset
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Analysing heat-wave conditions...")
def get_analysis(_data, key, terrain_override):
    # `_data` is not hashed by Streamlit; `key` decides when to recompute.
    daily = classify(add_normals(regional_daily(_data)), terrain_override)
    return daily, build_events(daily)


daily_all, events_all = get_analysis(df, data_key, terrain_choice)
region_days = daily_all[daily_all["region"] == region]
region_events = events_all[events_all["region"] == region]
status = assess(region_days, pd.Timestamp(end))

# ----------------------------------------------------------------------
# 1. Early-warning status
# ----------------------------------------------------------------------
if status is not None:
    colour = status["hex"]
    st.markdown(
        f"""<div style="border-left:10px solid {colour};background:{colour}22;
padding:14px 18px;border-radius:8px;margin-bottom:10px;">
<div style="font-size:1.4rem;font-weight:700;">{status['emoji']} {status['colour']} · {status['label']}</div>
<div style="opacity:.85;">{region} · as of {status['as_of']:%d %b %Y} · {status['reason']}</div>
</div>""",
        unsafe_allow_html=True,
    )
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Tmax on that day", f"{status['tmax']:.1f} °C")
    w2.metric("Normal for the date", f"{status['normal']:.1f} °C")
    w3.metric("Departure from normal", f"{status['dep']:+.1f} °C")
    w4.metric("Heat-wave days in a row", status["run_pos"])
    st.caption(
        f"Criteria used ({status['terrain']}): {criteria_text(status['terrain'])} "
        "The region value is the average of its stations and the normal is the dataset's own long-term "
        "average, so this is an indicative monitoring signal, not an official IMD warning."
    )
    with st.expander("Advisories for this alert level"):
        tabs = st.tabs(list(ADVISORIES[status["level"]].keys()))
        for tab, (audience, tips) in zip(tabs, ADVISORIES[status["level"]].items()):
            with tab:
                for tip in tips:
                    st.markdown(f"- {tip}")
        st.caption("These advisories are written from fixed templates for each alert level. "
                   "For official warnings, follow IMD and your state disaster-management authority.")

# ----------------------------------------------------------------------
# 2. Observations
# ----------------------------------------------------------------------
a, b, c, d = st.columns(4)
a.metric("Records", f"{len(result):,}")
b.metric("Average Tmax", f"{result['tmax'].mean():.2f} °C")
c.metric("Highest Tmax", f"{result['tmax'].max():.2f} °C")
d.metric("Records above threshold", f"{int((result['tmax'] >= threshold).sum()):,}")

daily = result.groupby("date", as_index=False)["tmax"].mean()
fig = px.line(
    daily, x="date", y="tmax", markers=True,
    title=f"Temperature Trend: {region}",
    labels={"date": "Date", "tmax": "Average Tmax (°C)"},
)
fig.data[0].name = "Average Tmax"
fig.data[0].showlegend = True
window_days = region_days[(region_days["date"].dt.date >= start) & (region_days["date"].dt.date <= end)]
fig.add_scatter(x=window_days["date"], y=window_days["normal"], mode="lines",
                name="Normal for the date", line=dict(dash="dot"))
hw_days = window_days[window_days["event_day"]]
if not hw_days.empty:
    fig.add_scatter(x=hw_days["date"], y=hw_days["tmax"], mode="markers", name="Heat-wave day",
                    marker=dict(color="red", size=9, symbol="diamond"))
fig.add_hline(y=threshold, line_dash="dash", annotation_text="Selected threshold")
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    fig2 = px.histogram(
        result, x="tmax", nbins=25, title="Temperature Distribution",
        labels={"tmax": "Maximum temperature (°C)"},
    )
    st.plotly_chart(fig2, use_container_width=True)

with right:
    period = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]
    summary = period.groupby("region", as_index=False)["tmax"].mean().sort_values("tmax", ascending=False)
    fig3 = px.bar(
        summary, x="region", y="tmax", title="Average Maximum Temperature by Region",
        labels={"region": "Region", "tmax": "Average Tmax (°C)"},
    )
    st.plotly_chart(fig3, use_container_width=True)

if "tmin" in result.columns:
    st.subheader("Minimum Temperature Trend")
    daily_min = result.groupby("date", as_index=False)["tmin"].mean()
    fig4 = px.line(
        daily_min, x="date", y="tmin", markers=True,
        title=f"Daily Average Minimum Temperature: {region}",
        labels={"date": "Date", "tmin": "Average Tmin (°C)"},
    )
    st.plotly_chart(fig4, use_container_width=True)

# ----------------------------------------------------------------------
# 3. Heat-wave intelligence
# ----------------------------------------------------------------------
st.divider()
st.header("🔥 Heat-wave Intelligence")
tab_events, tab_trends, tab_hot, tab_outlook, tab_quality = st.tabs(
    ["Heat-wave events", "Frequency, intensity and duration", "Regional hotspots",
     "Outlook (next 5 days)", "Data reliability"]
)

with tab_events:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    win_events = region_events[(region_events["end"] >= start_ts) & (region_events["start"] <= end_ts)]
    e1, e2, e3 = st.columns(3)
    e1.metric("Heat-wave events", len(win_events))
    e2.metric("Heat-wave days", int(window_days["event_day"].sum()))
    e3.metric("Longest event", f"{int(win_events['days'].max()) if len(win_events) else 0} days")
    if win_events.empty:
        st.info("No heat-wave event (criteria met for 2 or more consecutive days) in this region and period.")
    else:
        table = pd.DataFrame({
            "Start": win_events["start"].dt.date,
            "End": win_events["end"].dt.date,
            "Days": win_events["days"],
            "Peak Tmax (°C)": win_events["peak_tmax"].round(1),
            "Max departure (°C)": win_events["max_dep"].round(1),
            "Severity": win_events["severity"],
        })
        st.dataframe(table.sort_values("Start", ascending=False), hide_index=True, use_container_width=True)
    st.caption("An event is a spell of at least 2 consecutive days that meets the heat-wave criteria. "
               "Heat-wave days are also marked with red diamonds on the temperature trend above.")

with tab_trends:
    ys = yearly_stats(region_days, region_events)
    complete = ys[ys["complete"]]
    if len(complete) < 5:
        st.info("At least 5 complete years of data are needed to show a trend.")
    else:
        ys["Data"] = ys["complete"].map({True: "Complete year", False: "Incomplete year"})
        charts = [
            ("events", "Frequency: events per year", "Events"),
            ("mean_departure", "Intensity: average departure on heat-wave days (°C)", "°C above normal"),
            ("longest_event", "Duration: longest event per year (days)", "Days"),
        ]
        cols = st.columns(3)
        for col, (field, title, ylabel) in zip(cols, charts):
            fit = trend_per_decade(complete["year"], complete[field])
            tfig = px.bar(ys, x="year", y=field, color="Data", title=title,
                          labels={"year": "Year", field: ylabel},
                          color_discrete_map={"Complete year": "#e4572e", "Incomplete year": "#9aa0a6"})
            if fit:
                tfig.add_scatter(x=complete["year"], y=fit["slope"] * complete["year"] + fit["intercept"],
                                 mode="lines", name="Trend", line=dict(color="black", width=2))
            tfig.update_layout(showlegend=False, title_font_size=14)
            col.plotly_chart(tfig, use_container_width=True)
            if fit:
                col.metric("Change per decade", f"{fit['per_decade']:+.2f}")
        st.caption("The black line is a straight-line fit over complete years, shown as the change per decade. "
                   "A positive value means more, stronger or longer heat waves over time in this dataset. "
                   "With few events per year the trend can be noisy, so read it as a guide, not proof.")

with tab_hot:
    summ = region_summary(daily_all, events_all, start, end)
    centres = df.groupby("region")[["lat", "lon"]].mean().reset_index()
    centres["region"] = centres["region"].astype(str)
    summ = summ.merge(centres, left_on="Region", right_on="region", how="left").drop(columns="region")
    ranked = summ.sort_values(["Heat-wave days", "Peak Tmax (°C)"], ascending=False)
    top = ranked.iloc[0]
    if top["Heat-wave days"] > 0:
        n_ev = int(top["Events"])
        st.info(f"Hotspot for {start} to {end}: **{top['Region']}** with {int(top['Heat-wave days'])} heat-wave days "
                f"in {n_ev} event{'s' if n_ev != 1 else ''} (peak Tmax {top['Peak Tmax (°C)']} °C).")
    else:
        st.info("No region had a heat-wave event in this period.")
    h1, h2 = st.columns(2)
    hfig = px.bar(ranked, x="Region", y="Heat-wave days", color="Peak Tmax (°C)",
                  color_continuous_scale="YlOrRd", title="Heat-wave days by region")
    h1.plotly_chart(hfig, use_container_width=True)
    ranked = ranked.assign(Size=ranked["Heat-wave days"] + 2)
    mfig = px.scatter_geo(
        ranked, lat="lat", lon="lon", size="Size", color="Max departure (°C)", hover_name="Region",
        hover_data={"Heat-wave days": True, "Events": True, "Peak Tmax (°C)": True,
                    "Size": False, "lat": False, "lon": False},
        color_continuous_scale="YlOrRd", size_max=40, title="Hotspot map (bubble size = heat-wave days)",
    )
    mfig.update_geos(fitbounds="locations", showcountries=True, showcoastlines=True)
    h2.plotly_chart(mfig, use_container_width=True)
    st.dataframe(ranked.drop(columns=["lat", "lon", "Size"]), hide_index=True, use_container_width=True)
    st.caption("Each region uses the criteria for its terrain (or the terrain chosen in the sidebar).")

with tab_outlook:
    known = region_days[region_days["date"] <= pd.Timestamp(end)]
    out = outlook(known) if len(known) >= 60 else None
    if out is None:
        st.info("At least 60 days of data before the End date are needed for the outlook.")
    else:
        oc = out["hex"]
        st.markdown(
            f"""<div style="border-left:10px solid {oc};background:{oc}22;padding:12px 16px;border-radius:8px;margin-bottom:10px;">
<div style="font-size:1.2rem;font-weight:700;">{out['emoji']} Expected: {out['colour']} · {out['label']}</div>
<div style="opacity:.85;">{region} · next 5 days after {known['date'].max():%d %b %Y} · {out['reason']}</div>
</div>""",
            unsafe_allow_html=True,
        )
        fc = out["table"]
        st.dataframe(pd.DataFrame({
            "Date": fc["date"].dt.date,
            "Forecast Tmax (°C)": fc["forecast"].round(1),
            "Normal (°C)": fc["normal"].round(1),
            "Departure (°C)": fc["dep"].round(1),
            "Heat-wave criteria met": fc["flag"].map({True: "Yes", False: "No"}),
        }), hide_index=True, use_container_width=True)
        ofig = px.line(known.tail(30), x="date", y="tmax", markers=True,
                       title=f"Last 30 days and 5-day outlook: {region}",
                       labels={"date": "Date", "tmax": "Regional Tmax (°C)"})
        ofig.data[0].name = "Observed Tmax"
        ofig.data[0].showlegend = True
        ofig.add_scatter(x=fc["date"], y=fc["forecast"], mode="lines+markers", name="Forecast",
                         line=dict(dash="dash"))
        ofig.add_hline(y=out["threshold"], line_dash="dot",
                       annotation_text=f"{out['terrain']} threshold")
        st.plotly_chart(ofig, use_container_width=True)
        st.caption(
            f"Method: forecast = normal for that date + today's departure from normal x {out['phi']:.2f} per day "
            "(the departure fades back to normal). This is a simple statistical outlook built on the same indexed "
            "data, not the AI/LSTM model of the full project and not an official IMD forecast."
        )
        bt = backtest_forecast(known)
        if bt is None:
            st.info("The back-test needs at least 6 years of data.")
        else:
            st.markdown("**Back-test on the last 3 years** (average error in °C, lower is better)")
            st.dataframe(bt.round(2), hide_index=True, use_container_width=True)
            st.caption("The persistence factor was fitted on earlier years only. The outlook is compared with "
                       "'tomorrow equals today' (persistence) and 'always the normal' (climatology). "
                       "Use the 'Days tested' column to judge how much data each number rests on.")

with tab_quality:
    q = data_quality(df, region)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows for this region", f"{q['rows']:,}")
    q2.metric("Missing days", f"{q['missing_days']:,}")
    q3.metric("Repeated rows", f"{q['duplicates']:,}")
    q4.metric("Implausible Tmax", f"{q['implausible_tmax']:,}")
    st.caption(f"Data covers {q['first']:%d %b %Y} to {q['last']:%d %b %Y}.")
    checks = [
        (q["missing_days"] == 0, "No missing days between the first and last date.",
         f"{q['missing_days']:,} days have no observation, so a heat wave there could be missed or cut short."),
        (q["duplicates"] == 0, "No repeated (date, location) rows.",
         f"{q['duplicates']:,} repeated rows. They are counted twice in the averages."),
        (q["implausible_tmax"] == 0, "All Tmax values lie between 0 and 55 °C.",
         f"{q['implausible_tmax']:,} Tmax values are outside 0 to 55 °C and may be sensor or entry errors."),
    ]
    if q["tmax_below_tmin"] is not None:
        checks.append((q["tmax_below_tmin"] == 0, "Tmax is never below Tmin.",
                       f"{q['tmax_below_tmin']:,} rows have Tmax below Tmin."))
    if q["missing_tmin"] is not None:
        checks.append((q["missing_tmin"] == 0, "No missing Tmin values.", f"{q['missing_tmin']:,} Tmin values are missing."))
    for ok, good, bad in checks:
        (st.success if ok else st.warning)(good if ok else bad)
    st.caption("Reliable warnings need complete, clean data. These checks run on the selected region's raw rows.")

st.subheader("Temperature Records")
st.dataframe(result, use_container_width=True, hide_index=True)
st.download_button(
    "Download filtered results as CSV",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="heatwave_results.csv",
    mime="text/csv",
)

# ----------------------------------------------------------------------
# 4. Data structure view: Hash Table + B+ Tree (the design from the IA-2 report)
# ----------------------------------------------------------------------
def get_index(data, key):
    """One index per browser session, rebuilt only when the dataset changes. It lives in session_state
    (not in a shared cache) because section 5 inserts and deletes readings, and those edits must stay
    private to this session."""
    if st.session_state.get("index_key") != key:
        with st.spinner("Building the hash table + B+ Tree index (one-time, please wait)..."):
            st.session_state["index_build"] = build_index(data)
        st.session_state.update(index_key=key, live_log=[], live_added={}, live_deleted=set(), live_msg=None)
    return st.session_state["index_build"]


def to_int(day):
    return day.year * 10000 + day.month * 100 + day.day


st.divider()
st.header("🌳 Data Structure View: Hash Table + B+ Tree")
st.caption(
    "The dashboard above filters with pandas, which scans every row. This section answers the "
    "same query with the index from our IA-2 report: a hash table (region → tree) and one "
    "B+ Tree per region, keyed by (date, city), with linked leaves."
)

if st.toggle("Build and show the B+ Tree index", value=True):
    index, n_inserted, n_dups = get_index(df, data_key)
    note = f", {n_dups:,} duplicate (date, city) rows skipped" if n_dups else ""
    st.write(f"Index built: **{n_inserted:,}** readings inserted{note}.")

    # 1. Hash table ------------------------------------------------------
    st.subheader("1 · Hash table: region → B+ Tree")
    st.dataframe(
        pd.DataFrame(
            [{"Bucket": i, "Regions stored": ", ".join(names) if names else "(empty)"}
             for i, names in index.bucket_table()]
        ),
        hide_index=True, use_container_width=True,
    )
    st.caption("Looking up a region is O(1) on average. Two regions in one bucket is a collision, "
               "handled by chaining (a short list inside the bucket).")

    size_rows = []
    for name in index.regions():
        t = index.tree(name)
        size_rows.append({
            "Region": name,
            "Hash bucket": index.bucket_of(name),
            "Records": t.count,
            "B+ Tree height": t.height(),
            "Leaf nodes": t.leaf_count(),
            "Binary-tree height (about log2 n)": ceil(log2(t.count + 1)) if t.count else 0,
        })
    st.dataframe(pd.DataFrame(size_rows), hide_index=True, use_container_width=True)
    st.caption("The B+ Tree stays only a few levels tall because each node holds many keys, "
               "while a binary tree (BST or AVL) of the same size needs far more levels.")

    # 2. Range query -----------------------------------------------------
    st.subheader(f"2 · Range query: {region}, {start} to {end}")
    tree = index.tree(region)
    found, stats = index.range_query(region, to_int(start), to_int(end))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records found", f"{len(found):,}")
    m2.metric("Tree height (levels)", stats["height"])
    m3.metric("Leaves visited", f"{stats['leaves_visited']:,}")
    m4.metric("Records examined", f"{stats['records_examined']:,}")
    st.caption(f"The tree went down one root-to-leaf path, then followed the leaf chain. "
               f"A full scan of this region would examine all {tree.count:,} records.")

    keys_in_result = set(zip(date_ints(result["date"]).tolist(), city_labels(result)))
    live_added, live_deleted = st.session_state["live_added"], st.session_state["live_deleted"]
    lo_i, hi_i = to_int(start), to_int(end)
    expected = keys_in_result | {k for (reg, k) in live_added if reg == region and lo_i <= k[0] <= hi_i}
    expected = expected - {k for (reg, k) in live_deleted if reg == region}
    found_keys = {k for k, _ in found}
    live_note = (f" (after {len(live_added)} live insert(s) and {len(live_deleted)} live delete(s) from section 5)"
                 if live_added or live_deleted else "")
    if found_keys == expected and len(found) == len(found_keys):
        st.success("The B+ Tree returned exactly the same readings as the pandas filter" + live_note + ".")
    else:
        st.warning(f"B+ Tree found {len(found):,} readings, the pandas filter has {len(expected):,} "
                   "distinct (date, city) readings. Check the data for repeated rows.")

    if found:
        preview = pd.DataFrame(
            [{"Date": fmt_key(k).split(" ", 1)[0], "City": k[1], "Tmax": r[0], "Tmin": r[1], "Rain": r[2]}
             for k, r in found[:8]]
        )
        st.caption("First rows, already in time order straight from the leaf chain (no sorting needed):")
        st.dataframe(preview, hide_index=True, use_container_width=True)

    # 2b. Region + season query ------------------------------------------
    st.subheader(f"2b · Region + season query: {region}")
    st.caption("The case study asks for region-season queries. Season is not a hash key: it is a set of months "
               "(winter Jan-Feb, pre-monsoon Mar-May, monsoon Jun-Sep, post-monsoon Oct-Dec), so each year's "
               "season is one time-window range query on the same tree.")
    sq1, sq2, sq3 = st.columns(3)
    season = sq1.selectbox("Season", list(SEASONS), index=1, key="sq_season")
    y_min, y_max = min_date.year, max_date.year
    sq_from = sq2.number_input("From year", min_value=y_min, max_value=y_max,
                               value=max(y_min, y_max - 4), step=1, key="sq_from")
    sq_to = sq3.number_input("To year", min_value=y_min, max_value=y_max, value=y_max, step=1, key="sq_to")
    if sq_from > sq_to:
        st.warning("From year must not be after To year.")
    else:
        per_year = index.season_query(region, season, int(sq_from), int(sq_to))
        season_rows, n_total, n_leaves, n_examined = [], 0, 0, 0
        for item in per_year:
            tm = [r[0] for _, r in item["records"]]
            n_total += len(tm)
            n_leaves += item["stats"]["leaves_visited"]
            n_examined += item["stats"]["records_examined"]
            season_rows.append({
                "Year": item["year"], "Readings": len(tm),
                "Mean Tmax (°C)": round(sum(tm) / len(tm), 2) if tm else None,
                "Highest Tmax (°C)": round(max(tm), 1) if tm else None,
                "Leaves visited": item["stats"]["leaves_visited"],
            })
        q1, q2, q3 = st.columns(3)
        q1.metric("Readings found", f"{n_total:,}")
        q2.metric("Range queries run", len(per_year))
        q3.metric("Records examined", f"{n_examined:,}")
        st.dataframe(pd.DataFrame(season_rows), hide_index=True, use_container_width=True)
        if live_added or live_deleted:
            st.caption("Live changes from section 5 are in the index, so the pandas cross-check is skipped.")
        else:
            m1, m2 = SEASONS[season]
            in_season = df[(df["region"] == region) & df["date"].dt.month.between(m1, m2)
                           & df["date"].dt.year.between(int(sq_from), int(sq_to))]
            if len(in_season) == n_total:
                st.success(f"The {len(per_year)} range queries returned exactly the same {n_total:,} readings "
                           "as filtering by month and year in pandas.")
            else:
                st.warning(f"B+ Tree found {n_total:,} readings, pandas found {len(in_season):,}.")

    # 3. Exact search ----------------------------------------------------
    st.subheader("3 · Search for one exact reading")
    city_options = sorted({k[1] for k in expected})
    s1, s2 = st.columns(2)
    q_date = s1.date_input("Date", value=start, min_value=min_date, max_value=max_date, key="ds_date")
    q_city = s2.selectbox("City / station (lat_lon if the CSV has no city column)", city_options, key="ds_city")
    hit = tree.search((to_int(q_date), q_city))
    if hit:
        st.success(f"Found: Tmax {hit[0]:.2f} °C, Tmin {hit[1]:.2f} °C, Rain {hit[2]:.2f}")
    else:
        st.warning("No reading with that (date, city) key in this region.")

    # 4. Tiny tree you can watch -----------------------------------------
    st.subheader("4 · Watch the tree split and merge (tiny version, 4 keys per node)")
    sample = tree.first_items(20)
    if len(sample) < 2:
        st.info("This region needs at least 2 readings for the demo.")
    else:
        t1, t2 = st.columns(2)
        n_ins = t1.slider("Readings inserted (oldest first)", 1, len(sample), min(9, len(sample)), key="ds_ins")
        n_del = t2.slider("Then delete this many (oldest first)", 0, n_ins, 0, key="ds_del")
        tiny = BPlusTree(max_keys=4)
        for k, r in sample[:n_ins]:
            tiny.insert(k, r)
        for k, _ in sample[:n_del]:
            tiny.delete(k)
        st.code(tiny.dump(), language=None)
        st.caption(f"Records: {tiny.count}, height: {tiny.height()}. "
                   "`Node | ... |` holds routing keys only, `Leaf [...]` holds the readings, and "
                   "`-> next leaf` is the link used by range queries. Raise the first slider to see "
                   "leaves split; raise the second to see siblings borrow keys or merge.")

    # 5. Live feed: insert / delete on the real index ---------------------
    st.subheader("5 · Live feed: insert and delete readings in the real index")
    st.caption(
        "AWS stations keep sending new readings. Each insert below goes into this region's B+ Tree in O(log n), "
        "with no re-sorting. Changes last for this browser session only; the CSV file is not modified."
    )
    rdf = df[df["region"] == region]
    rlabels = city_labels(rdf)
    loc = dict(zip(rlabels, zip(rdf["lat"].tolist(), rdf["lon"].tolist())))
    region_cities = sorted(loc)
    last_row = rdf.assign(city=rlabels).sort_values("date").groupby("city").tail(1)
    stations = {r.city: (float(r.tmax), float(r.lat), float(r.lon)) for r in last_row.itertuples()}
    newest = tree.last_item()
    default_day = max_date + timedelta(days=1)

    def say(text, warn=False):
        st.session_state["live_msg"] = (text, warn)
        st.session_state["live_log"].append(text)

    def live_insert(idx, reg, locs, added, deleted):
        key = (to_int(st.session_state["live_date"]), st.session_state[f"live_city_{reg}"])
        la, lo = locs.get(key[1], (float("nan"), float("nan")))
        rec = (float(st.session_state["live_tmax"]), float("nan"), float("nan"), la, lo)
        if idx.insert(reg, key, rec):
            added[(reg, key)] = rec
            deleted.discard((reg, key))
            say(f"Inserted {fmt_key(key)}: Tmax {rec[0]:.1f} °C")
        else:
            say(f"Not inserted: a reading for {fmt_key(key)} already exists (keys must be unique).", warn=True)

    def live_delete(idx, reg, added, deleted):
        key = (to_int(st.session_state["live_date"]), st.session_state[f"live_city_{reg}"])
        if idx.delete(reg, key):
            deleted.add((reg, key))
            added.pop((reg, key), None)
            say(f"Deleted {fmt_key(key)}")
        else:
            say(f"Nothing deleted: no reading for {fmt_key(key)} in this region.", warn=True)

    def live_simulate(idx, reg, stns, added, deleted):
        t = idx.tree(reg)
        d = t.last_item()[0][0]
        base = date(d // 10000, d // 100 % 100, d % 100)
        n_new = 0
        for i in range(1, 6):
            day = base + timedelta(days=i)
            for city, (t0, la, lo) in stns.items():
                key = (to_int(day), city)
                rec = (round(t0 + random.uniform(-1.5, 1.5), 2), float("nan"), float("nan"), la, lo)
                if idx.insert(reg, key, rec):
                    added[(reg, key)] = rec
                    deleted.discard((reg, key))
                    n_new += 1
        say(f"Simulated feed: {n_new} new readings ({len(stns)} stations x 5 days after "
            f"{base:%d %b %Y}). The values are random, for demonstration only.")

    def reset_live():
        st.session_state["index_key"] = None

    f1, f2, f3 = st.columns(3)
    f1.selectbox("Station", region_cities, key=f"live_city_{region}")
    f2.date_input("Date", value=default_day, key="live_date")
    f3.number_input("Tmax (°C)", min_value=0.0, max_value=60.0, value=41.0, step=0.1, key="live_tmax")
    g1, g2, g3, g4 = st.columns(4)
    args = (index, region, loc, live_added, live_deleted)
    g1.button("Insert reading", on_click=live_insert, args=args, use_container_width=True)
    g2.button("Delete reading", on_click=live_delete, args=(index, region, live_added, live_deleted),
              use_container_width=True)
    g3.button("Simulate 5 days of AWS data", on_click=live_simulate,
              args=(index, region, stations, live_added, live_deleted), use_container_width=True)
    g4.button("Reset live changes", on_click=reset_live, use_container_width=True)

    msg = st.session_state.get("live_msg")
    if msg:
        (st.warning if msg[1] else st.success)(msg[0])
    st.caption(
        f"{region} tree now holds {tree.count:,} readings (height {tree.height()}, {tree.leaf_count()} leaves). "
        f"Newest reading in the tree: {fmt_key(newest[0]) if newest else 'none'}. "
        f"Live changes this session: {len(live_added)} inserted, {len(live_deleted)} deleted."
    )
    if st.session_state["live_log"]:
        st.code("\n".join(reversed(st.session_state["live_log"][-8:])), language=None)

    # 6. Benchmark --------------------------------------------------------
    st.subheader("6 · Benchmark: B+ Tree range query vs scanning a plain list")
    st.caption(
        "Same data, same 30-day windows. The list is the 'simple storage' option from the report: it must check "
        "every record, while the tree descends once and follows the leaf chain."
    )
    if st.button("Run benchmark (about 5 seconds)"):
        with st.spinner("Timing both methods..."):
            st.session_state["bench"] = run_benchmark()
    bench = st.session_state.get("bench")
    if bench is not None:
        st.dataframe(bench, hide_index=True, use_container_width=True)
        bfig = px.line(bench, x="Records (n)", y=["B+ Tree query (ms)", "List scan (ms)"], markers=True,
                       log_y=True, title="Time for one range query (log scale)",
                       labels={"value": "Milliseconds per query", "variable": "Method"})
        st.plotly_chart(bfig, use_container_width=True)
        if not bench["Same result"].all():
            st.error("The two methods returned different results. Do not trust this run.")
        st.caption("The list scan grows in proportion to n (O(n)); the tree query barely changes (O(log n + k)). "
                   "Absolute times depend on the machine, so compare the trend and the speed-up column.")
