"""
heatwave_analysis.py
Heat-wave monitoring and early-warning logic for the Heatwave Watch dashboard.

Heat-wave rules follow the IMD criteria:
  * a station-level threshold on Tmax: 40 C plains, 37 C coastal, 30 C hilly
  * Heat wave        : Tmax >= threshold and departure from normal >= 4.5 C
  * Severe heat wave : Tmax >= threshold and departure from normal  > 6.4 C
  * Plains only      : Tmax >= 45 C is a heat wave, Tmax >= 47 C is a severe heat wave
  * A spell must last at least 2 consecutive days to count as a heat-wave event

"Normal" here is the long-term average Tmax for that calendar day over all years in the
dataset (smoothed over 15 days), not IMD's official 30-year normal. Each region's daily
value is the average over the stations in the region.
"""
import numpy as np
import pandas as pd

THRESHOLDS = {"Plains": 40.0, "Coastal": 37.0, "Hilly": 30.0}
DEP_HEATWAVE = 4.5
DEP_SEVERE = 6.4
ABS_HEATWAVE = 45.0      # plains only
ABS_SEVERE = 47.0        # plains only
MIN_SPELL_DAYS = 2
SMOOTH_DAYS = 15         # odd number

LEVELS = {
    0: {"emoji": "🟢", "colour": "GREEN", "label": "No heat-wave warning", "hex": "#2e9e5b"},
    1: {"emoji": "🟡", "colour": "YELLOW", "label": "Heat alert (watch)", "hex": "#d4a800"},
    2: {"emoji": "🟠", "colour": "ORANGE", "label": "Heat wave warning", "hex": "#f07f1c"},
    3: {"emoji": "🔴", "colour": "RED", "label": "Severe heat wave warning", "hex": "#d63b3b"},
}

ADVISORIES = {
    0: {
        "Citizens": ["No heat-wave warning right now.",
                     "Drink water regularly and avoid long exposure to the midday sun."],
        "Farmers": ["Normal field work can continue.",
                    "Keep irrigation plans ready in case temperatures rise."],
        "Health departments": ["No special measures needed.",
                               "Keep ORS and cooling supplies stocked at the start of the hot season."],
        "Authorities": ["Continue routine monitoring.",
                        "Keep the heat action plan and contact lists up to date."],
    },
    1: {
        "Citizens": ["Temperatures are high or rising: drink water often, even if not thirsty.",
                     "Avoid heavy work outdoors from about 12:00 to 15:00 and wear light, loose cotton clothes."],
        "Farmers": ["Do field work in the early morning or evening.",
                    "Irrigate in the cooler hours and make sure livestock have shade and water."],
        "Health departments": ["Stock ORS, IV fluids and cooling supplies.",
                               "Brief staff on recognising heat exhaustion and heat stroke."],
        "Authorities": ["Issue a heat alert through local channels.",
                        "Check that drinking-water points and shaded rest areas are working."],
    },
    2: {
        "Citizens": ["Stay indoors during the hottest hours and avoid strenuous activity outside.",
                     "Take extra care of children, older people, pregnant women and people with chronic illness."],
        "Farmers": ["Avoid field work in the afternoon.",
                    "Irrigate more often and protect crops and livestock from heat stress."],
        "Health departments": ["Keep beds and cooling facilities ready for heat-stroke cases.",
                               "Treat confusion, fainting or very high body temperature as an emergency."],
        "Authorities": ["Activate the heat action plan: cooling centres, water distribution, adjusted hours for outdoor workers.",
                        "Broadcast daily advisories in local languages."],
    },
    3: {
        "Citizens": ["Avoid going out unless essential and stay in cool, ventilated places.",
                     "Seek medical help immediately for fainting, confusion or a very high body temperature."],
        "Farmers": ["Suspend field work during the day.",
                    "Protect animals and sensitive crops and make sure water is available for livestock."],
        "Health departments": ["Run hospitals on a heat-emergency footing with extra staff and cooling capacity.",
                               "Record and track heat-related illness every day."],
        "Authorities": ["Treat as an emergency: open cooling centres, restrict outdoor work in peak hours, review school timings.",
                        "Coordinate with disaster-management teams and check power and water supply."],
    },
}


# ----------------------------------------------------------------------
# Criteria
# ----------------------------------------------------------------------
def terrain_for(region, override=None):
    """Terrain class used for the heat-wave threshold."""
    if override and override != "Auto":
        return override
    name = str(region).lower()
    if "coast" in name:
        return "Coastal"
    if "himal" in name or "hill" in name or "mountain" in name:
        return "Hilly"
    return "Plains"


def criteria_text(terrain):
    thr = THRESHOLDS[terrain]
    text = (f"Tmax of at least {thr:.0f} °C and at least {DEP_HEATWAVE} °C above normal "
            f"(severe: more than {DEP_SEVERE} °C above normal)")
    if terrain == "Plains":
        text += f", or Tmax of {ABS_HEATWAVE:.0f} °C or more (severe: {ABS_SEVERE:.0f} °C or more)"
    return text + f", for at least {MIN_SPELL_DAYS} consecutive days."


# ----------------------------------------------------------------------
# Daily series, normals, flags
# ----------------------------------------------------------------------
def regional_daily(df):
    """One row per (region, date): average Tmax over the stations in the region."""
    d = df.groupby(["region", "date"], as_index=False)["tmax"].mean()
    d["region"] = d["region"].astype(str)
    return d.sort_values(["region", "date"]).reset_index(drop=True)


def add_normals(daily, smooth=SMOOTH_DAYS):
    """Adds `normal`: long-term mean Tmax for each calendar day, smoothed circularly."""
    out = daily.copy()
    out["md"] = out["date"].dt.month * 100 + out["date"].dt.day
    raw = out.groupby(["region", "md"])["tmax"].mean()
    half = smooth // 2
    parts = []
    for region, s in raw.groupby(level=0):
        vals = s.to_numpy()
        keys = s.index.get_level_values(1)
        if len(vals) > smooth:
            padded = np.r_[vals[-half:], vals, vals[:half]]
            vals = np.convolve(padded, np.ones(smooth) / smooth, mode="valid")
        else:
            vals = np.full(len(vals), vals.mean())
        parts.append(pd.DataFrame({"region": region, "md": keys, "normal": vals}))
    out = out.merge(pd.concat(parts, ignore_index=True), on=["region", "md"], how="left")
    return out.drop(columns="md")


def classify(daily, terrain_override=None):
    """Flags heat-wave days and spells. Expects columns region, date, tmax, normal."""
    out = daily.sort_values(["region", "date"]).reset_index(drop=True)
    terrain_map = {r: terrain_for(r, terrain_override) for r in out["region"].unique()}
    out["terrain"] = out["region"].map(terrain_map)
    out["threshold"] = out["terrain"].map(THRESHOLDS)
    out["dep"] = out["tmax"] - out["normal"]

    hot = out["tmax"] >= out["threshold"]
    plains = out["terrain"] == "Plains"
    out["flag"] = (hot & (out["dep"] >= DEP_HEATWAVE)) | (plains & (out["tmax"] >= ABS_HEATWAVE))
    out["severe"] = (hot & (out["dep"] > DEP_SEVERE)) | (plains & (out["tmax"] >= ABS_SEVERE))

    # spells of consecutive flagged calendar days
    grp = out.groupby("region")
    gap = grp["date"].diff().dt.days
    new_segment = (out["flag"] != grp["flag"].shift()) | (gap != 1)
    out["run_id"] = new_segment.cumsum()
    run_len = out.groupby("run_id")["flag"].transform("size")
    run_pos = out.groupby("run_id").cumcount() + 1
    out["run_len"] = np.where(out["flag"], run_len, 0)
    out["run_pos"] = np.where(out["flag"], run_pos, 0)          # day number within the spell so far
    out["event_day"] = out["flag"] & (out["run_len"] >= MIN_SPELL_DAYS)
    return out


def build_events(classified):
    """One row per heat-wave event (a spell of at least 2 flagged days)."""
    cols = ["region", "start", "end", "days", "peak_tmax", "max_dep", "severe_days", "severity"]
    ev = classified[classified["event_day"]]
    if ev.empty:
        return pd.DataFrame(columns=cols)
    events = ev.groupby("run_id").agg(
        region=("region", "first"), start=("date", "min"), end=("date", "max"),
        days=("date", "size"), peak_tmax=("tmax", "max"), max_dep=("dep", "max"),
        severe_days=("severe", "sum"),
    )
    events["severity"] = np.where(events["severe_days"] > 0, "Severe heat wave", "Heat wave")
    return events.sort_values(["region", "start"]).reset_index(drop=True)[cols]


# ----------------------------------------------------------------------
# Early-warning status
# ----------------------------------------------------------------------
def assess(region_days, when):
    """Alert level for one region on the latest day on or before `when`.
    region_days: classified rows of ONE region. Returns a dict, or None if no data."""
    sub = region_days[region_days["date"] <= when]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    run_pos = int(row["run_pos"])
    if row["flag"]:
        if run_pos >= MIN_SPELL_DAYS:
            level = 3 if row["severe"] else 2
            reason = f"Heat-wave criteria have been met for {run_pos} consecutive days."
            if level == 3:
                reason += " Today's temperature is in the severe range."
        else:
            level = 1
            reason = ("Heat-wave criteria are met today (day 1). "
                      "It becomes a heat wave if conditions persist tomorrow.")
    else:
        last3 = sub.tail(3)
        rising = len(last3) == 3 and last3["tmax"].iloc[-1] > last3["tmax"].iloc[0]
        near = row["tmax"] >= row["threshold"] - 2 and row["dep"] >= 3.0
        if rising and near:
            level = 1
            reason = ("Temperatures are building: rising over 3 days, within 2 °C of the threshold "
                      "and at least 3 °C above normal.")
        else:
            level = 0
            reason = "Heat-wave criteria are not met."
    return {
        "level": level, **LEVELS[level], "as_of": row["date"], "tmax": float(row["tmax"]),
        "normal": float(row["normal"]), "dep": float(row["dep"]), "threshold": float(row["threshold"]),
        "terrain": row["terrain"], "run_pos": run_pos, "reason": reason,
    }


# ----------------------------------------------------------------------
# Trends, regional summary, data quality
# ----------------------------------------------------------------------
def yearly_stats(region_days, region_events):
    """Per-year frequency, intensity and duration of heat waves, plus a completeness flag."""
    years = region_days["date"].dt.year
    all_years = np.arange(years.min(), years.max() + 1)
    ed = region_days[region_days["event_day"]]
    stats = pd.DataFrame(index=pd.Index(all_years, name="year"))
    stats["heatwave_days"] = ed.groupby(ed["date"].dt.year).size().reindex(all_years, fill_value=0)
    if len(region_events):
        by_start = region_events.groupby(region_events["start"].dt.year)
        stats["events"] = by_start.size().reindex(all_years, fill_value=0)
        stats["longest_event"] = by_start["days"].max().reindex(all_years, fill_value=0)
    else:
        stats["events"] = 0
        stats["longest_event"] = 0
    stats["mean_departure"] = ed.groupby(ed["date"].dt.year)["dep"].mean().reindex(all_years)   # NaN if none
    days_present = region_days.groupby(years).size().reindex(all_years, fill_value=0)
    stats["complete"] = days_present >= 0.95 * 365
    return stats.reset_index()


def trend_per_decade(years, values):
    """Least-squares slope per decade (NaN values ignored). None if fewer than 5 points."""
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 5:
        return None
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return {"per_decade": float(slope * 10), "slope": float(slope), "intercept": float(intercept)}


def region_summary(classified, events, start, end):
    """Heat-wave statistics per region for the window [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = classified[(classified["date"] >= start) & (classified["date"] <= end)]
    in_window = events[(events["end"] >= start) & (events["start"] <= end)] if len(events) else events
    rows = []
    for region, g in window.groupby("region"):
        ev = in_window[in_window["region"] == region] if len(in_window) else in_window
        rows.append({
            "Region": region,
            "Terrain": g["terrain"].iloc[0],
            "Heat-wave days": int(g["event_day"].sum()),
            "Events": int(len(ev)),
            "Longest event (days)": int(ev["days"].max()) if len(ev) else 0,
            "Peak Tmax (°C)": round(float(g["tmax"].max()), 1),
            "Max departure (°C)": round(float(g["dep"].max()), 1),
        })
    return pd.DataFrame(rows)


def data_quality(df, region):
    """Simple reliability checks on the raw rows of one region."""
    d = df[df["region"] == region]
    if d.empty:
        return None
    dates = d["date"].dt.normalize()
    span = (dates.max() - dates.min()).days + 1
    q = {
        "rows": int(len(d)),
        "first": dates.min(), "last": dates.max(),
        "missing_days": int(span - dates.nunique()),
        "duplicates": int(d.duplicated(subset=["date", "lat", "lon"]).sum()),
        "implausible_tmax": int(((d["tmax"] < 0) | (d["tmax"] > 55)).sum()),
        "tmax_below_tmin": int((d["tmax"] < d["tmin"]).sum()) if "tmin" in d.columns else None,
        "missing_tmin": int(d["tmin"].isna().sum()) if "tmin" in d.columns else None,
        "missing_rain": int(d["Rain"].isna().sum()) if "Rain" in d.columns else None,
    }
    return q


# ----------------------------------------------------------------------
# Short-range outlook (statistical forecast) and its back-test
# ----------------------------------------------------------------------
FORECAST_DAYS = 5
BACKTEST_YEARS = 3


def _calendar_arrays(region_days):
    """Daily calendar arrays (tmax, normal) from first to last date; NaN where tmax is missing.
    The normal for any calendar day comes from the region's own day-of-year normals."""
    rd = region_days.sort_values("date")
    md_normal = rd.assign(md=rd["date"].dt.month * 100 + rd["date"].dt.day).groupby("md")["normal"].mean()
    cal = pd.date_range(rd["date"].min(), rd["date"].max(), freq="D")
    tmax = rd.set_index("date")["tmax"].reindex(cal).to_numpy(dtype=float)
    md = cal.month * 100 + cal.day
    normal = md_normal.reindex(md).to_numpy(dtype=float)
    if md_normal.isna().any() or np.isnan(normal).any():          # 29 Feb in a short record
        normal = pd.Series(normal).interpolate(limit_direction="both").to_numpy()
    return cal, tmax, normal, md_normal


def fit_persistence(dep):
    """Lag-1 autocorrelation of the departure-from-normal series (consecutive days only)."""
    x, y = dep[:-1], dep[1:]
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 30 or np.std(x[ok]) == 0:
        return 0.0
    return float(np.clip(np.corrcoef(x[ok], y[ok])[0, 1], 0.0, 0.99))


def forecast_region(region_days, horizon=FORECAST_DAYS, phi=None):
    """Tmax forecast for the next `horizon` days after the last row of `region_days`:

        forecast(h) = normal(date + h) + phi**h * departure(today)

    i.e. today's departure from normal fades back to normal at rate phi per day. A simple
    statistical baseline, not the AI model in the case study. Returns None if there is no data."""
    if region_days.empty:
        return None
    cal, tmax, normal, md_normal = _calendar_arrays(region_days)
    dep = tmax - normal
    if phi is None:
        phi = fit_persistence(dep)
    last = region_days.sort_values("date").iloc[-1]
    today_dep = float(last["tmax"] - last["normal"])
    rows = []
    for h in range(1, horizon + 1):
        day = last["date"] + pd.Timedelta(days=h)
        md = day.month * 100 + day.day
        nrm = float(md_normal.get(md, md_normal.mean()))
        rows.append({"date": day, "h": h, "normal": nrm, "forecast": nrm + (phi ** h) * today_dep})
    out = pd.DataFrame(rows)
    out["dep"] = out["forecast"] - out["normal"]
    return out, phi


def outlook(region_days, horizon=FORECAST_DAYS):
    """Forecast plus the heat-wave rules applied to the forecast days, continuing today's spell.
    Returns a dict with the forecast table, the expected alert level and a reason, or None."""
    res = forecast_region(region_days, horizon)
    if res is None:
        return None
    fc, phi = res
    last = region_days.sort_values("date").iloc[-1]
    terrain, thr = last["terrain"], float(last["threshold"])
    plains = terrain == "Plains"
    hot = fc["forecast"] >= thr
    fc["flag"] = (hot & (fc["dep"] >= DEP_HEATWAVE)) | (plains & (fc["forecast"] >= ABS_HEATWAVE))
    fc["severe"] = (hot & (fc["dep"] > DEP_SEVERE)) | (plains & (fc["forecast"] >= ABS_SEVERE))
    run = int(last["run_pos"]) if bool(last["flag"]) else 0         # spell already running today
    level, peak_run = 0, run
    for _, r in fc.iterrows():
        run = run + 1 if r["flag"] else 0
        peak_run = max(peak_run, run)
        if r["flag"]:
            level = max(level, 3 if (r["severe"] and run >= MIN_SPELL_DAYS) else
                        2 if run >= MIN_SPELL_DAYS else 1)
    if level == 0:
        reason = "The forecast does not meet the heat-wave criteria in the next %d days." % horizon
    elif level == 1:
        reason = "A single day may meet the heat-wave criteria in the next %d days." % horizon
    else:
        reason = "The heat-wave criteria may be met for %d consecutive days within %d days." % (peak_run, horizon)
    return {"table": fc, "phi": phi, "level": level, **LEVELS[level], "reason": reason,
            "threshold": thr, "terrain": terrain}


def backtest_forecast(region_days, horizon=FORECAST_DAYS, years=BACKTEST_YEARS):
    """Honest check of the forecast on the last `years` years. phi is fitted on the earlier data only.
    Compares MAE (deg C) with two baselines: persistence (tomorrow = today) and climatology (= normal).
    Returns a DataFrame (one row per lead day) or None if there is not enough data."""
    cal, tmax, normal, _ = _calendar_arrays(region_days)
    dep = tmax - normal
    split = len(cal) - int(years * 365)
    if split < 3 * 365:
        return None
    phi = fit_persistence(dep[:split])
    rows = []
    for h in range(1, horizon + 1):
        o = np.arange(split, len(cal) - h)                          # forecast origins
        actual, today = tmax[o + h], tmax[o]
        ok = ~np.isnan(actual) & ~np.isnan(today)
        if ok.sum() < 30:
            continue
        ar1 = normal[o + h] + (phi ** h) * dep[o]
        rows.append({
            "Lead (days)": h,
            "Forecast MAE (°C)": float(np.mean(np.abs(ar1[ok] - actual[ok]))),
            "Persistence MAE (°C)": float(np.mean(np.abs(today[ok] - actual[ok]))),
            "Climatology MAE (°C)": float(np.mean(np.abs(normal[o + h][ok] - actual[ok]))),
            "Days tested": int(ok.sum()),
        })
    return pd.DataFrame(rows) if rows else None
