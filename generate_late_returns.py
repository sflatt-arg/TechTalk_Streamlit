"""
generate_late_returns.py

Synthetic data generator focused on correlations that cause LATE RETURNS.
- No PII (opaque IDs only)
- Realistic temporal/weather/station effects
- Noisy logistic function to produce the late_return label

Usage:
    python generate_late_returns.py --n 5000 --seed 42 --out data.csv
"""

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ----------------------------
# 0) Columns (announced up front)
# ----------------------------
n_digits=3
COLUMNS = [
    # Observed at checkout (features)
    "ride_id",
    "user_id",
    "start_station",
    "start_ts",
    "weekday",
    "is_weekend",
    "is_holiday",
    "month",
    "planned_duration_min",
    "weather",              # sun/clouds/rain/snow/wind
    "temp_bucket",          # cold/mild/warm
    "wind_bucket",          # calm/breezy/windy
    "event_nearby",         # 0/1

    # Latent / contextual drivers that can be observed or hidden
    "station_congestion_base",     # station trait in [0,1]
    "network_congestion_index",    # time-varying context in [0,1]
    "user_tardiness_propensity",   # user trait in [0,1]

    # Label
    "late_return",          # 0/1
]

# ----------------------------
# 1) Helpers
# ----------------------------
@dataclass
class Catalog:
    stations: np.ndarray       # ['S1', 'S2', ...]
    station_weights: np.ndarray  # Zipf-like popularity
    station_cong_base: np.ndarray  # per-station congestion base in [0,1]
    users: np.ndarray          # ['U1001', ...]
    user_propensity: np.ndarray # per-user tardiness propensity ~ Beta


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Alternative approach: Adjust user propensity distribution to be more skewed toward punctual users
def make_catalogs(n_stations=30, n_users=2000, rng=None):
    """Create stations and users with MORE punctual users overall."""
    rng = rng or np.random.default_rng(0)

    # Stations: same as before
    stations = np.array([f"S{i+1}" for i in range(n_stations)])
    ranks = np.arange(1, n_stations + 1)
    station_weights = 1.0 / ranks
    station_weights = station_weights / station_weights.sum()

    # Station congestion: slightly lower overall
    cong = (station_weights - station_weights.min()) / (station_weights.max() - station_weights.min() + 1e-9)
    station_cong_base = np.clip(cong * 0.8 + rng.normal(0, 0.06, size=n_stations), 0, 1).round(n_digits)  # scaled down

    # Users: MORE skewed toward punctual behavior
    users = np.array([f"U{1000+i}" for i in range(n_users)])
    # Changed from Beta(1.5, 8.0) to Beta(1.2, 12.0) - even more skewed toward low values
    user_propensity = rng.beta(a=1.2, b=12.0, size=n_users).round(n_digits)  # mean ≈ 0.09 instead of 0.16

    return Catalog(
        stations=stations,
        station_weights=station_weights,
        station_cong_base=station_cong_base,
        users=users,
        user_propensity=user_propensity
    )



def sample_calendar(n, start_date="2024-01-01", rng=None):
    """Uncorrelated calendar skeleton: start timestamps across months with daily+hourly patterns."""
    rng = rng or np.random.default_rng(0)
    # Simulate n timestamps within a 6-month window
    base = pd.Timestamp(start_date)
    # Concentrate around commuting peaks and weekend middays
    days = rng.integers(0, 180, size=n)  # within ~6 months
    # Hour mixture: morning peak, evening peak, midday, night
    hour_modes = rng.choice(
        [8, 12, 18, 22],
        size=n,
        p=[0.30, 0.20, 0.40, 0.10]
    )
    hour_jitter = rng.normal(0, 1.0, size=n).round().astype(int)
    hours = np.clip(hour_modes + hour_jitter, 0, 23)
    minutes = rng.integers(0, 60, size=n)

    ts = [base + timedelta(days=int(d), hours=int(h), minutes=int(m)) for d, h, m in zip(days, hours, minutes)]
    ts = pd.to_datetime(ts, utc=False)
    return ts


def month_weather_mix(month):
    """More favorable weather distributions - more sun, less harsh weather."""
    # More sunny days across all seasons
    winter = { "sun": 0.40, "clouds": 0.25, "rain": 0.25, "wind": 0.08, "snow": 0.02 }  # more sun, less snow/rain
    spring = { "sun": 0.55, "clouds": 0.22, "rain": 0.18, "wind": 0.05, "snow": 0.00 }  # more sun, less rain
    summer = { "sun": 0.70, "clouds": 0.18, "rain": 0.10, "wind": 0.02, "snow": 0.00 }  # even more sun
    autumn = { "sun": 0.50, "clouds": 0.25, "rain": 0.20, "wind": 0.05, "snow": 0.00 }  # more sun, less rain

    if month in (12, 1, 2):
        return winter
    if month in (3, 4, 5):
        return spring
    if month in (6, 7, 8):
        return summer
    return autumn


def sample_weather_and_env(ts, rng=None):
    """Correlated generation: weather/temperature/wind by month; weekends/holidays/events."""
    rng = rng or np.random.default_rng(0)
    n = len(ts)

    months = pd.Series(ts).dt.month.values
    weekdays = pd.Series(ts).dt.dayofweek.values  # 0=Mon ... 6=Sun
    is_weekend = (weekdays >= 5).astype(int)

    # Simple holiday flag: flag a few specific dates (toy example)
    dt = pd.Series(ts).dt.normalize()
    holidays = set(pd.to_datetime(["2024-01-01", "2024-04-01", "2024-05-01", "2024-12-25"]))
    is_holiday = dt.isin(holidays).astype(int)

    # Weather per month
    weathers = []
    for m in months:
        mix = month_weather_mix(m)
        cats = np.array(list(mix.keys()))
        probs = np.array(list(mix.values()))
        weathers.append(np.random.choice(cats, p=probs))
    weathers = np.array(weathers)

    # Temp buckets correlated with month
    temp_bucket = []
    for m in months:
        if m in (12,1,2):
            temp_bucket.append(np.random.choice(["cold","mild","warm"], p=[0.75,0.23,0.02]))
        elif m in (6,7,8):
            temp_bucket.append(np.random.choice(["cold","mild","warm"], p=[0.05,0.35,0.60]))
        else:
            temp_bucket.append(np.random.choice(["cold","mild","warm"], p=[0.20,0.55,0.25]))
    temp_bucket = np.array(temp_bucket)

    # Wind bucket (slightly higher in colder months)
    wind_bucket = []
    for m in months:
        if m in (12,1,2):
            wind_bucket.append(np.random.choice(["calm","breezy","windy"], p=[0.55,0.30,0.15]))
        else:
            wind_bucket.append(np.random.choice(["calm","breezy","windy"], p=[0.65,0.28,0.07]))
    wind_bucket = np.array(wind_bucket)

    # Events near popular stations, more often on weekend evenings
    hours = pd.Series(ts).dt.hour.values
    evening = ((hours >= 18) | (hours <= 6)).astype(int)
    # Base event probability + uplift on weekend evenings
    base_p = 0.03
    uplift = 0.10
    p_event = base_p + uplift * (is_weekend & evening)
    event_nearby = (rng.random(n) < p_event).astype(int)

    # Network congestion index (time-varying; spikes on some days/hours)
    daily = pd.Series(ts).dt.date
    day_counts = daily.value_counts().to_dict()
    # More rides on a day -> more congestion
    net_index = np.array([min(1.0, 0.2 + 0.03 * day_counts[daily.iloc[i]]) for i in range(n)])
    # Add hour-based bump at evening commute
    net_index += 0.10 * evening
    net_index = np.clip(net_index + rng.normal(0, 0.03, size=n), 0, 1).round(n_digits)

    return {
        "weekday": weekdays,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday.values,
        "month": months,
        "weather": weathers,
        "temp_bucket": temp_bucket,
        "wind_bucket": wind_bucket,
        "event_nearby": event_nearby,
        "network_congestion_index": net_index,
    }


def sample_planned_duration(n, rng=None):
    """Right-skewed durations (uncorrelated base), in minutes."""
    rng = rng or np.random.default_rng(0)
    # Lognormal centered around ~28 min, long tail to ~120+
    dur = rng.lognormal(mean=3.3, sigma=0.4, size=n)  # exp(3.5) ~ 28
    dur = np.clip(dur, 8, 160)
    return dur.round().astype(int)


# ----------------------------
# 2) Noisy function to create label
# ----------------------------
def compute_late_prob(
    weather, hour, weekend, holiday, log_dur,
    station_cong, net_cong, event_nearby, user_prop,
    temp_bucket, wind_bucket, rng=None
):
    """
    Logistic model with noise. Adjusted coefficients to yield lower base late rate (≈15–25%).
    
    Key changes:
    1. More negative intercept (-2.8 instead of -2.0)
    2. Reduced weather impact coefficients
    3. Scaled down user propensity impact
    4. Reduced noise variance
    5. Added more protective factors
    """
    rng = rng or np.random.default_rng(0)

    # One-hot-ish helpers
    is_rain = (weather == "rain").astype(float)
    is_snow = (weather == "snow").astype(float)
    is_wind = (weather == "wind").astype(float)
    is_sun = (weather == "sun").astype(float)  # NEW: protective factor
    is_evening = ((hour >= 18) | (hour <= 6)).astype(float)
    is_morning_commute = ((hour >= 7) & (hour <= 9)).astype(float)  # NEW: protective factor

    temp_cold = (temp_bucket == "cold").astype(float)
    temp_warm = (temp_bucket == "warm").astype(float)
    temp_mild = (temp_bucket == "mild").astype(float)  # NEW: protective factor
    wind_windy = (wind_bucket == "windy").astype(float)
    wind_calm = (wind_bucket == "calm").astype(float)  # NEW: protective factor

    # ADJUSTED COEFFICIENTS - Lower impact overall
    b0 = -2.5  # More negative baseline (was -2.0)
    
    # Weather effects - reduced impact
    b_rain = 0.5  # reduced from 0.6
    b_snow = 0.6  # reduced from 0.8
    b_wind = 0.08  # reduced from 0.1
    b_sun = 0  # NEW: sunny weather helps punctuality
    
    # Time effects
    b_evening = 0.3  # reduced from 0.4
    b_morning_commute = -0.2  # NEW: morning commuters more punctual
    b_weekend = 0.15  # reduced from 0.25
    b_holiday = 0.1   # reduced from 0.15
    
    # Duration and infrastructure
    b_logdur = 0.2   # reduced from 0.25
    b_station = 0.6  # reduced from 0.8
    b_network = 0.7  # reduced from 0.9
    b_event = 0.3    # reduced from 0.5
    
    # User behavior - significantly reduced
    b_user = 0.6     # reduced from 1.2
    
    # Temperature/wind - add protective factors
    b_cold = 0.08    # reduced from 0.1
    b_warm = -0.08   # slightly more negative (was -0.05)
    b_mild = -0.1    # NEW: mild weather helps
    b_windy = 0.15   # reduced from 0.25
    b_calm = -0.1    # NEW: calm weather helps

    # Interactions - reduced impact
    g_rain_evening = 0.15      # reduced from 0.25
    g_logdur_weekend = 0.05    # reduced from 0.10
    g_station_event = 0.2      # reduced from 0.30
    g_sun_morning = -0.1       # NEW: sunny mornings are great for punctuality

    # Linear index
    logit = (
        b0
        + b_rain * is_rain
        + b_snow * is_snow
        + b_wind * is_wind
        + b_sun * is_sun
        + b_evening * is_evening
        + b_morning_commute * is_morning_commute
        + b_weekend * weekend
        + b_holiday * holiday
        + b_logdur * log_dur
        + b_station * station_cong
        + b_network * net_cong
        + b_event * event_nearby
        + b_user * user_prop
        + b_cold * temp_cold
        + b_warm * temp_warm
        + b_mild * temp_mild
        + b_windy * wind_windy
        + b_calm * wind_calm
        + g_rain_evening * is_rain * is_evening
        + g_logdur_weekend * log_dur * weekend
        + g_station_event * station_cong * event_nearby
        + g_sun_morning * is_sun * is_morning_commute
        + rng.normal(0, 0.15, size=len(is_rain))  
    )

    return sigmoid(logit)


# ----------------------------
# 3) Main generator
# ----------------------------
def generate(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    # Catalogs
    cat = make_catalogs(n_stations=5, n_users=2000, rng=rng)

    # Calendar & environment (correlated)
    start_ts = sample_calendar(n, start_date="2024-01-01", rng=rng)
    env = sample_weather_and_env(start_ts, rng=rng)

    # Stations (sample by popularity) + their congestion trait
    station_idx = rng.choice(np.arange(len(cat.stations)), size=n, p=cat.station_weights)
    start_station = cat.stations[station_idx]
    station_cong = cat.station_cong_base[station_idx]

    # Users (uniform sample) + their propensity trait
    user_idx = rng.integers(0, len(cat.users), size=n)
    user_id = cat.users[user_idx]
    user_prop = cat.user_propensity[user_idx]

    # Planned durations
    planned_duration = sample_planned_duration(n, rng=rng)
    logdur = np.log(planned_duration)

    # Weather etc (from env)
    weather = env["weather"]
    temp_bucket = env["temp_bucket"]
    wind_bucket = env["wind_bucket"]
    weekday = env["weekday"]
    is_weekend = env["is_weekend"].astype(float)
    is_holiday = env["is_holiday"].astype(float)
    month = env["month"]
    event_nearby = env["event_nearby"].astype(float)
    net_cong = env["network_congestion_index"]

    # Hour for evening detection inside the late prob function
    hour = pd.Series(start_ts).dt.hour.values

    # Compute probability and sample label
    p_late = compute_late_prob(
        weather=weather,
        hour=hour,
        weekend=is_weekend,
        holiday=is_holiday,
        log_dur=np.log(planned_duration),
        station_cong=station_cong,
        net_cong=net_cong,
        event_nearby=event_nearby,
        user_prop=user_prop,
        temp_bucket=temp_bucket,
        wind_bucket=wind_bucket,
        rng=rng
    )
    late_return = (rng.random(n) < p_late).astype(int)

    # Build DataFrame
    df = pd.DataFrame({
        "ride_id": [f"R{i:05d}" for i in range(1, n+1)],
        "user_id": user_id,
        "start_station": start_station,
        "start_ts": start_ts,
        "weekday": weekday,                           # 0=Mon ... 6=Sun
        "is_weekend": is_weekend.astype(int),
        "is_holiday": is_holiday.astype(int),
        "month": month,
        "planned_duration_min": planned_duration,
        "weather": weather,
        "temp_bucket": temp_bucket,
        "wind_bucket": wind_bucket,
        "event_nearby": event_nearby.astype(int),
        "station_congestion_base": station_cong,
        "network_congestion_index": net_cong,
        "user_tardiness_propensity": user_prop,
        "late_return": late_return,
    })[COLUMNS]  # enforce column order

    return df


# ----------------------------
# 4) CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ride data with late-return label.")
    parser.add_argument("--n", type=int, default=5000, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="", help="Optional path to CSV output")
    args = parser.parse_args()

    print("Columns:")
    for c in COLUMNS:
        print(f" - {c}")

    df = generate(n=args.n, seed=args.seed)
    print("\nPreview:")
    print(df.head(5).to_string(index=False))

    late_rate = df["late_return"].mean()
    print(f"\nRows: {len(df):,} | Late rate: {late_rate:.3f}")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
