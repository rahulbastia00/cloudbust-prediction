import streamlit as st
import pandas as pd
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="☁️ Cloudburst Predictor",
    page_icon="⛈️",
    layout="centered",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (clean dark weather aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stApp { background: linear-gradient(160deg, #0d1117 60%, #0e1e2e 100%); }

div[data-testid="stForm"] {
    background: rgba(22, 35, 50, 0.85);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2rem;
    backdrop-filter: blur(8px);
}

div.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    padding: 0.65rem 2rem;
    width: 100%;
    transition: transform 0.15s, box-shadow 0.15s;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(56,139,253,0.35);
}

.result-box {
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-top: 1.5rem;
    text-align: center;
    font-family: 'Space Mono', monospace;
}
.result-danger {
    background: linear-gradient(135deg, #3d1111, #5a1a1a);
    border: 2px solid #f85149;
    color: #ffa198;
}
.result-safe {
    background: linear-gradient(135deg, #0f2d1f, #1a4030);
    border: 2px solid #3fb950;
    color: #7ee787;
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-label { font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0; }
.result-sub { font-size: 0.9rem; opacity: 0.75; }

.feature-card {
    background: rgba(22, 35, 50, 0.7);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.feature-card .val { font-size: 1.4rem; font-weight: 700; color: #79c0ff; font-family: 'Space Mono', monospace; }
.feature-card .lbl { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }

.step-badge {
    display: inline-block;
    background: #1f6feb;
    color: white;
    border-radius: 50%;
    width: 26px; height: 26px;
    line-height: 26px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL DEFINITION  (must match training code)
# ─────────────────────────────────────────────
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, gru_output):
        attn_weights = self.attention(gru_output)          # (batch, seq, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * gru_output, dim=1)
        return context_vector, attn_weights


class CloudburstAttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=0.4)
        self.attention = AttentionBlock(hidden_size)
        self.bn  = nn.BatchNorm1d(hidden_size)
        self.fc  = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _         = self.gru(x)
        context, _     = self.attention(out)
        out            = self.bn(context)
        return self.fc(out)


# ─────────────────────────────────────────────
#  LOAD MODEL + SCALER  (cached so it only runs once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """
    Load the saved scaler and model weights.
    Place  best_cloud_model_attn.pth  and  cloudburst_scaler_attn.pkl
    in the SAME folder as this script.
    """
    try:
        scaler = joblib.load("cloudburst_scaler_attn.pkl")
        # Derive input_size from scaler
        input_size = scaler.n_features_in_
        model = CloudburstAttentionGRU(input_size=input_size)
        model.load_state_dict(
            torch.load("best_cloud_model_attn.pth", map_location="cpu")
        )
        model.eval()
        return scaler, model, None          # None = no error
    except FileNotFoundError as e:
        return None, None, str(e)


scaler, model, load_error = load_artifacts()

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "temp", "hum", "press", "cloud", "soil_m",
    "press_diff", "hum_diff", "temp_diff",
    "dew_point_dep", "press_lag1", "temp_lag1",
    "rain_roll3", "rain_roll6",
    "hour", "month",
]

@st.cache_data(ttl=3600)
def get_coordinates(place: str):
    geolocator = Nominatim(user_agent="cloudburst_predictor_app_v1")
    try:
        loc = geolocator.geocode(f"{place}, India", timeout=10)
        if loc:
            return loc.latitude, loc.longitude, None
        # Fallback
        parts = place.split(",")
        if len(parts) >= 2:
            fb = f"{parts[-2].strip()}, {parts[-1].strip()}, India"
            loc = geolocator.geocode(fb, timeout=10)
            if loc:
                return loc.latitude, loc.longitude, None
        return None, None, "Location not found. Try a broader name (e.g. 'Kullu, Himachal Pradesh')."
    except Exception as e:
        return None, None, str(e)


def fetch_weather(lat, lon, start_date, end_date):
    """Fetch hourly weather from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "surface_pressure,cloudcover,soil_moisture_0_to_7cm"
        ),
        "timezone": "GMT",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame({
            "time":   h["time"],
            "temp":   h["temperature_2m"],
            "hum":    h["relative_humidity_2m"],
            "precip": h["precipitation"],
            "press":  h["surface_pressure"],
            "cloud":  h["cloudcover"],
            "soil_m": h["soil_moisture_0_to_7cm"],
        })
        df["time"] = pd.to_datetime(df["time"])
        return df, None
    except Exception as e:
        return None, str(e)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate ALL feature engineering from the training script."""
    df = df.sort_values("time").copy()

    df["press_diff"]    = df["press"].diff().fillna(0)
    df["hum_diff"]      = df["hum"].diff().fillna(0)
    df["temp_diff"]     = df["temp"].diff().fillna(0)
    df["dew_point_dep"] = df["temp"] - ((100 - df["hum"]) / 5)
    df["press_lag1"]    = df["press"].shift(1).bfill()
    df["temp_lag1"]     = df["temp"].shift(1).bfill()
    df["rain_roll3"]    = df["precip"].rolling(3).sum().fillna(0)
    df["rain_roll6"]    = df["precip"].rolling(6).sum().fillna(0)
    df["hour"]          = df["time"].dt.hour
    df["month"]         = df["time"].dt.month

    return df.dropna()


WINDOW_SIZE = 12   # must match training

def build_sequence(df: pd.DataFrame, scaler, target_time: datetime):
    """
    Extract the 12-hour window ending AT target_time and scale it.
    Returns a (1, 12, n_features) tensor or None if not enough history.
    """
    # Keep only rows up to and including target_time
    window_df = df[df["time"] <= target_time].tail(WINDOW_SIZE)

    if len(window_df) < WINDOW_SIZE:
        return None, (
            f"Not enough historical rows before the selected time "
            f"(need {WINDOW_SIZE}, got {len(window_df)}). "
            "Try selecting a later hour or fetch more days of data."
        )

    X = window_df[FEATURE_COLS].values          # (12, n_features)
    X_scaled = scaler.transform(X)              # normalise
    tensor = torch.FloatTensor(X_scaled).unsqueeze(0)  # (1, 12, n_features)
    return tensor, None


def predict(model, tensor):
    with torch.no_grad():
        logit = model(tensor).squeeze().item()
        prob  = torch.sigmoid(torch.tensor(logit)).item()
    return prob


# ─────────────────────────────────────────────
#  UI  — HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem;'>
    <div style='font-size:3.5rem;'>⛈️</div>
    <h1 style='margin:0.2rem 0 0.4rem; font-size:2rem;'>Cloudburst Predictor</h1>
    <p style='color:#8b949e; font-size:0.95rem; max-width:520px; margin:auto;'>
        Enter a location, date and time.  The app fetches real weather data 
        from Open-Meteo, runs it through an Attention-GRU model, and tells 
        you whether a cloudburst is likely.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SHOW MODEL-LOAD ERROR  (if any)
# ─────────────────────────────────────────────
if load_error:
    st.error(
        f"⚠️ **Could not load model files.**\n\n"
        f"`{load_error}`\n\n"
        "Make sure **`best_cloud_model_attn.pth`** and "
        "**`cloudburst_scaler_attn.pkl`** are in the same folder as this script."
    )
    st.stop()

# ─────────────────────────────────────────────
#  INPUT FORM
# ─────────────────────────────────────────────
st.markdown("---")
with st.form("prediction_form"):
    st.markdown("### 📍 Prediction Inputs")

    location_input = st.text_input(
        "🗺️ Location (district / area, state)",
        placeholder="e.g.  Kullu, Himachal Pradesh",
        help="Be as specific as possible for better geocoding accuracy.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        pred_date = st.date_input(
            "📅 Date",
            value=datetime.today().date() - timedelta(days=2),   # 2 days ago (archive data available)
            help="The archive API typically has data up to 2–3 days ago.",
        )
    with col2:
        pred_hour = st.slider("🕐 Hour (UTC)", min_value=0, max_value=23, value=12)
    with col3:
        extra_days = st.number_input(
            "📆 Days of history to fetch",
            min_value=1, max_value=30, value=3,
            help="Fetch N days ending on the selected date to build the 12-hour input window.",
        )

    threshold = st.slider(
        "⚙️ Decision Threshold (probability)",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Lower = more sensitive (more alerts). Higher = fewer false positives.",
    )

    submitted = st.form_submit_button("🔍  Run Prediction")

# ─────────────────────────────────────────────
#  PREDICTION PIPELINE
# ─────────────────────────────────────────────
if submitted:
    if not location_input.strip():
        st.warning("Please enter a location name.")
        st.stop()

    target_dt = datetime(pred_date.year, pred_date.month, pred_date.day, pred_hour, 0)

    # ── Step 1: Geocode ──
    with st.status("🌐 Step 1 — Geocoding location…", expanded=True) as status:
        lat, lon, geo_err = get_coordinates(location_input)
        if geo_err:
            status.update(label="❌ Geocoding failed", state="error")
            st.error(geo_err)
            st.stop()
        st.write(f"✅  Found: **{lat:.4f}°N, {lon:.4f}°E**")
        status.update(label="✅ Location found", state="complete")

    # ── Step 2: Fetch weather ──
    with st.status("☁️ Step 2 — Fetching weather data from Open-Meteo…", expanded=True) as status:
        start_dt  = target_dt - timedelta(days=int(extra_days))
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str   = pred_date.strftime("%Y-%m-%d")
        raw_df, fetch_err = fetch_weather(lat, lon, start_str, end_str)
        if fetch_err:
            status.update(label="❌ Weather fetch failed", state="error")
            st.error(fetch_err)
            st.stop()
        st.write(f"✅  Downloaded **{len(raw_df)}** hourly rows  ({start_str} → {end_str})")
        status.update(label="✅ Weather data ready", state="complete")

    # ── Step 3: Feature engineering ──
    with st.status("🔧 Step 3 — Engineering features…", expanded=True) as status:
        feat_df = engineer_features(raw_df)
        st.write(f"✅  Feature matrix shape: **{feat_df.shape}**")
        status.update(label="✅ Features ready", state="complete")

    # ── Step 4: Build sequence & predict ──
    with st.status("🤖 Step 4 — Running Attention-GRU model…", expanded=True) as status:
        tensor, seq_err = build_sequence(feat_df, scaler, target_dt)
        if seq_err:
            status.update(label="❌ Sequence error", state="error")
            st.error(seq_err)
            st.stop()

        prob = predict(model, tensor)
        is_cloudburst = prob >= threshold
        status.update(label="✅ Prediction complete", state="complete")

    # ─────────────────────────────────────────
    #  RESULT CARD
    # ─────────────────────────────────────────
    if is_cloudburst:
        st.markdown(f"""
        <div class="result-box result-danger">
            <div class="result-icon">⛈️</div>
            <div class="result-label">CLOUDBURST LIKELY</div>
            <div class="result-sub">
                Probability: <strong>{prob*100:.1f}%</strong> &nbsp;|&nbsp; Threshold: {threshold*100:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box result-safe">
            <div class="result-icon">🌤️</div>
            <div class="result-label">CONDITIONS NORMAL</div>
            <div class="result-sub">
                Probability: <strong>{prob*100:.1f}%</strong> &nbsp;|&nbsp; Threshold: {threshold*100:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    #  SNAPSHOT OF THE LAST HOUR'S WEATHER
    # ─────────────────────────────────────────
    st.markdown("### 🌡️ Weather Snapshot at Prediction Time")
    snap = feat_df[feat_df["time"] <= target_dt].tail(1)
    if not snap.empty:
        r = snap.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics = [
            (c1, "🌡️ Temp",        f"{r['temp']:.1f} °C"),
            (c2, "💧 Humidity",    f"{r['hum']:.0f} %"),
            (c3, "🌬️ Pressure",   f"{r['press']:.0f} hPa"),
            (c4, "☁️ Cloud Cover", f"{r['cloud']:.0f} %"),
            (c5, "🌧️ Rain (6h)",   f"{r['rain_roll6']:.1f} mm"),
        ]
        for col, lbl, val in metrics:
            with col:
                st.markdown(f"""
                <div class="feature-card">
                    <div class="val">{val}</div>
                    <div class="lbl">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    #  RAW DATA EXPANDER
    # ─────────────────────────────────────────
    with st.expander("📊 View raw fetched data"):
        st.dataframe(raw_df.tail(48), use_container_width=True)

# ─────────────────────────────────────────────
#  FOOTER / HOW-TO
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    <span class="step-badge">1</span> **Type the location** — district + state works best (e.g. *Kullu, Himachal Pradesh*).  
    <span class="step-badge">2</span> **Pick a date** — the Open-Meteo archive needs data to be at least 2–3 days old.  
    <span class="step-badge">3</span> **Choose the hour (UTC)** — the model predicts for that specific hour.  
    <span class="step-badge">4</span> **Days of history** — the model needs ≥12 hours of prior data; fetching 3 days is safe.  
    <span class="step-badge">5</span> **Threshold** — 0.50 is balanced; lower it if you want earlier warnings.  

    **Model files needed** (place in the same folder as this script):
    - `best_cloud_model_attn.pth`
    - `cloudburst_scaler_attn.pkl`

    **Run the app:**
    ```bash
    pip install streamlit torch joblib geopy requests scikit-learn
    streamlit run cloudburst_app.py
    ```
    """, unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#30363d; font-size:0.8rem; margin-top:2rem;'>"
    "Attention-GRU · Open-Meteo Archive · Built with Streamlit</p>",
    unsafe_allow_html=True,
)