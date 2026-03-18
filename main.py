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
#  CUSTOM CSS
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
#  MODEL DEFINITION
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
        attn_weights = self.attention(gru_output)
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
#  LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load("cloudburst_scaler_attn.pkl")
        input_size = scaler.n_features_in_
        model = CloudburstAttentionGRU(input_size=input_size)
        model.load_state_dict(torch.load("best_cloud_model_attn.pth", map_location="cpu"))
        model.eval()
        return scaler, model, None
    except Exception as e:
        return None, None, str(e)

scaler, model, load_error = load_artifacts()

# ─────────────────────────────────────────────
#  LOCATIONS LIST
# ─────────────────────────────────────────────
LOCATIONS = [
    "Chamba, Himachal Pradesh", "Nainital, Uttarakhand",
    "Bageshwar, Uttarakhand", "Kullu, Himachal Pradesh",
    "Chamba, Himachal Pradesh", "Uttarkashi, Uttarakhand",
    "Lahaul & Spiti, Himachal Pradesh", "Kinnaur, Himachal Pradesh",
    "Uttarkashi, Uttarakhand", "Kullu, Himachal Pradesh",
    "Kullu, Himachal Pradesh", "Bilaspur, Himachal Pradesh",
    "Anantnag, Jammu and Kashmir", "Bageshwar, Uttarakhand",
    "Pithoragarh, Uttarakhand", "Nainital, Uttarakhand",
    "Kullu, Himachal Pradesh", "Chamba, Himachal Pradesh",
    "Chamba, Himachal Pradesh", "Shimla, Himachal Pradesh",
    "Kinnaur, Himachal Pradesh", "Kinnaur, Himachal Pradesh",
    "Lahaul & Spiti, Himachal Pradesh", "Chamoli, Uttarakhand",
    "Lahaul & Spiti, Himachal Pradesh", "Kullu, Himachal Pradesh",
    "Pithoragarh, Uttarakhand", "Chamoli, Uttarakhand",
    "Pauri Garhwal, Uttarakhand", "Rudraprayag, Uttarakhand",
    "Chamoli, Uttarakhand", "Uttarkashi, Uttarakhand",
    "Dehradun, Uttarakhand", "Dehradun, Uttarakhand",
    "Tehri Garhwal, Uttarakhand", "Pauri Garhwal, Uttarakhand",
    "Uttarkashi, Uttarakhand", "Rudraprayag, Uttarakhand",
    "Tehri Garhwal, Uttarakhand", "Pithoragarh, Uttarakhand",
    "Dehradun, Uttarakhand", "East Sikkim, Sikkim",
    "Anantnag, Jammu and Kashmir", "Kulgam, Jammu and Kashmir",
    "Pulwama, Jammu and Kashmir", "East Sikkim, Sikkim",
    "Kulgam, Jammu and Kashmir", "Kullu, Himachal Pradesh",
    "Leh, Ladakh", "Mandi, Himachal Pradesh",
    "Kullu, Himachal Pradesh", "Kupwara, Jammu and Kashmir",
    "Reasi, Jammu and Kashmir", "Chamba, Himachal Pradesh",
    "Leh, Ladakh", "Srinagar, Jammu and Kashmir",
    "Kulgam, Jammu and Kashmir", "Doda, Jammu and Kashmir",
    "Kullu, Himachal Pradesh", "Leh, Ladakh",
    "Kupwara, Jammu and Kashmir", "Kinnaur, Himachal Pradesh",
    "Ernakulam, Kerala", "Ernakulam, Kerala",
    "Solan, Himachal Pradesh", "Sirmaur, Himachal Pradesh",
    "Mandi, Himachal Pradesh", "Sirmaur, Himachal Pradesh",
    "Kullu, Himachal Pradesh", "Lahaul & Spiti, Himachal Pradesh",
    "Kinnaur, Himachal Pradesh", "Kullu, Himachal Pradesh",
    "Shimla, Himachal Pradesh", "Shimla, Himachal Pradesh",
    "Kullu, Himachal Pradesh", "Mandi, Himachal Pradesh",
    "Shimla, Himachal Pradesh", "Kullu, Himachal Pradesh",
    "Kullu, Himachal Pradesh", "Mandi, Himachal Pradesh",
    "Tehri Garhwal, Uttarakhand", "Mandi, Himachal Pradesh",
    "Lahaul & Spiti, Himachal Pradesh", "Lahaul & Spiti, Himachal Pradesh",
    "Rudraprayag, Uttarakhand", "Lahaul & Spiti, Himachal Pradesh",
    "Lahaul & Spiti, Himachal Pradesh", "Shimla, Himachal Pradesh",
    "Sirmaur, Himachal Pradesh", "Kinnaur, Himachal Pradesh",
    "Kinnaur, Himachal Pradesh", "Shimla, Himachal Pradesh",
    "Shimla, Himachal Pradesh", "Sirmaur, Himachal Pradesh",
    "Kupwara, Jammu and Kashmir", "Kullu, Himachal Pradesh",
    "Kullu, Himachal Pradesh", "Uttarkashi, Uttarakhand",
    "Mandi, Himachal Pradesh", "Mandi, Himachal Pradesh",
    "Mandi, Himachal Pradesh", "Mandi, Himachal Pradesh",
    "Mandi, Himachal Pradesh", "Mandi, Himachal Pradesh",
    "Mandi, Himachal Pradesh", "Kullu, Himachal Pradesh",
    "Uttarkashi, Uttarakhand", "Uttarkashi, Uttarakhand",
    "Pauri Garhwal, Uttarakhand", "Pauri Garhwal, Uttarakhand",
    "Pauri Garhwal, Uttarakhand", "Pauri Garhwal, Uttarakhand",
    "Kishtwar, Jammu and Kashmir", "Jammu and Kashmir",
    "Chamoli, Uttarakhand", "Doda, Jammu and Kashmir",
    "Dehradun, Uttarakhand", "Bageshwar, Uttarakhand",
    "Chennai, Tamil Nadu", "Dehradun, Uttarakhand",
    "Dehradun, Uttarakhand", "Dehradun, Uttarakhand",
    "Kolkata, West Bengal", "Kolkata, West Bengal",
    "Kolkata, West Bengal", "Kolkata, West Bengal",
    "Kolkata, West Bengal"
]

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
FEATURE_COLS = ["temp", "hum", "press", "cloud", "soil_m", "press_diff", "hum_diff", "temp_diff", "dew_point_dep", "press_lag1", "temp_lag1", "rain_roll3", "rain_roll6", "hour", "month"]

@st.cache_data(ttl=3600)
def get_coordinates(place: str):
    geolocator = Nominatim(user_agent="cloudburst_predictor_v1_rahul")
    try:
        loc = geolocator.geocode(f"{place}, India", timeout=10)
        if loc: return loc.latitude, loc.longitude, None
        return None, None, "Location not found. Try 'City, State'."
    except Exception as e:
        return None, None, str(e)

def fetch_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,cloudcover,soil_moisture_0_to_7cm",
        "timezone": "GMT",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame({"time": h["time"], "temp": h["temperature_2m"], "hum": h["relative_humidity_2m"], "precip": h["precipitation"], "press": h["surface_pressure"], "cloud": h["cloudcover"], "soil_m": h["soil_moisture_0_to_7cm"]})
        df["time"] = pd.to_datetime(df["time"])
        return df, None
    except Exception as e:
        return None, str(e)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").copy()
    df["press_diff"] = df["press"].diff().fillna(0)
    df["hum_diff"] = df["hum"].diff().fillna(0)
    df["temp_diff"] = df["temp"].diff().fillna(0)
    df["dew_point_dep"] = df["temp"] - ((100 - df["hum"]) / 5)
    df["press_lag1"] = df["press"].shift(1).bfill()
    df["temp_lag1"] = df["temp"].shift(1).bfill()
    df["rain_roll3"] = df["precip"].rolling(3).sum().fillna(0)
    df["rain_roll6"] = df["precip"].rolling(6).sum().fillna(0)
    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    return df.dropna()

def build_sequence(df: pd.DataFrame, scaler, target_time: datetime):
    window_df = df[df["time"] <= target_time].tail(12)
    if len(window_df) < 12: return None, "Insufficient data history (need 12h)."
    X = window_df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    return torch.FloatTensor(X_scaled).unsqueeze(0), None

def predict(model, tensor):
    with torch.no_grad():
        logit = model(tensor).squeeze().item()
        return torch.sigmoid(torch.tensor(logit)).item()

# ─────────────────────────────────────────────
#  UI — HEADER & FORM
# ─────────────────────────────────────────────
st.markdown("<div style='text-align:center;'><h1>⛈️ Cloudburst Predictor</h1></div>", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ Model load error: {load_error}")
    st.stop()

with st.form("prediction_form"):
    st.markdown("### 📍 Location & Time")
    
    # DROP DOWN AND CUSTOM TOGGLE
    loc_choice = st.selectbox("🗺️ Choose Location", ["Select a predefined location", "Enter Custom Location"] + sorted(LOCATIONS))
    
    location_input = ""
    if loc_choice == "Enter Custom Location":
        location_input = st.text_input("Enter manually", placeholder="District, State")
    elif loc_choice != "Select a predefined location":
        location_input = loc_choice

    col1, col2, col3 = st.columns(3)
    with col1:
        pred_date = st.date_input("📅 Date", value=datetime.today().date() - timedelta(days=2))
    with col2:
        pred_hour = st.slider("🕐 Hour (UTC)", 0, 23, 12)
    with col3:
        extra_days = st.number_input("📆 History Days", 1, 30, 3)

    submitted = st.form_submit_button("🔍 Run Prediction")

# ─────────────────────────────────────────────
#  PREDICTION LOGIC
# ─────────────────────────────────────────────
if submitted:
    if not location_input:
        st.warning("Please select or enter a location.")
        st.stop()

    target_dt = datetime(pred_date.year, pred_date.month, pred_date.day, pred_hour, 0)

    with st.status("Processing...", expanded=True) as status:
        lat, lon, geo_err = get_coordinates(location_input)
        if geo_err:
            st.error(geo_err); st.stop()
        
        raw_df, fetch_err = fetch_weather(lat, lon, (target_dt - timedelta(days=int(extra_days))).strftime("%Y-%m-%d"), pred_date.strftime("%Y-%m-%d"))
        if fetch_err:
            st.error(fetch_err); st.stop()
        
        feat_df = engineer_features(raw_df)
        tensor, seq_err = build_sequence(feat_df, scaler, target_dt)
        if seq_err:
            st.error(seq_err); st.stop()

        prob = predict(model, tensor)
        status.update(label="Analysis Complete", state="complete")

    # RESULTS
    is_cb = prob >= 0.65
    res_class = "result-danger" if is_cb else "result-safe"
    res_icon = "⛈️" if is_cb else "🌤️"
    res_label = "CLOUDBURST LIKELY" if is_cb else "CONDITIONS NORMAL"

    st.markdown(f"""
    <div class="result-box {res_class}">
        <div class="result-icon">{res_icon}</div>
        <div class="result-label">{res_label}</div>
        <div>Prob: <strong>{prob*100:.1f}%</strong> | Threshold: 65%</div>
    </div>
    """, unsafe_allow_html=True)

    # SNAPSHOT
    snap = feat_df[feat_df["time"] <= target_dt].tail(1)
    if not snap.empty:
        r = snap.iloc[0]
        st.markdown("### 🌡️ Current Conditions")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temp", f"{r['temp']:.1f}°C")
        c2.metric("Humidity", f"{r['hum']:.0f}%")
        c3.metric("Pressure", f"{r['press']:.0f} hPa")
        c4.metric("Rain (6h)", f"{r['rain_roll6']:.1f}mm")