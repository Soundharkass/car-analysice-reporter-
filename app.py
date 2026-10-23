"""
Car Video Analyzer - Streamlit app
Upload a car video -> detect & track the car -> identify brand/model (CLIP zero-shot)
-> measure observed speed/acceleration -> look up specs -> recommend similar cars
-> simulate a similar car's acceleration.

Run:  streamlit run app.py
"""
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Car Video Analyzer", page_icon="🚗", layout="wide")

# ----------------------------------------------------------------------------
# Specs database (APPROXIMATE figures - edit or upload your own CSV in the sidebar)
# columns: brand, model, hp, accel_0_100 (s), top_speed (km/h), weight (kg)
# ----------------------------------------------------------------------------
DEFAULT_SPECS = pd.DataFrame(
    [
        ("Toyota", "Camry", 203, 8.0, 210, 1500),
        ("Toyota", "Supra", 382, 4.1, 250, 1540),
        ("Toyota", "Fortuner", 204, 10.0, 180, 2100),
        ("Honda", "Civic", 158, 9.0, 200, 1300),
        ("Hyundai", "Elantra", 147, 9.0, 200, 1300),
        ("Maruti Suzuki", "Swift", 89, 12.0, 165, 900),
        ("Tata", "Nexon", 118, 11.0, 180, 1250),
        ("Mahindra", "Thar", 150, 11.0, 155, 1750),
        ("Ford", "Mustang GT", 450, 4.5, 250, 1700),
        ("Chevrolet", "Corvette Stingray", 495, 3.0, 312, 1530),
        ("BMW", "3 Series", 255, 5.6, 250, 1600),
        ("BMW", "M3", 503, 3.5, 290, 1730),
        ("Mercedes-Benz", "C-Class", 255, 6.0, 250, 1600),
        ("Mercedes-AMG", "GT", 577, 3.7, 318, 1615),
        ("Audi", "A4", 261, 5.7, 250, 1600),
        ("Audi", "R8", 602, 3.4, 331, 1670),
        ("Porsche", "911 Carrera", 379, 4.2, 293, 1500),
        ("Porsche", "911 Turbo S", 640, 2.7, 320, 1640),
        ("Tesla", "Model 3", 350, 4.4, 233, 1800),
        ("Tesla", "Model S Plaid", 1020, 2.1, 322, 2160),
        ("Lamborghini", "Huracan", 631, 2.9, 325, 1422),
        ("Ferrari", "488", 661, 3.0, 330, 1475),
        ("Nissan", "GT-R", 565, 2.9, 315, 1740),
    ],
    columns=["brand", "model", "hp", "accel_0_100", "top_speed", "weight"],
)

VEHICLE_CLASSES = [2, 7]  # COCO: car, truck


# ----------------------------------------------------------------------------
# Model loaders (cached so they load once)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO detector...")
def load_yolo(name: str):
    from ultralytics import YOLO

    return YOLO(name)  # downloads weights on first run


@st.cache_resource(show_spinner="Loading CLIP classifier...")
def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, proc, torch


# ----------------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------------
def track_video(path, yolo, stride, conf, progress):
    """Run YOLO tracking; return dict of track_id -> records, plus fps."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    tracks = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            res = yolo.track(
                frame, persist=True, classes=VEHICLE_CLASSES, conf=conf, verbose=False
            )[0]
            if res.boxes is not None and res.boxes.id is not None:
                for box, tid, c in zip(
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.id.cpu().numpy().astype(int),
                    res.boxes.conf.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = box
                    rec = {
                        "frame": idx,
                        "cx": (x1 + x2) / 2,
                        "cy": (y1 + y2) / 2,
                        "w": x2 - x1,
                        "h": y2 - y1,
                        "conf": float(c),
                    }
                    t = tracks.setdefault(tid, {"rows": [], "best": None, "best_score": 0})
                    t["rows"].append(rec)
                    score = c * (x2 - x1) * (y2 - y1)
                    if score > t["best_score"]:
                        t["best_score"] = score
                        t["best"] = (frame.copy(), (int(x1), int(y1), int(x2), int(y2)))
        idx += 1
        if idx % 10 == 0:
            progress.progress(min(idx / total, 1.0))
    cap.release()
    progress.progress(1.0)
    return tracks, fps


def identify_car(crop_bgr, specs):
    """Zero-shot brand/model classification with CLIP. Returns ranked DataFrame."""
    model, proc, torch = load_clip()
    labels = [f"{b} {m}" for b, m in zip(specs["brand"], specs["model"])]
    prompts = [f"a photo of a {l} car" for l in labels]
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    inputs = proc(text=prompts, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image[0]
        probs = logits.softmax(dim=0).numpy()
    out = specs.copy()
    out["probability"] = probs
    return out.sort_values("probability", ascending=False).reset_index(drop=True)


def compute_motion(rows, fps, stride, meters_per_px, smooth):
    """Speed (km/h) and acceleration (m/s^2) from centroid track."""
    df = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
    dt_frame = stride / fps
    df["t"] = (df["frame"] - df["frame"].iloc[0]) / fps
    # Smooth positions to reduce detector jitter
    df["sx"] = df["cx"].rolling(smooth, center=True, min_periods=1).median()
    df["sy"] = df["cy"].rolling(smooth, center=True, min_periods=1).median()
    dist_px = np.hypot(df["sx"].diff(), df["sy"].diff())
    gap = df["frame"].diff() / fps
    v = (dist_px * meters_per_px / gap).replace([np.inf, -np.inf], np.nan)
    df["speed_ms"] = v.rolling(smooth, center=True, min_periods=1).mean().fillna(0)
    df["speed_kmh"] = df["speed_ms"] * 3.6
    df["accel"] = (
        df["speed_ms"].diff() / gap
    ).rolling(smooth, center=True, min_periods=1).mean().fillna(0)
    return df


def simulate_acceleration(hp, weight, target_kmh=100.0, cd=0.32, area=2.2, mu_cap=1.1):
    """Simple longitudinal physics: power-limited + traction cap, drag, rolling resistance."""
    rho, g, crr, eff = 1.225, 9.81, 0.015, 0.85
    P = hp * 745.7 * eff
    dt, t, v = 0.01, 0.0, 0.5
    ts, vs = [0.0], [0.0]
    target = target_kmh / 3.6
    while v < target and t < 60:
        f_power = P / max(v, 0.5)
        f_trac = mu_cap * weight * g
        f_drive = min(f_power, f_trac)
        f_res = 0.5 * rho * cd * area * v**2 + crr * weight * g
        a = (f_drive - f_res) / weight
        if a <= 0:
            break
        v += a * dt
        t += dt
        ts.append(t)
        vs.append(v * 3.6)
    return np.array(ts), np.array(vs)


def similar_cars(specs, ref_idx, n=5):
    feats = ["hp", "accel_0_100", "top_speed", "weight"]
    X = specs[feats].astype(float)
    Xn = (X - X.mean()) / X.std().replace(0, 1)
    d = np.linalg.norm(Xn.values - Xn.values[ref_idx], axis=1)
    out = specs.copy()
    out["distance"] = d
    return out.drop(index=ref_idx).sort_values("distance").head(n)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🚗 Car Video Analyzer")
st.caption("Detect → identify brand/model → measure speed → find & simulate similar cars")

with st.sidebar:
    st.header("Settings")
    yolo_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    stride = st.slider("Process every Nth frame", 1, 10, 2)
    conf = st.slider("Detection confidence", 0.1, 0.9, 0.35)
    smooth = st.slider("Smoothing window (samples)", 1, 15, 5)

    st.subheader("Speed calibration")
    calib = st.radio(
        "Method",
        ["Use real car length (side view)", "Manual meters per pixel"],
    )
    if calib.startswith("Use"):
        car_len = st.number_input("Real car length (m)", 2.5, 6.5, 4.5, 0.1)
        mpp_manual = None
    else:
        mpp_manual = st.number_input("Meters per pixel", 0.0001, 1.0, 0.01, 0.001, format="%.4f")
        car_len = None

    st.subheader("Specs database")
    up_csv = st.file_uploader("Optional CSV (brand,model,hp,accel_0_100,top_speed,weight)", type="csv")

specs = DEFAULT_SPECS.copy()
if up_csv is not None:
    try:
        custom = pd.read_csv(up_csv)
        need = {"brand", "model", "hp", "accel_0_100", "top_speed", "weight"}
        if need.issubset(custom.columns):
            specs = custom
        else:
            st.sidebar.error(f"CSV needs columns: {sorted(need)}")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

video = st.file_uploader("Upload a car video", type=["mp4", "mov", "avi", "mkv"])

if video is None:
    st.info("Upload a short video (5-30 s) where one car is clearly visible. "
            "Side-on shots give the best speed estimates.")
    st.stop()

suffix = os.path.splitext(video.name)[1] or ".mp4"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(video.read())
    video_path = tmp.name

st.video(video_path)

if not st.button("🔍 Analyze", type="primary"):
    st.stop()

# ---- 1. Track ---------------------------------------------------------------
yolo = load_yolo(yolo_name)
st.subheader("1. Detection & tracking")
bar = st.progress(0.0)
tracks, fps = track_video(video_path, yolo, stride, conf, bar)

if not tracks:
    st.error("No cars detected. Try lowering the confidence or using a clearer video.")
    st.stop()

main_id = max(tracks, key=lambda k: len(tracks[k]["rows"]))
main = tracks[main_id]
st.success(f"Tracked {len(tracks)} vehicle(s). Analyzing the longest track (ID {main_id}, "
           f"{len(main['rows'])} samples, {fps:.1f} fps).")

frame, (x1, y1, x2, y2) = main["best"]
annotated = frame.copy()
cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
crop = frame[max(y1, 0):y2, max(x1, 0):x2]

c1, c2 = st.columns([2, 1])
c1.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Best detection", use_container_width=True)
c2.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="Crop used for identification", use_container_width=True)

# ---- 2. Identify ------------------------------------------------------------
st.subheader("2. Brand & model identification")
with st.spinner("Running CLIP..."):
    ranked = identify_car(crop, specs)

top = ranked.iloc[0]
st.markdown(f"### Best match: **{top['brand']} {top['model']}** "
            f"({top['probability']*100:.1f}% confidence)")
st.dataframe(
    ranked.head(5)[["brand", "model", "probability"]].style.format({"probability": "{:.1%}"}),
    use_container_width=True,
)
st.caption("Zero-shot guess limited to the cars in the specs database. "
           "Low confidence? Pick the correct car manually below.")

labels = [f"{b} {m}" for b, m in zip(specs["brand"], specs["model"])]
default_idx = labels.index(f"{top['brand']} {top['model']}")
chosen = st.selectbox("Reference car (override if wrong)", labels, index=default_idx)
ref_idx = labels.index(chosen)
ref = specs.iloc[ref_idx]

# ---- 3. Observed performance --------------------------------------------------
st.subheader("3. Observed performance from the video")
if car_len is not None:
    med_w = float(np.median([r["w"] for r in main["rows"]]))
    mpp = car_len / med_w
else:
    mpp = mpp_manual
st.write(f"Scale used: **{mpp:.4f} m/pixel**")

motion = compute_motion(main["rows"], fps, stride, mpp, smooth)

m1, m2, m3 = st.columns(3)
m1.metric("Max speed", f"{motion['speed_kmh'].max():.0f} km/h")
m2.metric("Average speed", f"{motion['speed_kmh'].mean():.0f} km/h")
m3.metric("Peak acceleration", f"{motion['accel'].max():.1f} m/s²")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].plot(motion["t"], motion["speed_kmh"])
ax[0].set(title="Speed vs time", xlabel="s", ylabel="km/h")
ax[0].grid(alpha=0.3)
ax[1].plot(motion["sx"], motion["sy"])
ax[1].invert_yaxis()
ax[1].set(title="Trajectory (pixels)", xlabel="x", ylabel="y")
ax[1].grid(alpha=0.3)
st.pyplot(fig)
st.caption("Speed estimates assume a roughly side-on camera and a stationary camera. "
           "Perspective, camera motion and zoom can cause large errors.")

# ---- 4. Specs ----------------------------------------------------------------
st.subheader("4. Specs of the reference car")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Power", f"{ref['hp']:.0f} hp")
s2.metric("0-100 km/h", f"{ref['accel_0_100']:.1f} s")
s3.metric("Top speed", f"{ref['top_speed']:.0f} km/h")
s4.metric("Weight", f"{ref['weight']:.0f} kg")
st.caption("Database figures are approximate and vary by model year and trim.")

# ---- 5. Create something like that -------------------------------------------
st.subheader("5. Similar cars & acceleration simulation")
sims = similar_cars(specs, ref_idx, n=5)
st.write("**Most similar cars by performance profile:**")
st.dataframe(
    sims[["brand", "model", "hp", "accel_0_100", "top_speed", "weight"]],
    use_container_width=True,
)

st.write("**Physics simulation (0-100 km/h):**")
fig2, ax2 = plt.subplots(figsize=(9, 4))
t, v = simulate_acceleration(ref["hp"], ref["weight"])
ax2.plot(t, v, label=f"{chosen} (simulated)", linewidth=2.5)
for _, r in sims.head(3).iterrows():
    ts, vs = simulate_acceleration(r["hp"], r["weight"])
    ax2.plot(ts, vs, "--", label=f"{r['brand']} {r['model']}")
ax2.set(xlabel="Time (s)", ylabel="Speed (km/h)", title="Simulated acceleration")
ax2.grid(alpha=0.3)
ax2.legend()
st.pyplot(fig2)
st.caption(f"Simulated 0-100: {t[-1]:.1f} s vs quoted {ref['accel_0_100']:.1f} s "
           "(simple model: no gear shifts, launch control or drivetrain detail).")

# ---- Report download ----------------------------------------------------------
report = f"""CAR ANALYSIS REPORT
===================
Video: {video.name}
Identified as: {chosen} (CLIP confidence for top guess: {top['probability']*100:.1f}%)

Observed (from video)
  Max speed:        {motion['speed_kmh'].max():.0f} km/h
  Average speed:    {motion['speed_kmh'].mean():.0f} km/h
  Peak accel:       {motion['accel'].max():.1f} m/s^2

Database specs (approximate)
  Power:            {ref['hp']:.0f} hp
  0-100 km/h:       {ref['accel_0_100']:.1f} s
  Top speed:        {ref['top_speed']:.0f} km/h
  Weight:           {ref['weight']:.0f} kg

Similar cars
""" + "\n".join(f"  - {r['brand']} {r['model']}" for _, r in sims.iterrows())

st.download_button("⬇️ Download report", report, file_name="car_report.txt")
st.download_button("⬇️ Download motion data (CSV)",
                   motion.to_csv(index=False), file_name="car_motion.csv")
