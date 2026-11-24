"""
Flaticoo – AI podiatry & custom-footwear suite (Streamlit)

Files expected next to this app.py
  pesplanus_cnn.keras          -> X-ray classifier (used as-is)
  material_features.py         -> feature engineering of the regression model (used as-is)
  material_pipeline.joblib     -> saved by step 8 of material_prediction_pipeline.ipynb
        (fallback: flatfoot_shoe_materials.csv -> the same Ridge pipeline is rebuilt with the notebook's code)
Run:  streamlit run app.py
"""
import io, json, os, sys, time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)  # so joblib can import material_features.add_features

KERAS_PATH = os.path.join(APP_DIR, "pesplanus_cnn.keras")
JOBLIB_PATH = os.path.join(APP_DIR, "material_pipeline.joblib")
CSV_NAMES = ["flatfoot_shoe_materials.csv", os.path.join("data", "flatfoot_shoe_materials.csv")]
IMG_SIZE = 227  # model input: 227 x 227 x 1

# ---- column definitions, copied from the notebook ------------------------------------------------
X_COLS = ["foot_length_cm", "ball_girth_cm", "waist_girth_cm", "instep_girth_cm",
          "long_heel_girth_cm", "ankle_circ_cm", "heel_width_cm", "arch_angle_deg", "body_weight_kg"]
PCT_COLS = ["pct_eva_biofoam", "pct_tpu_cradle", "pct_jacquard_mesh", "pct_carbon_stabilizer", "pct_bamboo_sockliner"]
SPEC_COLS = ["medial_eva_shore_c", "lateral_eva_shore_c", "heel_counter_mm", "medial_post_mm", "heel_midsole_mm"]
Y_COLS = PCT_COLS + SPEC_COLS
LIMITS = {"foot_length_cm": (20, 34), "ball_girth_cm": (18, 36), "waist_girth_cm": (17, 35), "instep_girth_cm": (18, 36),
          "long_heel_girth_cm": (25, 45), "ankle_circ_cm": (14, 34), "heel_width_cm": (3, 11),
          "arch_angle_deg": (110, 180), "body_weight_kg": (25, 200)}
# typical (dataset mean) starting values
DEFAULTS = dict(foot_length_cm=26.5, ball_girth_cm=24.6, waist_girth_cm=23.6, instep_girth_cm=24.9,
                long_heel_girth_cm=32.6, ankle_circ_cm=22.4, heel_width_cm=7.0, arch_angle_deg=146.0, body_weight_kg=73.0)
# the sample patient used at the end of the notebook (also the sample shown in the design)
SAMPLE = dict(foot_length_cm=27.8, ball_girth_cm=26.4, waist_girth_cm=25.9, instep_girth_cm=26.8,
              long_heel_girth_cm=34.2, ankle_circ_cm=23.5, heel_width_cm=7.4, arch_angle_deg=168.4, body_weight_kg=82.0)

FIELDS = [  # key, number, title, tag, description
    ("foot_length_cm", 1, "Foot Length", "Pterion · Apex", "Back of the heel (pterion) to the tip of the longest toe, weight-bearing."),
    ("ball_girth_cm", 2, "Ball Girth (Joint Girth)", "Critical", "Circumference around the 1st and 5th metatarsal heads. Primary for width."),
    ("waist_girth_cm", 3, "Waist Girth", "Midfoot contour", "Circumference around the narrowest part of the midfoot, directly behind the ball."),
    ("instep_girth_cm", 4, "Instep Girth", "Navicular zone", "Highest point of the arch (cuneiform / navicular) under the midfoot."),
    ("long_heel_girth_cm", 5, "Long Heel Girth", "Heel-to-instep", "Diagonal from the base of the heel around the instep apex."),
    ("ankle_circ_cm", 6, "Ankle / Top-Line", "Collar line", "Perimeter around the malleoli where the upper collar and heel counter end."),
    ("heel_width_cm", 7, "Heel Width (Calcaneal Base)", "Posting anchor", "Horizontal width across the calcaneal fat pad."),
    ("arch_angle_deg", 8, "Arch Angle (Moreau-Costa-Bartani)", "From lateral X-ray", "Measured on the weight-bearing lateral radiograph. Normal 120–130°, flat foot from 135°."),
    ("body_weight_kg", 9, "Body Weight", "Load input", "Body weight in kilograms – sets how much support the formulation needs."),
]
MATERIALS = {  # column -> (name, colour)
    "pct_eva_biofoam": ("Dual-Density Bio-Foam EVA", "#09485b"),
    "pct_tpu_cradle": ("Thermoplastic Polyurethane (TPU) Rigid Cradle", "#d97706"),
    "pct_jacquard_mesh": ("3D Jacquard Engineered Breathable Mesh", "#306579"),
    "pct_carbon_stabilizer": ("Carbon-Infused Heel Stabilizer Counter", "#1e293b"),
    "pct_bamboo_sockliner": ("Bamboo Charcoal Memory Foam Sockliner", "#94a3b8"),
}

st.set_page_config(page_title="Flaticoo · Kinetic Podiatry", page_icon="🦶", layout="wide",
                   initial_sidebar_state="expanded")

# ====================================================================================================
# Styling (Kinetic Apex design system)
# ====================================================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--pet:#09485b;--pet2:#0d5e75;--amb:#d97706;--sl:#1e293b;--bone:#f1f5f9;--bg:#f8fafc;--bd:#e2e8f0;--mut:#64748b;--red:#ba1a1a;--grn:#15803d}
html,body,[class*="css"],.stApp{font-family:'Plus Jakarta Sans',system-ui,sans-serif;color:var(--sl)}
.stApp{background:var(--bg)}
header[data-testid="stHeader"]{background:rgba(255,255,255,.88);backdrop-filter:blur(12px);border-bottom:1px solid var(--bd)}
#MainMenu,footer{visibility:hidden}
.block-container{padding-top:2.2rem;max-width:1280px}
h1,h2,h3,h4,.sora{font-family:'Sora',sans-serif!important;color:var(--sl);letter-spacing:-.01em}
.lbl{font-family:'Space Grotesk',sans-serif;font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase}
.eyebrow{font-family:'Space Grotesk';font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--amb);margin-bottom:.2rem}
.mut{color:var(--mut);font-size:14px;line-height:22px}
/* sidebar */
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--bd)}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:.8rem}
.brand .logo{width:34px;height:34px;border-radius:8px;background:var(--pet);color:#fff;display:flex;align-items:center;justify-content:center;font-family:Sora;font-weight:800}
.brand b{font-family:Sora;font-size:17px;display:block;line-height:1}
.brand span{font-family:'Space Grotesk';font-size:9px;letter-spacing:.1em;color:var(--mut);text-transform:uppercase}
.pill{display:inline-flex;align-items:center;gap:6px;border-radius:9999px;padding:3px 10px;font-family:'Space Grotesk';font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase}
.pill.pet{background:#e6f1f5;color:var(--pet);border:1px solid #c5dde6}
.pill.amb{background:#fff4e5;color:#904d00;border:1px solid #fed7aa}
.pill.red{background:#ffdad6;color:#93000a;border:1px solid #fca5a5}
.pill.grn{background:#dcfce7;color:#14532d;border:1px solid #86efac}
.pill.gry{background:var(--bone);color:var(--mut);border:1px solid var(--bd)}
.dot{width:7px;height:7px;border-radius:50%;background:currentColor;display:inline-block}
/* buttons */
.stButton>button,.stDownloadButton>button{border-radius:8px;font-family:'Sora';font-weight:600;font-size:14px;padding:.6rem 1.2rem;border:1.5px solid var(--pet);color:var(--pet);background:#fff;transition:all .15s}
.stButton>button:hover,.stDownloadButton>button:hover{background:rgba(9,72,91,.05);border-color:var(--pet);color:var(--pet);box-shadow:0 8px 24px -4px rgba(9,72,91,.08)}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--pet);color:#fff;border-color:var(--pet)}
.stButton>button[kind="primary"]:hover{background:var(--pet2);color:#fff}
.st-key-cta button{background:var(--amb)!important;border-color:var(--amb)!important;color:#fff!important}
.st-key-cta button:hover{background:#b45f05!important}
.stButton>button:disabled{opacity:.45}
/* bordered containers = cards */
div[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--bd)!important;border-radius:12px!important;background:#fff;box-shadow:none}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{box-shadow:0 8px 24px -4px rgba(9,72,91,.08)}
/* inputs */
div[data-baseweb="input"],div[data-baseweb="base-input"]{border-radius:8px!important}
div[data-baseweb="input"]:focus-within{box-shadow:0 0 0 2px var(--pet)}
[data-testid="stFileUploaderDropzone"]{border:2px dashed #cbd5e1;border-radius:12px;background:var(--bone)}
/* hero */
.hero{background:linear-gradient(135deg,#00303f 0%,#09485b 60%,#0d5e75 100%);border-radius:16px;padding:2.2rem 2.4rem;color:#fff;margin-bottom:1.2rem}
.hero h1{color:#fff!important;font-weight:800;font-size:40px;line-height:46px;margin:.4rem 0 .6rem}
.hero p{color:#cfe3ea;max-width:720px;font-size:15px;line-height:24px}
.hero .stats{display:flex;gap:2.2rem;margin-top:1.2rem;flex-wrap:wrap}
.hero .stats div b{font-family:'Space Grotesk';font-size:22px;display:block;color:#fff}
.hero .stats div span{font-size:12px;color:#9bcee5}
.banner{background:var(--pet);color:#fff;border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.2rem;display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;align-items:center}
.banner b{font-family:Sora;font-size:17px}
.banner small{color:#9bcee5;display:block;margin-top:2px}
/* diagnostic result */
.diag{border-radius:12px;padding:1.1rem 1.3rem;margin:.6rem 0 1rem}
.diag.pos{background:#fef2f2;border:1px solid #fecaca}
.diag.neg{background:#f0fdf4;border:1px solid #bbf7d0}
.diag h3{margin:.4rem 0 .2rem;font-size:20px}
.meter{height:10px;border-radius:9999px;background:var(--bone);overflow:hidden;margin:.4rem 0}
.meter>i{display:block;height:100%;border-radius:9999px}
.kpi{background:var(--bone);border-radius:8px;padding:.7rem .9rem}
.kpi small{font-family:'Space Grotesk';font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--mut)}
.kpi b{font-family:'Space Grotesk';font-size:26px;display:block}
.thr{display:flex;justify-content:space-between;font-family:'Space Grotesk';font-size:13px;padding:.35rem 0;border-bottom:1px solid var(--bd)}
.thr:last-child{border:0}
.spec{display:flex;justify-content:space-between;align-items:baseline;padding:.65rem .9rem;border-radius:8px;background:var(--bone);margin-bottom:.5rem}
.spec span{font-size:13px}.spec small{display:block;color:var(--mut);font-size:11px}
.spec b{font-family:'Space Grotesk';font-size:18px;white-space:nowrap}
.mat{margin-bottom:1rem}
.mat .top{display:flex;justify-content:space-between;align-items:baseline;gap:1rem}
.mat .top b{font-family:Sora;font-size:14px}.mat .top em{font-family:'Space Grotesk';font-style:normal;font-weight:700;font-size:17px}
.mat p{font-size:12px;color:var(--mut);margin:.15rem 0 0}
.mat .bar{height:7px;border-radius:9999px;background:var(--bone);margin-top:.35rem}.mat .bar i{display:block;height:100%;border-radius:9999px}
.fnum{display:inline-flex;width:22px;height:22px;border-radius:50%;background:var(--pet);color:#fff;font-family:'Space Grotesk';font-weight:700;font-size:11px;align-items:center;justify-content:center;margin-right:8px}
.ftop{display:flex;justify-content:space-between;align-items:center}
.ftitle{font-family:Sora;font-weight:700;font-size:15px;margin:.35rem 0 .1rem}
.spectrum{position:relative;height:12px;border-radius:9999px;margin:1.4rem 0 .3rem;background:linear-gradient(90deg,#2563eb 0%,#16a34a 18%,#16a34a 30%,#facc15 45%,#f97316 62%,#dc2626 100%)}
.spectrum i{position:absolute;top:-6px;width:4px;height:24px;background:var(--sl);border-radius:2px;transform:translateX(-2px)}
.spectrum i::after{content:attr(data-v);position:absolute;top:-20px;left:50%;transform:translateX(-50%);font-family:'Space Grotesk';font-size:12px;font-weight:700;white-space:nowrap}
.scale{display:flex;justify-content:space-between;font-family:'Space Grotesk';font-size:10px;color:var(--mut)}
.disc{font-size:12px;color:var(--mut);border-top:1px solid var(--bd);padding-top:.8rem;margin-top:1.5rem}
@media(max-width:768px){.hero h1{font-size:28px;line-height:34px}.hero{padding:1.4rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def H(html: str):
    """Render HTML (blank lines / indentation stripped so markdown does not treat it as code)."""
    st.markdown("\n".join(l.strip() for l in html.splitlines() if l.strip()), unsafe_allow_html=True)


# ====================================================================================================
# Model loading (cached)
# ====================================================================================================
@st.cache_resource(show_spinner="Loading X-ray classifier…")
def load_cnn():
    if not os.path.exists(KERAS_PATH):
        return None, f"`pesplanus_cnn.keras` was not found next to app.py ({KERAS_PATH})."
    try:
        import keras
        return keras.models.load_model(KERAS_PATH, compile=False), None
    except Exception as e:  # noqa
        return None, f"Could not load the Keras model: {e}"


@st.cache_resource(show_spinner="Loading material regression model…")
def load_regressor():
    try:
        import joblib
        import material_features  # noqa: F401  (needed to unpickle add_features)
    except Exception as e:  # noqa
        return None, f"Missing dependency: {e}", "none"
    if os.path.exists(JOBLIB_PATH):
        try:
            return joblib.load(JOBLIB_PATH), None, "joblib"
        except Exception as e:  # noqa
            return None, f"Could not load material_pipeline.joblib: {e}", "none"
    csv = next((os.path.join(APP_DIR, n) for n in CSV_NAMES if os.path.exists(os.path.join(APP_DIR, n))), None)
    if csv is None:
        return None, ("`material_pipeline.joblib` was not found. Run the last cell of the notebook "
                      "(step 8) and copy the file next to app.py – or put `flatfoot_shoe_materials.csv` there."), "none"
    # --- identical to notebook steps 1-5 (Ridge, alpha=1, seed 42) -----------------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import Ridge
    from material_features import add_features
    df = pd.read_csv(csv)
    ok = np.ones(len(df), bool)
    for c, (lo, hi) in LIMITS.items():
        ok &= df[c].between(lo, hi)
    df = df[ok].dropna(subset=X_COLS + Y_COLS).reset_index(drop=True)
    X_tr, _, Y_tr, _ = train_test_split(df[X_COLS], df[Y_COLS], test_size=0.2, random_state=42)
    model = TransformedTargetRegressor(
        Pipeline([("f", FunctionTransformer(add_features)), ("s", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        transformer=StandardScaler())
    model.fit(X_tr, Y_tr)
    return {"model": model, "x_cols": X_COLS, "y_cols": Y_COLS}, None, "csv"


def renormalize(P):  # copied from the notebook: the 5 percentages must be >= 0.5 and sum to 100
    P = P.copy()
    p = np.clip(P[:, :5], 0.5, None)
    P[:, :5] = 100 * p / p.sum(1, keepdims=True)
    return P


# ====================================================================================================
# Helpers
# ====================================================================================================
def preprocess_xray(img: Image.Image) -> np.ndarray:
    """-> (1, 227, 227, 1) float32 in [0, 1] (the model rescales to 0-255 internally)."""
    img = ImageOps.exif_transpose(img)
    if img.mode.startswith("I") or img.mode == "F":  # 16-bit grayscale
        a = np.asarray(img, dtype=np.float32)
        a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6)
        img = Image.fromarray((a * 255).astype(np.uint8))
    img = img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0)[None, ..., None]


def arch_class(a: float):
    if a < 120: return "High arch (cavus range)", "gry"
    if a <= 130: return "Normal arch", "grn"
    if a < 135: return "Borderline", "amb"
    if a <= 150: return "Moderate flat foot", "amb"
    return "Severe structural collapse", "red"


def go(page: str):
    st.session_state.page = page


def reset_scan():
    for k in ("scan", "scan_name"):
        st.session_state.pop(k, None)
    st.session_state.uploader_id = st.session_state.get("uploader_id", 0) + 1
    st.session_state.page = "scanner"


def autofill():
    st.session_state.vals = dict(SAMPLE)
    for k in [k for k in st.session_state if str(k).startswith("w_")]:
        del st.session_state[k]


# ---- session state ---------------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("page", "scanner")
ss.setdefault("unit", "cm")
ss.setdefault("vals", dict(DEFAULTS))
ss.setdefault("uploader_id", 0)
ss.setdefault("flip", False)  # False: sigmoid output = P(pes planus)

cnn, cnn_err = load_cnn()
reg, reg_err, reg_src = load_regressor()

# ====================================================================================================
# Sidebar
# ====================================================================================================
with st.sidebar:
    H("""<div class="brand"><div class="logo">F</div><div><b>FLATICOO</b><span>Kinetic Podiatry</span></div></div>""")
    ready = cnn is not None and reg is not None
    H(f"""<span class="pill {'pet' if ready else 'red'}"><i class="dot"></i>{'Inference active' if ready else 'Model files missing'}</span>""")
    st.write("")
    pages = [("scanner", "AI Arch Scanner"), ("formulation", "7-Pt Formulation"), ("prescription", "Prescription Summary")]
    for key, label in pages:
        st.button(label, key=f"nav_{key}", use_container_width=True,
                  type="primary" if ss.page == key else "secondary", on_click=go, args=(key,))
    st.write("")
    with st.expander("Advanced · classifier output"):
        opt = st.radio("Sigmoid output of the CNN is the probability of…",
                       ["Pes planus (flat foot)", "Normal arch"], index=1 if ss.flip else 0,
                       help="The .keras file ends in a single sigmoid unit. Pick which class the value 1 stands for in your training labels.")
        ss.flip = opt.startswith("Normal")
        ss.threshold = st.slider("Decision threshold", 0.05, 0.95, ss.get("threshold", 0.5), 0.05)
    H("""<div class="lbl" style="color:#64748b;margin-top:.8rem">Privacy</div>
         <div class="mut" style="font-size:12px">Uploaded images are held in this session's memory only and are never written to disk by the app.</div>""")
    st.write("")
    st.button("↻  Re-analyze scan", on_click=reset_scan, use_container_width=True, type="primary")

for msg in (cnn_err, reg_err):
    if msg:
        st.error(msg)

# ====================================================================================================
# PAGE 1 – AI Arch Scanner (Keras classifier)
# ====================================================================================================
def page_scanner():
    H("""<div class="hero">
        <span class="pill amb"><i class="dot"></i>Flaticoo AI podiatry suite</span>
        <h1>Engineered for flat foot recovery &amp; athletic performance</h1>
        <p>Upload a weight-bearing lateral foot X-ray. Our CNN screens it for pes planus, then the 7-point formulation
        turns your measurements into a custom shoe construction.</p>
        <div class="stats"><div><b>227 × 227</b><span>Grayscale model input</span></div>
        <div><b>EfficientNetB0</b><span>Backbone + custom head</span></div>
        <div><b>Session only</b><span>No image stored</span></div></div></div>""")
    H("""<div class="eyebrow">AI arch diagnostic</div><h2 style="margin:0">Weight-bearing lateral X-ray screening</h2>""")
    st.write("")
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        with st.container(border=True):
            H("""<div class="ftitle" style="font-size:17px">Load patient radiograph</div>
                 <div class="mut">Weight-bearing lateral view · PNG, JPG, BMP or TIFF (max 45 MB)</div>""")
            f = st.file_uploader("X-ray", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                                 label_visibility="collapsed", key=f"up_{ss.uploader_id}")
            if f is not None:
                try:
                    img = Image.open(f)
                    img.load()
                    st.image(img, caption=f.name, use_container_width=True, clamp=True)
                except Exception as e:  # noqa
                    st.error(f"Could not read this image: {e}")
                    f = None
            st.write("")
            with st.container(key="cta"):
                run = st.button("Execute arch classification", use_container_width=True,
                                disabled=(f is None or cnn is None))
            if f is None:
                st.caption("Upload an X-ray to enable the analysis.")
        with st.container(border=True):
            H("""<div class="lbl" style="color:#64748b;margin-bottom:.4rem">Clinical angle reference thresholds</div>
                <div class="thr"><span>Normal Moreau arch angle</span><b>120° – 130°</b></div>
                <div class="thr"><span>Moderate flat foot (pes planus)</span><b>135° – 150°</b></div>
                <div class="thr"><span>Severe structural collapse</span><b>&gt; 150°</b></div>
                <div class="thr"><span>Optimal calcaneal pitch</span><b>18° – 25°</b></div>""")

    if f is not None and run:
        with st.spinner("Running inference…"):
            t0 = time.perf_counter()
            x = preprocess_xray(img)
            p_raw = float(np.asarray(cnn.predict(x, verbose=0)).ravel()[0])
            ms = (time.perf_counter() - t0) * 1000
        ss.scan = {"p_raw": p_raw, "ms": ms, "file": f.name, "time": datetime.now().strftime("%Y-%m-%d %H:%M")}

    with right:
        with st.container(border=True):
            H("""<div class="lbl" style="color:#64748b">Inference diagnostic output</div>""")
            sc = ss.get("scan")
            if not sc:
                H("""<div class="ftitle" style="font-size:20px">Awaiting radiograph</div>
                     <div class="mut">The classification result, model probability and next steps will appear here.</div>""")
            else:
                p_flat = (1 - sc["p_raw"]) if ss.flip else sc["p_raw"]
                positive = p_flat >= ss.threshold
                conf = p_flat if positive else 1 - p_flat
                cls = "pos" if positive else "neg"
                title = "Pes planus (flat foot) detected" if positive else "No pes planus detected"
                body = ("The model finds features consistent with a collapsed medial longitudinal arch. "
                        "Continue to the 7-point formulation to size and specify a supportive shoe."
                        if positive else
                        "The model does not find features of a flat foot in this radiograph. "
                        "You can still continue to the formulation with your own measurements.")
                H(f"""<div class="diag {cls}"><span class="pill {'red' if positive else 'grn'}">{'Positive case' if positive else 'Negative case'}</span>
                    <h3>{title}</h3><div class="mut">{body}</div></div>""")
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"""<div class="kpi"><small>P(pes planus)</small><b>{p_flat*100:.1f}%</b></div>""", unsafe_allow_html=True)
                c2.markdown(f"""<div class="kpi"><small>Model confidence</small><b>{conf*100:.1f}%</b></div>""", unsafe_allow_html=True)
                c3.markdown(f"""<div class="kpi"><small>Latency</small><b>{sc['ms']:.0f} ms</b></div>""", unsafe_allow_html=True)
                col = "#ba1a1a" if positive else "#15803d"
                H(f"""<div class="lbl" style="margin-top:1rem;color:#64748b">Probability of pes planus (threshold {ss.threshold:.2f})</div>
                    <div class="meter"><i style="width:{p_flat*100:.1f}%;background:{col}"></i></div>""")
                st.caption("The percentage is the raw model output, not a calibrated clinical probability. "
                           "This screening tool supports – it does not replace – a clinician's assessment.")
                st.button("Proceed to biometric anthropometry & custom last sizing →", type="primary",
                          on_click=go, args=("formulation",), use_container_width=True)


# ====================================================================================================
# PAGE 2 – 7-Point formulation (regression)
# ====================================================================================================
def input_card(spec):
    key, n, title, tag, desc = spec
    unit = ss.unit
    is_len = key not in ("arch_angle_deg", "body_weight_kg")
    factor = (1 / 2.54 if unit == "inches" else 1.0) if is_len else 1.0
    lo, hi = LIMITS[key]
    suffix = {"arch_angle_deg": "°", "body_weight_kg": "kg"}.get(key, unit)
    decimals = 2 if (is_len and unit == "inches") else 1
    shown = round(ss.vals[key] * factor, decimals)
    with st.container(border=True):
        H(f"""<div class="ftop"><span><span class="fnum">{n}</span><span class="lbl" style="color:#64748b">{tag}</span></span></div>
              <div class="ftitle">{title}</div><div class="mut" style="font-size:12px;line-height:18px;min-height:36px">{desc}</div>""")
        step = 0.05 if decimals == 2 else 0.1
        v = st.number_input(title, min_value=float(np.floor(lo * factor * 100) / 100), max_value=float(np.ceil(hi * factor * 100) / 100),
                            value=float(shown), step=step, format=f"%.{decimals}f",
                            key=f"w_{key}_{unit if is_len else 'x'}", label_visibility="collapsed")
        st.caption(suffix)
        if abs(v - shown) > 1e-9:
            ss.vals[key] = v / factor


def predict_materials(vals: dict):
    bundle = reg
    row = pd.DataFrame([vals])[bundle["x_cols"]]
    t0 = time.perf_counter()
    out = renormalize(bundle["model"].predict(row))[0]
    return {c: float(v) for c, v in zip(bundle["y_cols"], out)}, (time.perf_counter() - t0) * 1000


def page_formulation():
    sc = ss.get("scan")
    if sc:
        p_flat = (1 - sc["p_raw"]) if ss.flip else sc["p_raw"]
        pos = p_flat >= ss.threshold
        head = "Diagnostic match: pes planus detected" if pos else "Screening result: no pes planus detected"
        sub = f"CNN probability {p_flat*100:.1f}% · scan {sc['file']}"
    else:
        head, sub = "No X-ray analysed yet", "You can still run the formulation with manual measurements."
    H(f"""<div class="banner"><div><span class="lbl" style="color:#fdba74">Step 2 · Precision biometric sizing &amp; material formulation</span>
        <b style="display:block;margin-top:4px">{head}</b><small>{sub}</small></div>
        <span class="pill amb"><i class="dot"></i>Ridge regression · 9 inputs → 10 outputs</span></div>""")
    H("""<div class="eyebrow">Sub-millimeter last anthropometry</div>
         <h2 style="margin:0">Personalized biometric last &amp; anatomical formulation</h2>
         <p class="mut" style="max-width:760px;margin-top:.4rem">Enter the 7 caliper measurements plus the arch angle and body weight. The model returns the
         material mix and construction specification for your custom shoe.</p>""")

    t1, t2, _ = st.columns([1.1, 1.3, 3])
    with t1:
        u = st.radio("Units", ["cm", "inches"], horizontal=True, index=0 if ss.unit == "cm" else 1, label_visibility="collapsed")
        ss.unit = u
    with t2:
        st.button("Autofill sample biometrics", on_click=autofill)

    H("""<div class="lbl" style="color:#64748b;margin:.8rem 0 .4rem">7-point caliper anthropometric form + clinical inputs</div>""")
    for i in range(0, len(FIELDS), 3):
        cols = st.columns(3, gap="medium")
        for c, spec in zip(cols, FIELDS[i:i + 3]):
            with c:
                input_card(spec)

    v = ss.vals
    a = v["arch_angle_deg"]
    label, tone = arch_class(a)
    pos = min(max((a - 110) / 70 * 100, 0), 100)
    with st.container(border=True):
        H(f"""<div class="ftop"><span class="lbl" style="color:#64748b">Arch angle on the continuous spectrum</span>
              <span class="pill {tone}">{label}</span></div>
              <div class="spectrum"><i style="left:{pos:.1f}%" data-v="{a:.1f}°"></i></div>
              <div class="scale"><span>110° High</span><span>120–130° Normal</span><span>145° Borderline</span><span>160°+ Flat</span><span>180°</span></div>""")

    st.write("")
    with st.container(key="cta"):
        run = st.button("Execute ML synthesis", use_container_width=True, disabled=reg is None)

    if run:
        bad = [t for k, t, *_ in FIELDS if not (LIMITS[k][0] <= v[k] <= LIMITS[k][1])]
        if bad:
            st.error("Outside the supported range: " + ", ".join(bad))
        else:
            try:
                out, ms = predict_materials(v)
                ss.result = {"inputs": dict(v), "outputs": out, "ms": ms, "time": datetime.now().strftime("%Y-%m-%d %H:%M")}
            except Exception as e:  # noqa
                st.error(f"Prediction failed: {e}")

    res = ss.get("result")
    if res:
        render_results(res)
        st.write("")
        st.button("Continue to prescription summary →", type="primary", on_click=go, args=("prescription",))


def render_results(res):
    o, i = res["outputs"], res["inputs"]
    stale = any(abs(res["inputs"][k] - ss.vals[k]) > 1e-9 for k in X_COLS)
    st.write("")
    H(f"""<div class="eyebrow">Machine learning dual biometric synthesis</div>
         <h2 style="margin:0">Your custom formulation {'<span class="pill amb" style="vertical-align:middle">inputs changed – re-run</span>' if stale else ''}</h2>""")
    st.write("")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with st.container(border=True):
            H("""<div class="lbl" style="color:#64748b">Output stream 01</div><div class="ftitle" style="font-size:18px">Sizing &amp; construction geometry</div>""")
            H(f"""<div class="spec" style="background:#fff4e5"><span><small>Foot length (Mondo point)</small>Recommended size basis</span><b>{i['foot_length_cm']*10:.0f} mm</b></div>
                <div class="spec"><span>Medial EVA hardness<small>Shore C, arch side</small></span><b>{o['medial_eva_shore_c']:.0f} C</b></div>
                <div class="spec"><span>Lateral EVA hardness<small>Shore C, outer side</small></span><b>{o['lateral_eva_shore_c']:.0f} C</b></div>
                <div class="spec"><span>Medial post height<small>Anti-pronation wedge</small></span><b>{o['medial_post_mm']:.1f} mm</b></div>
                <div class="spec"><span>Heel counter thickness<small>Calcaneal cup</small></span><b>{o['heel_counter_mm']:.2f} mm</b></div>
                <div class="spec"><span>Heel midsole height<small>Cushioning stack at heel</small></span><b>{o['heel_midsole_mm']:.1f} mm</b></div>""")
            st.caption(f"Regressor latency: {res['ms']:.0f} ms")
    with right:
        with st.container(border=True):
            H("""<div class="lbl" style="color:#64748b">Output stream 02</div><div class="ftitle" style="font-size:18px">Prescription footwear material formulation</div>
                 <div class="mut" style="margin-bottom:1rem">Composite distribution formulated to resist excessive pronatory rotational torque. Shares sum to 100 %.</div>""")
            desc = {
                "pct_eva_biofoam": f"Medial shore {o['medial_eva_shore_c']:.0f}C / lateral shore {o['lateral_eva_shore_c']:.0f}C. Dampens navicular collapse while keeping a responsive toe-off.",
                "pct_tpu_cradle": "Prevents midfoot twisting under dynamic load and limits navicular sag through stance phase.",
                "pct_jacquard_mesh": f"Zonal containment with flexible expansion over the instep (girth {i['instep_girth_cm']:.1f} cm).",
                "pct_carbon_stabilizer": "Rigid calcaneal cup that resists outward valgus drift of the heel.",
                "pct_bamboo_sockliner": "Anti-microbial, shear-reducing sockliner cushioning the metatarsal heads.",
            }
            for c in sorted(PCT_COLS, key=lambda c: -o[c]):
                name, colr = MATERIALS[c]
                H(f"""<div class="mat"><div class="top"><b>{name}</b><em>{o[c]:.1f}%</em></div>
                    <div class="bar"><i style="width:{o[c]:.1f}%;background:{colr}"></i></div><p>{desc[c]}</p></div>""")
    if i["arch_angle_deg"] < 135:
        st.info("The entered arch angle is below 135°, i.e. in the normal range. The regression is trained on flat-foot cases, "
                "so treat the output as a reference rather than a prescription.")


# ====================================================================================================
# PAGE 3 – Prescription summary
# ====================================================================================================
def page_prescription():
    H("""<div class="eyebrow">Validated custom device specification</div><h2 style="margin:0">Prescription summary</h2>""")
    st.write("")
    sc, res = ss.get("scan"), ss.get("result")
    if not res:
        st.info("Run the 7-point formulation first to generate a prescription.")
        st.button("Go to 7-Pt Formulation", type="primary", on_click=go, args=("formulation",))
        return
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        with st.container(border=True):
            H("""<div class="lbl" style="color:#64748b">Diagnosis</div>""")
            if sc:
                p_flat = (1 - sc["p_raw"]) if ss.flip else sc["p_raw"]
                pos = p_flat >= ss.threshold
                H(f"""<div class="ftitle" style="font-size:18px">{'Pes planus detected' if pos else 'No pes planus detected'}</div>
                      <span class="pill {'red' if pos else 'grn'}">P(pes planus) {p_flat*100:.1f}%</span>""")
            else:
                H("""<div class="ftitle" style="font-size:18px">No X-ray analysed</div><span class="pill gry">Manual measurements</span>""")
            lab, tone = arch_class(res["inputs"]["arch_angle_deg"])
            H(f"""<div class="lbl" style="color:#64748b;margin-top:1rem">Arch angle</div>
                  <div class="kpi" style="margin-top:.3rem"><b>{res['inputs']['arch_angle_deg']:.1f}°</b></div>
                  <span class="pill {tone}" style="margin-top:.5rem">{lab}</span>""")
        with st.container(border=True):
            H("""<div class="lbl" style="color:#64748b;margin-bottom:.4rem">Measurements used (cm / kg / °)</div>""")
            st.dataframe(pd.DataFrame({"input": [t for _, _, t, *_ in FIELDS], "value": [res["inputs"][k] for k, *_ in FIELDS]}),
                         hide_index=True, use_container_width=True)
    with c2:
        with st.container(border=True):
            H("""<div class="lbl" style="color:#64748b;margin-bottom:.4rem">Material mix</div>""")
            for c in PCT_COLS:
                name, colr = MATERIALS[c]
                v = res["outputs"][c]
                H(f"""<div class="mat" style="margin-bottom:.6rem"><div class="top"><b style="font-size:13px">{name}</b><em style="font-size:15px">{v:.1f}%</em></div>
                    <div class="bar"><i style="width:{v:.1f}%;background:{colr}"></i></div></div>""")
            specs = pd.DataFrame({"specification": ["Medial EVA shore C", "Lateral EVA shore C", "Heel counter (mm)", "Medial post (mm)", "Heel midsole (mm)"],
                                  "value": [round(res["outputs"][c], 2) for c in SPEC_COLS]})
            st.dataframe(specs, hide_index=True, use_container_width=True)
    payload = {
        "generated": res["time"],
        "diagnosis": None if not sc else {"file": sc["file"], "model_output": sc["p_raw"], "p_pes_planus": (1 - sc["p_raw"]) if ss.flip else sc["p_raw"],
                                          "threshold": ss.threshold},
        "inputs": res["inputs"], "outputs": res["outputs"],
    }
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download spec (JSON)", json.dumps(payload, indent=2), "flaticoo_spec.json", "application/json", type="primary")
    flat = {**res["inputs"], **res["outputs"]}
    d2.download_button("Download spec (CSV)", pd.DataFrame([flat]).to_csv(index=False), "flaticoo_spec.csv", "text/csv")
    H("""<div class="disc"><b>Notes.</b> The classifier is an EfficientNetB0-based screening model and the material regressor (Ridge) was trained on a
         synthetic, rule-based dataset (notebook mean R² ≈ 0.86 on the held-out test set; weakest on bamboo-sockliner share and lateral EVA hardness).
         Outputs are decision support only and are not a medical device or a clinical prescription.</div>""")


{"scanner": page_scanner, "formulation": page_formulation, "prescription": page_prescription}[ss.page]()
