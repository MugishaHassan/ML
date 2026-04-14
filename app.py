import streamlit as st
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0d0d; color: #f0ece4; }
#MainMenu, footer, header { visibility: hidden; }

.hero {
    text-align: center;
    padding: 3rem 0 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-radius: 100px;
    padding: 0.3rem 1rem;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a0a0a0;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 6vw, 4rem);
    font-weight: 800;
    line-height: 1.05;
    color: #f0ece4;
    margin: 0 0 0.6rem;
}
.hero-title span { color: #ff6b35; }
.hero-sub {
    font-size: 1rem;
    color: #6e6e6e;
    font-weight: 300;
    max-width: 400px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
}

[data-testid="stFileUploader"] {
    background: #111111 !important;
    border: 1.5px dashed #2a2a2a !important;
    border-radius: 20px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] section { border: none !important; background: transparent !important; }

.img-card {
    background: #111111;
    border: 1px solid #1e1e1e;
    border-radius: 20px;
    overflow: hidden;
    margin: 1.5rem 0;
}
.img-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #444;
    padding: 0.8rem 1.2rem 0;
}

.result-card {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    animation: fadeUp 0.4s ease;
}
.result-card.cat { background: linear-gradient(135deg, #12121a, #1a1228); border: 1px solid #2d1f4a; }
.result-card.dog { background: linear-gradient(135deg, #121a12, #1a2812); border: 1px solid #1f4a1f; }
.result-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-label { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; margin: 0.2rem 0; }
.result-label.cat { color: #b388ff; }
.result-label.dog { color: #69f0ae; }
.result-conf { font-size: 0.85rem; color: #555; margin-top: 0.3rem; }

.conf-bar-track {
    background: #1a1a1a;
    border-radius: 100px;
    height: 6px;
    width: 60%;
    margin: 0.8rem auto 0;
    overflow: hidden;
}
.conf-bar-fill { height: 100%; border-radius: 100px; }
.conf-bar-fill.cat { background: #b388ff; }
.conf-bar-fill.dog { background: #69f0ae; }

.footer { text-align: center; padding: 2rem 0 1rem; font-size: 0.75rem; color: #333; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    return load_model("model.h5")

model = load_classifier()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🐾 Deep Learning Classifier</div>
    <h1 class="hero-title">Cat <span>vs</span> Dog</h1>
    <p class="hero-sub">Drop any photo — the AI will tell you what's in it in under a second.</p>
</div>
""", unsafe_allow_html=True)

# ── Uploader ──────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ── Predict ───────────────────────────────────────────────────────────────────
if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="img-card"><div class="img-label">Uploaded image</div>', unsafe_allow_html=True)
    st.image(pil_img, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess — same as notebook
    img_resized = pil_img.resize((200, 200))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analysing..."):
        result = float(model.predict(img_array)[0][0])

    is_dog     = result >= 0.5
    label      = "Dog" if is_dog else "Cat"
    emoji      = "🐶" if is_dog else "🐱"
    css_class  = "dog" if is_dog else "cat"
    confidence = result if is_dog else (1 - result)
    conf_pct   = round(confidence * 100, 1)
    bar_width  = round(confidence * 100)

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label {css_class}">{label}</div>
        <div class="result-conf">{conf_pct}% confidence</div>
        <div class="conf-bar-track">
            <div class="conf-bar-fill {css_class}" style="width:{bar_width}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown('<div style="text-align:center;padding:3rem 0;color:#2e2e2e;font-size:3rem;">🐾</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">Built with TensorFlow · Streamlit · ❤️</div>', unsafe_allow_html=True)
