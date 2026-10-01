"""
Diabetic Retinopathy Stage Screening — Streamlit web app

Loads the EfficientNetB3 model trained in the Kaggle notebook and, for an uploaded fundus photograph:
  1. applies the SAME preprocessing used in training (retina crop -> square pad -> resize -> enhancement),
  2. predicts one of 5 DR stages with class probabilities,
  3. turns the stage into a screening decision (any DR / referable DR) with a suggested next step,
  4. explains the decision with a Grad-CAM heatmap.

Lightweight deployment: full TensorFlow needs ~1.1 GB RAM, more than free hosting allows. The notebook therefore
exports the CNN backbone as TensorFlow Lite (run with the small LiteRT runtime) and the classification head as
NumPy weights. Because the head begins with Global Average Pooling, Grad-CAM's gradient can be computed exactly
in NumPy, so no TensorFlow is needed at run time.

Files expected next to this script: dr_features.tflite, dr_head.npz, model_meta.json, (optional) examples/*.png
Run locally:  streamlit run streamlit_app.py
"""
import json
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter   # LiteRT (lightweight TFLite runtime)
except ImportError:                                       # fallback if full TensorFlow is installed
    from tensorflow.lite import Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))
PRETTY = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
STAGE_INFO = [
    ("No signs of diabetic retinopathy detected.",
     "Routine re-screening in 12 months."),
    ("Microaneurysms only (small bulges in retinal blood vessels).",
     "Re-screen in 6–12 months; manage blood sugar and blood pressure."),
    ("More than microaneurysms: haemorrhages, hard exudates or cotton-wool spots.",
     "Refer to an ophthalmologist within 3–6 months."),
    ("Extensive haemorrhages, venous beading or intraretinal microvascular abnormalities.",
     "Urgent referral to an ophthalmologist (within weeks)."),
    ("Growth of abnormal new blood vessels: high risk of severe vision loss.",
     "Urgent referral to an ophthalmologist for treatment (e.g. laser / anti-VEGF)."),
]

st.set_page_config(page_title="DR Stage Screening", page_icon="👁️", layout="wide")


# --------------------------------------------------------------------------------------
# Model + metadata (loaded once per server, cached across users and reruns)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open(os.path.join(HERE, "model_meta.json")) as f:
        meta = json.load(f)
    interp = Interpreter(model_path=os.path.join(HERE, "dr_features.tflite"))
    interp.allocate_tensors()
    head = dict(np.load(os.path.join(HERE, "dr_head.npz")))
    return interp, head, meta


INTERP, HEAD, META = load_model()
IMG_SIZE = META["img_size"]
_IN, _OUT = INTERP.get_input_details()[0]["index"], INTERP.get_output_details()[0]["index"]


# --------------------------------------------------------------------------------------
# Preprocessing — identical to the training notebook
# --------------------------------------------------------------------------------------
def crop_to_retina(img, tol=7):
    """Cut away the dark border around the retina (threshold segmentation)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() < 0.05 * mask.size:
        return img
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def pad_to_square(img):
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left, cv2.BORDER_CONSTANT, value=0)


def circular_mask(size=IMG_SIZE, scale=0.97):
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), int(size / 2 * scale), 1, -1)
    return m


MASK = circular_mask()


def enhance_clahe(img):
    img = cv2.medianBlur(img, 3)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return img * MASK[..., None]


def enhance_ben_graham(img):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=img.shape[0] / 30)
    img = cv2.addWeighted(img, 4, blur, -4, 128)
    return img * MASK[..., None]


ENHANCE = {"raw": lambda im: im, "clahe": enhance_clahe, "bengraham": enhance_ben_graham}[META["preprocessing"]]


def base_preprocess(rgb):
    """Crop retina -> pad to square -> resize. Returns RGB uint8."""
    img = pad_to_square(crop_to_retina(rgb))
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def feature_maps(img_uint8):
    """Backbone (TFLite): RGB 0-255 image -> last conv feature maps (H x W x C).
    EfficientNet contains its own rescaling layer, so it takes raw 0-255 pixels."""
    INTERP.set_tensor(_IN, img_uint8[None].astype(np.float32))
    INTERP.invoke()
    return INTERP.get_tensor(_OUT)[0]


# --------------------------------------------------------------------------------------
# Classification head + Grad-CAM in NumPy
#   GAP -> BatchNorm -> Dense(256, ReLU) -> Dense(5) -> softmax   (dropout is inactive at inference)
# --------------------------------------------------------------------------------------
def head_forward(fmap):
    g = fmap.mean(axis=(0, 1))                                           # Global Average Pooling
    bn_scale = HEAD["bn_gamma"] / np.sqrt(HEAD["bn_var"] + HEAD["bn_eps"])
    z = (g - HEAD["bn_mean"]) * bn_scale + HEAD["bn_beta"]               # BatchNorm (inference mode)
    pre = z @ HEAD["w1"] + HEAD["b1"]
    a = np.maximum(pre, 0)                                               # Dense + ReLU
    logits = a @ HEAD["w2"] + HEAD["b2"]
    p = np.exp(logits - logits.max())
    return p / p.sum(), (bn_scale, pre)


def gradcam(fmap, probs, cache, class_idx):
    """Grad-CAM: weight each feature map by d(class score)/d(feature map), sum, ReLU.
    The gradient is back-propagated analytically through softmax -> Dense -> ReLU -> Dense -> BN -> GAP."""
    bn_scale, pre = cache
    d_logits = probs[class_idx] * (np.eye(len(probs))[class_idx] - probs)   # softmax derivative
    d_a = HEAD["w2"] @ d_logits
    d_pre = d_a * (pre > 0)                                                  # ReLU derivative
    d_z = HEAD["w1"] @ d_pre
    weights = d_z * bn_scale / (fmap.shape[0] * fmap.shape[1])               # BN + GAP derivatives
    cam = np.maximum((fmap * weights).sum(axis=-1), 0)
    cam = cv2.resize(cam.astype(np.float32), (IMG_SIZE, IMG_SIZE))
    return cam / (cam.max() + 1e-8) * MASK


def overlay(img, cam, alpha=0.45):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1]
    return np.uint8(img * (1 - alpha) + heat * alpha)


def analyse(rgb):
    base = base_preprocess(rgb)
    enhanced = ENHANCE(base).astype(np.uint8)
    fmap = feature_maps(enhanced)
    probs, cache = head_forward(fmap)
    stage = int(np.argmax(probs))
    heat = overlay(base, gradcam(fmap, probs, cache, stage))
    if META.get("use_tta"):   # average with flipped versions (stage/heatmap from the original view)
        probs = (probs + head_forward(feature_maps(enhanced[:, ::-1].copy()))[0]
                 + head_forward(feature_maps(enhanced[::-1].copy()))[0]) / 3
        stage = int(np.argmax(probs))
    return probs, stage, base, enhanced, heat


# --------------------------------------------------------------------------------------
# Sidebar: model card
# --------------------------------------------------------------------------------------
m = META["test_metrics"]
with st.sidebar:
    st.header("About the model")
    st.markdown(
        f"**Backbone:** {META['backbone']} (ImageNet transfer learning, two-phase fine-tuning)  \n"
        f"**Preprocessing:** {META['preprocessing']}  \n"
        f"**Input size:** {IMG_SIZE}×{IMG_SIZE}  \n"
        f"**Data:** APTOS 2019 (Kaggle) — {META['n_train']} train / {META['n_val']} val / {META['n_test']} test"
    )
    st.subheader("Held-out test performance")
    st.markdown(
        f"| Metric | Value |\n|---|---|\n"
        f"| Accuracy | {m['accuracy']:.1%} |\n"
        f"| Quadratic Weighted Kappa | {m['qwk']:.3f} |\n"
        f"| Macro AUC | {m['auc_macro']:.3f} |\n"
        f"| Any-DR sensitivity | {m['anyDR_sens']:.1%} |\n"
        f"| Any-DR specificity | {m['anyDR_spec']:.1%} |\n"
        f"| Referable-DR sensitivity | {m['referable_sens']:.1%} |\n"
        f"| Referable-DR specificity | {m['referable_spec']:.1%} |"
    )
    st.caption("Grad-CAM highlights the retinal regions that most influenced the prediction (red = strongest).")

# --------------------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------------------
st.title("👁️ Diabetic Retinopathy Stage Screening")
st.write("Upload a colour fundus photograph to estimate the diabetic retinopathy stage (0–4) "
         "and see which regions of the retina the model used.")
st.warning("⚕️ Research prototype for a university computer-vision coursework. **Not a medical device** — "
           "do not use for diagnosis. Always consult a qualified eye-care professional.")

ex_dir = os.path.join(HERE, "examples")
if not os.path.isdir(ex_dir):          # example images may also sit next to this script (stage0_*.png ...)
    ex_dir = HERE
examples = sorted(f for f in os.listdir(ex_dir)
                  if f.lower().startswith("stage") and f.lower().endswith((".png", ".jpg", ".jpeg")))

left, right = st.columns(2)
with left:
    uploaded = st.file_uploader("Fundus photograph", type=["png", "jpg", "jpeg", "tif", "tiff"])
    choice = None
    if examples:
        choice = st.selectbox("…or try an example from the test set", ["—"] + examples)

image = None
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
elif choice and choice != "—":
    image = Image.open(os.path.join(ex_dir, choice)).convert("RGB")

if image is None:
    with right:
        st.info("Upload an image or pick an example to begin.")
    st.stop()

with st.spinner("Analysing…"):
    probs, stage, base, enhanced, heat = analyse(np.asarray(image))

conf = float(probs[stage])
p_any, p_ref = float(probs[1:].sum()), float(probs[2:].sum())
desc, action = STAGE_INFO[stage]

with left:
    st.image(image, caption="Uploaded image", width="stretch")

with right:
    st.subheader(f"Stage {stage}: {PRETTY[stage]}")
    st.metric("Confidence", f"{conf:.0%}")
    if stage >= 2:
        st.error("🔴 **Referable diabetic retinopathy**")
    elif stage == 1:
        st.warning("🟡 **Diabetic retinopathy present (non-referable)**")
    else:
        st.success("🟢 **No diabetic retinopathy detected**")
    c1, c2 = st.columns(2)
    c1.metric("Any DR (stage ≥ 1)", f"{p_any:.0%}")
    c2.metric("Referable DR (stage ≥ 2)", f"{p_ref:.0%}")
    st.markdown(f"**What this stage means:** {desc}  \n**Suggested next step:** {action}")
    if conf < 0.5:
        st.caption("⚠️ Low confidence: the model is unsure between stages. A clinician should review this image.")

    st.markdown("**Stage probabilities**")
    for i, p in enumerate(probs):
        st.progress(float(p), text=f"{i}: {PRETTY[i]} — {p:.1%}")

st.divider()
c1, c2, c3 = st.columns(3)
c1.image(base, caption="1. Retina cropped + resized", width="stretch")
c2.image(enhanced, caption=f"2. What the model sees ({META['preprocessing']} preprocessing)", width="stretch")
c3.image(heat, caption="3. Grad-CAM explanation", width="stretch")
