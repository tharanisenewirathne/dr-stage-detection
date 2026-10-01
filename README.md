# Diabetic Retinopathy Stage Detection with CNN Transfer Learning

Five-stage diabetic retinopathy (DR) grading from colour fundus photographs, using image preprocessing, data augmentation, class balancing and ImageNet-pretrained CNNs. Includes Grad-CAM explainability and a deployed web app.

- **Notebook:** `dr_stage_detection.ipynb` (Kaggle, GPU T4)
- **Web app:** `app/` (Streamlit, hosted free on Streamlit Community Cloud) — [_https://dr-stage-screening.streamlit.app/_]
- **Demo video:** _add your video link here_

## Dataset

[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) — 3,662 labelled fundus images graded 0–4 by clinicians. Images are not included in this repository (competition licence); add the competition as an input in Kaggle.

| Stage | 0 No DR | 1 Mild | 2 Moderate | 3 Severe | 4 Proliferative |
|---|---|---|---|---|---|
| Images | 1,805 | 370 | 999 | 193 | 295 |

Data cleaning: 130 duplicate photo groups were found with a 256-bit difference hash; 36 had conflicting grades and were removed, leaving **3,491** images. Stratified split: **2,443 train / 349 validation / 699 test**.

## Pipeline

```
Fundus image → retina crop (threshold) → pad to square → resize 224×224
            → enhancement (Ben Graham / CLAHE + unsharp, chosen by ablation)
            → augmentation (flip, rotate, zoom, brightness, contrast; train only)
            → EfficientNetB3 (ImageNet) + GAP → BN → Dropout → Dense(256) → Dropout → Softmax(5)
            → two-phase training (frozen backbone, then fine-tune top 20% with BatchNorm frozen)
            → evaluation + Grad-CAM → lightweight export (TFLite + NumPy head) → Streamlit web app
```

## Experiments (selection on validation set)

| Experiment | Options | Selected |
|---|---|---|
| Preprocessing ablation (frozen EfficientNetB3) | raw 0.820 · CLAHE 0.824 · Ben Graham 0.831 (val QWK) | Ben Graham |
| Head learning rate | 1e-3 0.847 · 3e-4 0.818 · 1e-4 0.796 (val QWK) | 1e-3 |
| Backbone (full two-phase training) | EfficientNetB3 0.857 · ResNet50 0.825 · MobileNetV2 0.783 (val QWK) | EfficientNetB3 |
| Test-time augmentation | val QWK 0.858 without vs 0.854 with | not used |

## Results on the held-out test set (EfficientNetB3, 699 images)

| Metric | Value |
|---|---|
| Accuracy | 75.5% |
| Quadratic Weighted Kappa | 0.814 |
| Macro AUC (one-vs-rest) | 0.916 |
| Macro F1 / Weighted F1 | 0.566 / 0.758 |
| Any-DR sensitivity / specificity | 94.7% / 96.9% |
| Referable-DR (stage ≥ 2) sensitivity / specificity | 81.5% / 94.9% |
| Predictions within one stage of the true grade | 92.1% |

## Repository structure

```
├── dr_stage_detection.ipynb   # full pipeline: EDA → preprocessing → training → evaluation → export
├── figures/                   # all figures produced by the notebook
├── app/
│   ├── streamlit_app.py       # web app: same preprocessing as training, prediction, NumPy Grad-CAM
│   ├── requirements.txt       # streamlit, ai-edge-litert, opencv, numpy, pillow (no TensorFlow)
│   ├── dr_features.tflite     # EfficientNetB3 backbone (TensorFlow Lite, 42 MB)
│   ├── dr_head.npz            # classification-head weights
│   ├── model_meta.json        # classes, preprocessing, test metrics (exported by the notebook)
│   └── examples/              # one test image per stage
└── README.md
```

### Lightweight deployment

Full TensorFlow needs ~1.1 GB of RAM, above the ~1 GB limit of free hosting. The final model is therefore exported as:
- the CNN backbone in **TensorFlow Lite**, run with the small **LiteRT** runtime, and
- the classification head (BatchNorm → Dense → Dense) as **NumPy** weights.

Because the head begins with Global Average Pooling, the Grad-CAM gradient is computed exactly in NumPy. The lite pipeline was verified to reproduce the Keras model's probabilities (max probability difference 2.3e-5, identical stage on 50/50 test images) with peak memory of ~220 MB.

## How to reproduce

1. Open the notebook on Kaggle, add the **APTOS 2019 Blindness Detection** competition as input.
2. Settings → Accelerator **GPU T4 x2**, Internet **On**.
3. Run all cells (~1.5 h). Outputs: `figures.zip`, `deployment.zip` (includes the lite model files for the app).

Run the app locally:

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Disclaimer

Research prototype for a university computer-vision coursework. Not a medical device and not for clinical use.
