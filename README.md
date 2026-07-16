# FreshProduce Vision — Shelf-Life Predictor

Steps-only repository: **code + docs to train and run** a multitask fruit/vegetable shelf-life model.  
**No SavedModel weights, no `.keras` checkpoints, and no image dataset** are shipped in this repo.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

## What you get

| Included | Not included |
|----------|----------------|
| Streamlit app (`streamlit_app/`) | Model weights / SavedModel |
| Retrain & notebook scripts (`scripts/`) | Raw `images/` dataset |
| Training notebook | Zip archives of images |
| Annotations metadata, attribution | Pre-built `streamlit_model_bundle/` |

You train locally, then point the app at the bundle you export.

---

## Steps to run end-to-end

### 1. Clone and install

```bash
git clone https://github.com/Ganapathy-bot/FreshProduce-Vision.git
cd FreshProduce-Vision

pip install -r streamlit_app/requirements.txt
pip install scikit-learn pandas tqdm   # needed for retrain script
```

### 2. Add the image dataset (local only)

Place the Image Pack so it looks like:

```text
FreshProduce-Vision/
  images/
    Fruit/
      Apple/Fresh/*.jpg
      Apple/Spoiled/*.jpg
      Banana/...
    Vegetable/
      Tomato/Fresh/*.jpg
      ...
```

Do **not** set `IMAGES_DIR` to a bare `/kaggle` root. That yields broken 1-class labels (`input` / `datasets`).

### 3. Train and export the model bundle

From the **repo root**:

```bash
python scripts/retrain_streamlit_bundle.py
```

This writes (locally, gitignored):

```text
streamlit_model_bundle/
  label_maps.json
  run_config.json
  best_model_efficientnetv2b0_multitask.keras
  savedmodel_efficientnetv2b0_multitask/
    saved_model.pb
    variables/
```

Optional environment overrides:

| Variable | Default | Meaning |
|----------|---------|---------|
| `IMAGES_DIR` | `./images` | Dataset root |
| `OUT_DIR` | `./streamlit_model_bundle` | Export path |
| `MAX_PER_COMBINED` | `120` | Cap per produce×stage (`0` = use all) |
| `EPOCHS_HEAD` | `3` | Frozen-backbone epochs |
| `EPOCHS_FINETUNE` | `2` | Fine-tune epochs |
| `BATCH_SIZE` | `16` | Batch size |
| `BACKBONE` | `efficientnetv2b0` | Or `mobilenetv3` |

Example (Windows PowerShell):

```powershell
$env:IMAGES_DIR = "C:\path\to\images"
$env:MAX_PER_COMBINED = "0"   # full dataset (slower)
$env:EPOCHS_HEAD = "8"
$env:EPOCHS_FINETUNE = "20"
python scripts\retrain_streamlit_bundle.py
```

**Sanity check after export**

- `label_maps.json` → many real classes (e.g. Apple, Tomato, Fresh, Spoiled)
- **Not** only `input` / `datasets`
- SavedModel heads have shape **> 1** per head (e.g. 13 categories, 6 freshness stages)

### 4. Run the Streamlit app

```bash
cd streamlit_app
streamlit run app.py
```

Open http://localhost:8501

The app loads `../streamlit_model_bundle` by default. Override if needed:

```powershell
# Windows
$env:MODEL_DIR = "C:\path\to\streamlit_model_bundle"
streamlit run app.py
```

```bash
# Linux / macOS
export MODEL_DIR=/path/to/streamlit_model_bundle
streamlit run app.py
```

### 5. Use the UI

1. Upload or capture a produce image  
2. Click **Predict shelf life**  
3. Read category, freshness stage, proxy shelf-life days, and confidence  

**Notes**

- **Confidence** = model certainty about the **class pick**, not “how fresh” as a percentage.  
- **Shelf life** is a **proxy** from the predicted stage (not a food-safety guarantee).

---

## Optional: full research notebook

```bash
# regenerate notebook from script if needed
python scripts/generate_notebook.py
```

Open `FreshProduce_ShelfLife_TransferLearning.ipynb` and set:

```python
IMAGES_DIR = Path("images")  # real images tree — not Path("/kaggle") alone
```

Then run cells: data scan → labels → train → export SavedModel.

---

## Repository layout

```text
.
├── streamlit_app/          # Inference UI (no weights)
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── scripts/
│   ├── retrain_streamlit_bundle.py   # main train + export steps
│   ├── generate_notebook.py
│   └── build_ml_splits.py
├── annotations/            # CSV / split lists (metadata only)
├── sources/ATTRIBUTION.md
├── FreshProduce_ShelfLife_TransferLearning.ipynb
├── .gitignore              # excludes images + all model weights
└── README.md               # this file
```

---

## Expected model outputs (after you train)

| Head | Typical classes |
|------|-----------------|
| Category | Apple, Banana, BellPepper, BitterGourd, Cucumber, Eggplant, Mango, Okra, Orange, Papaya, Potato, Strawberry, Tomato |
| Freshness | Fresh, Semi_Fresh, Early_Ripening, Mid_Ripening, Fully_Ripe, Spoiled |

Stage → proxy days (see retrain script `DEFAULT_RSL_MAP`): Fresh≈7 … Spoiled=0.

---

## Dataset caveats

- Pack is assembled from open sources; not lab Day0→Spoiled longitudinal photography.  
- Proxy RSL only.  
- Cite upstream datasets before publishing (see `sources/ATTRIBUTION.md`).

---

## License & attribution

Code in this repo is for research/prototyping.  
Image data and ImageNet backbone weights remain under their original licenses. Review upstream READMEs before redistribution.

GitHub: [Ganapathy-bot/FreshProduce-Vision](https://github.com/Ganapathy-bot/FreshProduce-Vision)
