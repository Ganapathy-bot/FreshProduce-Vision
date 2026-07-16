# FreshProduce Streamlit App

Upload fruit/vegetable images and get:

- **Category** (Apple, Tomato, …)
- **Freshness stage** (Fresh, Fully_Ripe, Spoiled, …)
- **Remaining shelf life** (days, proxy map)
- **Confidence** (+ top-k probabilities)

## 1. Model bundle

Use the full folder (not only `saved_model.pb`):

```text
streamlit_model_bundle/
  label_maps.json
  run_config.json
  savedmodel_efficientnetv2b0_multitask/
    saved_model.pb
    variables/
      variables.data-00000-of-00001
      variables.index
```

Default path: `../streamlit_model_bundle` next to this app  
(or set `MODEL_DIR`).

## 2. Install & run

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Optional:

```bash
set MODEL_DIR=C:\path\to\streamlit_model_bundle
streamlit run app.py
```

## 3. Features in the UI

| Control | Description |
|--------|-------------|
| Upload | Choose one or more images |
| Camera | Capture a photo |
| Predict | Runs model + shows metrics |
| Top-K | How many class probabilities to list |
| Model dir | Override bundle path |

## 4. Note on labels

If `label_maps.json` was exported incorrectly (e.g. classes like `"input"` / `"datasets"`), retrain with `IMAGES_DIR` pointing at your real `images/` folder and re-export `label_maps.json`.
