# HDB Resale Price Predictor

A Streamlit app for Singapore HDB resale price exploration and point prediction, backed by a research notebook in `notebooks/`.

The repo is organized around a user-facing dashboard, not just a standalone ML script.

## Why This Repository Exists

- Estimate HDB resale prices from features a buyer can realistically provide: town, flat type/model, storey range, floor area, lease year, nearby amenities, school proximity, and transaction date.
- The app copy frames this for Singapore homebuyers and public-housing market exploration, with model output intended to support pricing expectations and market understanding.

## Architecture at a Glance

- `app.py` is only a bootstrap wrapper; it adjusts `sys.path` and calls `app.main.main()`.
- app/main.py configures Streamlit and routes four views: Home, Data Explorer, Make Prediction, and Model Performance.
- Page logic lives in app/views/home.py, app/views/data_explorer.py, app/views/prediction.py, and app/views/model_insights.py, with shared UI in `app/components/`.
- `src/` contains preprocessing, feature engineering, training/evaluation/prediction, utilities, and visualization code; `scripts/` holds environment verification, exploratory preprocessing, exploratory model training, and pipeline training; `configs/` centralizes app/model/feature settings.
- The repo ships raw data in `data/raw/`, processed outputs in `data/processed/`, and three checked-in pipeline artifacts in `models/` for linear, ridge, and lasso.

## Repository Layout

- `app/`
- `configs/`
- `data/`
- `models/`
- `notebooks/`
- `scripts/`
- `src/`
- `.gitignore`
- `app.py`
- `environment.yaml`
- `README.md`
- `requirements.txt`

## Setup and Run

1. Use Python 3.11.0 from `environment.yaml`; the pinned package set lives in `environment.yaml` and `requirements.txt`.
2. Run scripts/verify_environment.py to check Python, packages, directories, configs, model files, data files, and Streamlit.
3. Launch the app with `streamlit run app.py`.
4. If you retrain, use `python scripts/preprocess_data.py [--no-save] [--debug]` for exploratory outputs, `python scripts/train_models.py --model [linear|ridge|lasso|all] [--evaluate] [--no-save]` for exploratory models, or `python scripts/train_pipeline_model.py --model-type [linear|ridge|lasso]` for pipeline-compatible models.
5. The repo splits processing into exploratory and pipeline-compatible tracks, so decide which path you want before running training.

## Core Workflows

- The Home page loads summary stats from `data/processed/train_processed_exploratory.csv` and caches town/flat-type options for reuse.
- Data Explorer reads the exploratory processed file, reconstructs a `year_month` timeline, and shows price distribution, trends, feature-by-feature views, and correlations.
- Make Prediction loads option ranges from processed data, derives date/age features automatically, and returns a price plus model R2/RMSE.
- Model Performance compares the three loaded models with metric charts and coefficient analysis.
- Training is intentionally split between exploratory preprocessing/training and the pipeline-compatible path in `scripts/train_pipeline_model.py`.

## Known Limitations

- The checked-in `models/*_metrics.json` files report test R^2 around 0.8987 and RMSE around 45.4k, which does not match older documented figures of R^2 0.9261 and RMSE ~39,180.
- The code exposes point predictions in the Streamlit page and local helpers in `src/models/prediction.py`; there is no API route layer in the repo.
- Some older documentation still refers to `app/pages/`, but the current app uses `app/views/` and `app/components/`.
- The repo root does not include `CONTRIBUTING.md` or `LICENSE`.
- The exploratory preprocessing path is not pipeline-compatible, so exploratory and training instructions need to stay separate.
- One loader path still points at `data/processed/train_processed.csv`, which is not one of the checked-in processed outputs.
