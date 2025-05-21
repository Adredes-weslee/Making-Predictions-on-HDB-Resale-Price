# Processed Data Directory

This directory contains processed data files created by different preprocessing scripts:

## Exploratory Files
Files with `_exploratory` in the name are created by the `preprocess_data.py` script and are 
primarily intended for exploratory data analysis and visualization in the Streamlit dashboard.

⚠️ **Warning**: These files may not be compatible with models trained using the pipeline approach.

## Pipeline-Compatible Files
Files without the `_exploratory` suffix are created by the `train_pipeline_model.py` script and
use a consistent scikit-learn pipeline approach that ensures compatibility between training and prediction.

For model training and evaluation, please use the pipeline-compatible files.
