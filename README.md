# 🏘️ Predicting HDB Resale Prices in Singapore

## 📘 Overview
This project builds a predictive model for **HDB flat resale prices** in Singapore, a country where over 80% of residents live in public housing. Using a dataset of 60K+ transactions from [Kaggle](https://www.kaggle.com/datasets), we explore key factors affecting housing value, including flat type, lease decay, location, and urban planning policies.

## 🌱 Project Evolution
This project has evolved from a data science exploration to a full-featured application:

1. **Initial Notebook Phase**: Started as a Jupyter notebook focused on exploratory data analysis and model prototyping
2. **Code Restructuring**: Refactored into a modular Python package with proper separation of concerns
3. **Pipeline Development**: Created reproducible preprocessing and model training pipelines
4. **Interactive Dashboard**: Built a Streamlit web application for easy data exploration and price prediction

The application now provides both the analytical rigor of the original notebook and an accessible interface for non-technical users.

## 📌 Project Structure

```
hdb-price-prediction/
├── app/                      # Streamlit application
│   ├── main.py               # Main entry point for Streamlit
│   ├── views/                # Different views for the application
│   └── components/           # Reusable UI components
│
├── src/                      # Core logic
│   ├── data/                 # Data processing modules
│   │   ├── preprocessing_pipeline.py  # Standardized data preprocessing
│   │   └── loader.py         # Data loading utilities
│   ├── models/               # Model related code
│   │   ├── training.py       # Model training functions
│   │   └── prediction.py     # Prediction utilities
│   ├── visualization/        # Visualization code
│   └── utils/                # Utility functions
│
├── models/                   # Pre-trained model artifacts
│   ├── pipeline_linear_model.pkl     # Linear regression pipeline
│   ├── pipeline_ridge_model.pkl      # Ridge regression pipeline
│   └── pipeline_lasso_model.pkl      # Lasso regression pipeline
│
├── data/                     # Data files
│   ├── raw/                  # Original data
│   └── processed/            # Processed datasets
│       ├── train_processed.csv             # Pipeline-compatible training data
│       └── train_processed_exploratory.csv # Data for visualization
│
├── configs/                  # Configuration files
│   ├── model_config.yaml     # Model hyperparameters and settings
│   └── app_config.yaml       # Streamlit application settings
│
├── scripts/                  # Utility scripts
│   ├── verify_environment.py          # Check dependencies
│   ├── preprocess_data.py             # Data preprocessing script
│   ├── train_models.py                # Traditional model training
│   └── train_pipeline_model.py        # Pipeline model training
│
├── app.py                   # Entry point for Streamlit app
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## 🎯 Features
- **Interactive Dashboard**: Streamlit-powered web application for easy exploration
- **Data Exploration**: Visual analysis of HDB resale data with interactive filters
- **Price Prediction**: Predict HDB resale prices with an intuitive interface
- **Model Insights**: Understand feature importance and model performance
- **Geospatial Analysis**: Map-based visualization of property values
- **Reproducible Pipelines**: Consistent preprocessing between training and prediction

## 📂 Data Dictionary  
Key variables engineered or used:  

| Feature                   | Description                                             |
|---------------------------|---------------------------------------------------------|
| `resale_price`            | Dependent variable (SGD)                                |
| `floor_area_sqm`          | Flat size in square meters                              |
| `remaining_lease`         | Years of lease remaining                                |
| `flat_model`              | Type of flat model (e.g. Model A, Improved, DBSS)       |
| `town`, `region`          | Location metadata derived from `town`                   |
| `storey_range_encoded`    | Numerical storey range                                  |
| `lease_commence_year`     | Year flat was built                                     |
| `amenities_proximity`     | Distance to MRT, malls, schools (proxy values)          |
| `flat_type`               | Categorical flat size (e.g. 3-room, 5-room, Executive)  |

## 🚀 Installation & Usage

### 📦 Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hdb-price-prediction.git
cd hdb-price-prediction

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode (optional)
pip install -e .
```

### 🖥️ Running the Application

```bash
# Launch the Streamlit application
streamlit run app.py
```

### 🧰 Development Commands

```bash
# Process raw data for exploration
python scripts/preprocess_data.py

# Train pipeline models (creates artifacts for Streamlit)
python scripts/train_pipeline_model.py --model-type ridge

# Train traditional models for comparison
python scripts/train_models.py --model all

# Verify environment setup
python scripts/verify_environment.py
```

### 📊 Using the API
The core functionality can also be used as a Python package:


```python
from src.data.loader import load_training_data
from src.models.prediction import make_prediction

# Load data and model
X, y = load_training_data("path/to/data.csv")

# Load a pre-trained pipeline model
from joblib import load
model = load("models/pipeline_ridge_model.pkl")

# Make predictions
predictions = model.predict(X)
```

## 🧪 Methodology  

### 🧭 Data Preparation
- Implemented two preprocessing approaches:
  - **Exploratory pipeline** for data analysis and visualization
  - **Model pipeline** for consistent preprocessing in production
- Created scikit-learn transformers for categorical and numerical features
- Scaled features using StandardScaler and encoded categorical variables

### 🔍 EDA & Feature Selection  
- Explored price trends by region, flat type, and lease decay  
- Assessed multicollinearity using VIF and correlation matrices  
- Used **mutual information** and **domain heuristics** for feature pruning  

### 📊 Modeling  
- Trained **Linear Regression**, **Ridge**, and **Lasso** models  
- Implemented full scikit-learn pipelines for reproducible preprocessing
- Cross-validated with 5-fold CV; selected Ridge for final output  
- RMSE: ~39,180 SGD | R² Score: 0.9261  

## 🧠 Key Insights  
- **Lease Decay Matters:** Prices decline as lease falls below 60 years  
- **Flat Model Effects:** DBSS and Model A flats fetch higher premiums  
- **Location Premiums:** Central region flats are priced significantly higher  
- **Storey Ranges & Floor Area:** Mid-floor units and larger flats command better value  

## 🏛 Policy Implications  
- Consider **differentiated subsidies** for lower lease resale flats  
- Improve **public data transparency** on amenities and nearby upgrades  
- Use data-driven signals to **detect pricing anomalies** or bubbles  

## 🔧 Tools & Technologies
- **Core**: Python, Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Interface**: Streamlit
- **DevOps**: Git, GitHub
- **Configuration**: YAML

## 📁 Project Components
- **Original Notebook**: making_predictions_on_HDB_resale_price.ipynb
- **Streamlit Application**: Entry point app.py, views in views
- **Model Pipelines**: Defined in preprocessing_pipeline.py
- **Training Scripts**: train_pipeline_model.py

## 📚 References  

**HDB & Singapore Housing Policy**  
- https://www.hdb.gov.sg/about-us/our-role/public-housing-a-singapore-icon  
- https://www.channelnewsasia.com/commentary/emphasis-home-ownership-hdb-lease-review-of-public-housing-2071266  
- [Additional references omitted for brevity]

**Modeling & Statistics**  
- https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/  
- https://medium.com/towards-data-science/a-better-way-to-handle-missing-values-in-your-dataset-using-iterativeimputer-9e6e84857d98#f326  
- [Additional references omitted for brevity]

---  

**Author:** Wes Lee  
📍 Singapore | 🔗 [LinkedIn](https://www.linkedin.com/in/wes-lee)  
📜 License: MIT