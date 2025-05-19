# 🏘️ Predicting HDB Resale Prices in Singapore  

## 📘 Overview  
This project builds a predictive model for **HDB flat resale prices** in Singapore, a country where over 80% of residents live in public housing. Using a dataset of 60K+ transactions from [Kaggle](https://www.kaggle.com/datasets), we explore key factors affecting housing value, including flat type, lease decay, location, and urban planning policies.

The application is designed as a comprehensive dashboard with data exploration, visualization, and price prediction capabilities through an interactive Streamlit interface.

## 📌 Project Structure
```
hdb-price-prediction/
├── app/                      # Streamlit application
│   ├── main.py               # Main entry point for Streamlit
│   ├── pages/                # Multiple pages for different features
│   └── components/           # Reusable UI components
│
├── src/                      # Core logic
│   ├── data/                 # Data processing
│   ├── models/               # Model related code
│   ├── visualization/        # Visualization code
│   └── utils/                # Utility functions
│
├── models/                   # Pre-trained model files
├── data/                     # Data files
│   ├── raw/                  # Original data
│   └── processed/            # Processed datasets
├── configs/                  # Configuration files
├── scripts/                  # Utility scripts
├── tests/                    # Unit tests
├── notebooks/                # Original notebooks (for reference)
│
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup
├── Makefile                 # Make commands for common tasks
└── README.md                # Project documentation
```

## 🎯 Features
- **Interactive Dashboard**: Streamlit-powered web application for easy exploration
- **Data Exploration**: Visual analysis of HDB resale data with interactive filters
- **Price Prediction**: Predict HDB resale prices with an intuitive interface
- **Model Insights**: Understand feature importance and model performance
- **Geospatial Analysis**: Map-based visualization of property values

## 📂 Data Dictionary  
Key variables engineered or used:  

| Feature                   | Description                                             |
|---------------------------|---------------------------------------------------------|
| `resale_price`            | Dependent variable (SGD)                                |
| `floor_area_sqm`          | Flat size in square meters                              |
| `remaining_lease`         | Years of lease remaining                                |
| `flat_model`              | Type of flat model (e.g. Model A, Improved, DBSS)       |
| `town`, `region`          | Location metadata derived from `town`                  |
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

# Install the package in development mode
pip install -e .
```

### 🖥️ Running the Application
```bash
# Method 1: Using the Makefile
make run

# Method 2: Using streamlit directly
streamlit run app/main.py
```

### 🧰 Development Commands
```bash
# Process raw data
make process-data

# Train models
make train-models

# Run tests
make test

# Format code
make format
```

### 📊 Using the API
The core functionality can also be used as a Python package:

```python
from src.data.loader import load_raw_data
from src.models.prediction import load_model, predict

# Load data and model
df = load_raw_data("path/to/data.csv")
model = load_model("models/linear_regression_model.pkl")

# Make predictions
predictions = predict(model, df)
```

## 🧪 Methodology  

### 🧭 Data Preparation
- Imputed missing values using **IterativeImputer**  
- Derived features like lease decay and encoded location metadata  
- Scaled continuous features for regression consistency  

### 🔍 EDA & Feature Selection  
- Explored price trends by region, flat type, and lease decay  
- Assessed multicollinearity using VIF and correlation matrices  
- Used **mutual information** and **domain heuristics** for feature pruning  

### 📊 Modeling  
- Trained **Linear Regression**, **Ridge**, and **Lasso** models  
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

## 🔍 Tools & Libraries  
- Python · Pandas · NumPy · Scikit-learn · Seaborn · Matplotlib  
- Jupyter Notebook  

## 📁 Files  
- `making_predictions_on_HDB_resale_price.ipynb` – full EDA + modeling  
- `README.md` – this documentation  

## 📚 References  

**HDB & Singapore Housing Policy**  
- https://www.hdb.gov.sg/about-us/our-role/public-housing-a-singapore-icon  
- https://www.channelnewsasia.com/commentary/emphasis-home-ownership-hdb-lease-review-of-public-housing-2071266  
- https://www.hdb.gov.sg/about-us/history  
- https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families/cpf-housing-grants-for-resale-flats-families  
- https://www.cpf.gov.sg/member/cpf-overview  
- https://www.worldbank.org/en/country/singapore/overview  
- https://www.ura.gov.sg/Corporate/Planning/Long-Term-Plan-Review  
- https://cnaluxury.channelnewsasia.com/exceptional-homes/singaporean-home-buying-sentiment-h2-2021-191066  
- https://www.straitstimes.com/politics/continuity-in-policies-key-to-singapores-success-says-chan-chun-sing  
- https://dreamhomessg.co/dont-wait-its-time-to-buy-why-home-prices-in-singapore-are-unlikely-to-drop-in-2023/  
- https://www.businesstimes.com.sg/property/singapore-households-net-worth-grows-residential-asset-values-climb  
- https://www.channelnewsasia.com/singapore/hdb-resale-prices-every-singapore-town-current-property-boom-3043466  
- https://www.businesstimes.com.sg/property/proportion-delayed-bto-projects-down-90-40-hdb-clear-backlog-two-years  
- https://www.businesstimes.com.sg/lifestyle/rising-appeal-resale-hdb-flats  
- https://endowus.com/insights/planning-finances-hdb-bto  
- https://sbr.com.sg/exclusive/hdb-resale-good-investment  
- https://www.teoalida.com/singapore/hdbfloorplans/  
- https://www.redbrick.sg/blog/20-housing-types-singapore-1/  
- https://blog.carousell.com/property/hdb-flat-types-singapore/  
- https://sg.finance.yahoo.com/news/different-types-hdb-houses-call-020000642.html  
- https://www.propertyguru.com.sg/property-guides/dbss-singapore-17893  
- https://getforme.com/previous2004/previous290504_hdblaunchesthepinnancleatduxton.htm  

**Modeling & Statistics**  
- https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/  
- https://medium.com/towards-data-science/a-better-way-to-handle-missing-values-in-your-dataset-using-iterativeimputer-9e6e84857d98#f326  
- https://towardsdatascience.com/how-and-why-to-standardize-your-data-996926c2c832  
- https://saturncloud.io/blog/linear-regression-implementing-feature-scaling/  
- https://towardsdatascience.com/drop-first-can-hurt-your-ols-regression-models-interpretability-4ca529cfb707  
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html  
- https://guhanesvar.medium.com/feature-selection-based-on-mutual-information-gain-for-classification-and-regression-d0f86ea5262a  
- https://machinelearningmastery.com/information-gain-and-mutual-information/  
- https://towardsdatascience.com/understanding-entropy-the-golden-measurement-of-machine-learning-4ea97c663dc3  
- https://towardsdatascience.com/explaining-negative-r-squared-17894ca26321  
- http://www.fairlynerdy.com/what-is-r-squared/  
- https://statisticsbyjim.com/regression/root-mean-square-error-rmse/  
- https://neptune.ai/blog/feature-selection-methods  
- https://statisticsbyjim.com/regression/ols-linear-regression-assumptions/  

---  

**Author:** Wes Lee  
📍 Singapore | 🔗 [LinkedIn](https://www.linkedin.com/in/wes-lee)  
📜 License: MIT  
