# HDB Resale Price Predictor 🏠
*From Research to Production: A User-Centric Singapore Public Housing Price Prediction Application*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit 1.45.0](https://img.shields.io/badge/Streamlit-1.45.0-FF4B4B.svg)](https://streamlit.io/)
[![Pandas 2.2.3](https://img.shields.io/badge/Pandas-2.2.3-150458.svg)](https://pandas.pydata.org/)
[![Scikit-learn 1.6.1](https://img.shields.io/badge/Scikit--learn-1.6.1-F7931E.svg)](https://scikit-learn.org/)
[![Matplotlib 3.10.1](https://img.shields.io/badge/Matplotlib-3.10.1-11557c.svg)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project evolved from academic research to a **production-ready application** that helps Singaporeans make informed HDB resale purchase decisions. What started as a comprehensive statistical analysis with 150+ features has been thoughtfully refined into a user-centric prediction tool that balances accuracy with practicality.

> **Design Philosophy**: *Usability over perfect accuracy* - We prioritize features that real users can reasonably provide over granular academic variables that might marginally improve model performance.

### 🚀 Live Demo
Experience the application: [HDB Price Predictor Dashboard](https://adredes-weslee-making-predictions-on-hdb-resale-pric-app-iwznz4.streamlit.app)

---

## 🏗️ Project Evolution Journey

### 📊 Phase 1: Research Foundation
**Original Notebook Analysis** (`notebooks/making_predictions_on_HDB_resale_price.ipynb`)
- **Comprehensive EDA** with 150+ engineered features
- **Academic rigor**: Statistical significance testing, multicollinearity analysis
- **Best model**: 92.78% R² score with complex feature engineering
- **Focus**: Maximum predictive accuracy for research insights

### 🏭 Phase 2: Production Transformation
**Current Application** (`app/` + `src/` architecture)
- **User-centric design**: Streamlined to essential, user-providable features
- **Practical trade-offs**: "Distance to nearest MRT" vs. "Amenities within 500m/1km/2km"
- **Production pipeline**: Consistent preprocessing for training and inference
- **Focus**: Real-world usability for Singapore homebuyers

---

## ✨ Key Features

### 🎨 Interactive Dashboard
- **Intuitive input form** with Singapore-specific options
- **Real-time predictions** with confidence intervals
- **Market insights** and price trend visualizations
- **Responsive design** optimized for mobile and desktop

### 🔧 Smart Feature Engineering
- **Location intelligence**: MRT proximity, regional premiums
- **Temporal factors**: Lease decay effects, market timing
- **Property characteristics**: Flat type, storey range, floor area
- **Market context**: Historical trends and comparative analysis

### 🏗️ Production Architecture
```
├── app.py                  # Main Streamlit application entry point
├── app/                    # Streamlit application modules
│   ├── components/        # Reusable UI components (sidebar, visualizations)
│   ├── pages/             # Individual page modules
│   ├── views/             # Core view components (home, prediction, insights)
│   └── main.py            # Application routing and configuration
├── src/                   # Core business logic
│   ├── data/              # Data loading, preprocessing, and feature engineering
│   ├── models/            # Prediction models, training, and evaluation
│   ├── utils/             # Helper functions and utilities
│   └── visualization/     # Plotting and mapping utilities
├── models/                # Trained model artifacts (.pkl files and metadata)
├── data/                  # Dataset storage
│   ├── raw/               # Original HDB transaction data
│   └── processed/         # Cleaned and feature-engineered datasets
├── scripts/               # Training and utility scripts
├── configs/               # Configuration files (YAML, JSON)
├── notebooks/             # Original research notebook
└── requirements.txt       # Python dependencies
```

---

## 🚀 Quick Start

### 📋 Prerequisites
- Python 3.11 (recommended - matches development environment)
- Git (for cloning)
- 4GB+ RAM recommended

### 💻 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Making-Predictions-on-HDB-Resale-Price.git
cd Making-Predictions-on-HDB-Resale-Price
```

2. **Set up virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify setup**
```bash
python scripts/verify_environment.py
```

### 🎯 Launch the Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser, or visit the [live demo](https://adredes-weslee-making-predictions-on-hdb-resale-pric-app-iwznz4.streamlit.app) to start predicting HDB prices!

---

## 📱 Usage Guide

### 🏠 Making Predictions

1. **Select location details**
   - Choose town (e.g., "Ang Mo Kio", "Tampines")
   - Pick flat type (1-5 ROOM, EXECUTIVE, etc.)
   - Select block and street (if specific address needed)

2. **Enter property characteristics**
   - Floor area (sqm)
   - Storey range (01-03, 04-06, etc.)
   - Flat model (if applicable)

3. **Specify transaction details**
   - Lease commencement date
   - Transaction month/year

4. **Get your prediction**
   - Estimated resale price with confidence interval
   - Market insights and comparable properties
   - Price trend analysis

### 🔍 Understanding Results

- **Price Range**: Confidence interval based on model uncertainty
- **Market Context**: How this prediction compares to recent transactions
- **Key Factors**: Which features most influence this prediction
- **Recommendations**: Market timing and negotiation insights

---

## 🧠 Model Insights

### 📊 Performance Metrics
- **Final Model**: Ridge Regression with engineered features
- **RMSE**: ~39,180 SGD (approximately 8.5% of mean price)
- **R² Score**: 0.9261 (92.61% variance explained)
- **Cross-validation**: 5-fold CV for robust performance estimation

### 🔑 Key Price Drivers

1. **📍 Location Factors**
   - Central region premium: +15-25%
   - MRT proximity: Significant impact within 800m
   - Mature vs non-mature estates

2. **🏠 Property Characteristics**
   - Floor area: Linear relationship with price
   - Storey range: Mid-floors (7-12) often optimal
   - Flat model: DBSS and Model A command premiums

3. **⏰ Temporal Factors**
   - Lease decay: Accelerated below 60 years remaining
   - Market cycles: Seasonal and economic influences
   - Recent transaction trends

4. **🏗️ Development Features**
   - Block age vs renovation cycles
   - Nearby amenities and infrastructure
   - Future development plans impact

---

## 🛠️ Development Guide

### 🧪 Training Your Own Models

```bash
# Process fresh data
python scripts/preprocess_data.py

# Train production pipeline model
python scripts/train_pipeline_model.py --model-type ridge

# Train comparison models
python scripts/train_models.py --model all
```

### 🔧 Using as Python Package

```python
from src.data.loader import load_training_data
from src.models.prediction import make_prediction
from joblib import load

# Load pre-trained model
model = load("models/pipeline_ridge_model.pkl")

# Prepare input data
sample_input = {
    'town': 'ANG MO KIO',
    'flat_type': '4 ROOM',
    'floor_area_sqm': 95.0,
    'storey_range': '07 TO 09',
    'lease_commence_date': 1985,
    # ... other features
}

# Make prediction
prediction = make_prediction(model, sample_input)
print(f"Predicted price: ${prediction:,.2f}")
```

### 📊 API Integration

For programmatic access:

```python
import requests

response = requests.post(
    "http://localhost:8501/api/predict",
    json={"features": sample_input}
)
prediction = response.json()["prediction"]
```

---

## 🏗️ Architecture Decisions

### 🎯 Design Trade-offs

| **Research Approach** | **Production Approach** | **Rationale** |
|----------------------|------------------------|---------------|
| 150+ engineered features | ~15 core features | User data availability |
| Complex ensemble models | Ridge regression | Interpretability + performance |
| Perfect historical accuracy | Real-time usability | Production constraints |
| Academic feature importance | User-providable inputs | Practical deployment |

### 🔄 Pipeline Philosophy

1. **Consistency First**: Same preprocessing for training and inference
2. **Fail-Safe Defaults**: Graceful handling of missing/invalid inputs
3. **Scalable Architecture**: Modular design for easy feature additions
4. **User-Centric**: Every feature must be reasonably obtainable by users

---

## 📈 Business Impact

### 🏠 For Homebuyers
- **Informed decisions**: Data-driven price expectations
- **Market timing**: Understand optimal buying windows
- **Negotiation power**: Armed with market-rate estimates
- **Risk assessment**: Confidence intervals show uncertainty

### 🏢 For Industry
- **Market transparency**: Standardized pricing insights
- **Policy research**: Data-driven housing policy inputs
- **Real estate analytics**: Comparable transaction analysis
- **Academic research**: Open-source methodology

---

## 🔬 Technical Deep Dive

### 📊 Data Engineering
- **Source**: HDB resale transaction data (2017-2023)
- **Processing**: Automated ETL with data validation
- **Features**: Geographic, temporal, and property characteristics
- **Quality**: Outlier detection and data consistency checks

### 🤖 Model Pipeline
```python
Pipeline([
    ('imputer', IterativeImputer()),
    ('scaler', StandardScaler()),
    ('encoder', TargetEncoder()),
    ('selector', SelectKBest()),
    ('regressor', Ridge(alpha=1.0))
])
```

### 🔍 Feature Selection Strategy
1. **Domain expertise**: Singapore housing market knowledge
2. **Statistical analysis**: Mutual information and correlation
3. **User availability**: Can typical buyers provide this data?
4. **Stability**: Features that remain relevant over time

---

## 🚀 Future Enhancements

### 🎯 Short-term (Next 3 months)
- [ ] **Mobile app** with native iOS/Android interface
- [ ] **Email alerts** for price drops in selected areas
- [ ] **Mortgage calculator** integration
- [ ] **Comparative market analysis** reports

### 🌟 Medium-term (6 months)
- [ ] **Real-time data** integration via HDB APIs
- [ ] **Neighborhood scoring** beyond just price prediction
- [ ] **Investment analysis** tools (rental yield, etc.)
- [ ] **Market trend** forecasting

### 🚀 Long-term (1 year+)
- [ ] **Computer vision** for property condition assessment
- [ ] **NLP integration** for news sentiment impact
- [ ] **Multi-model ensemble** with deep learning
- [ ] **API marketplace** for third-party integrations

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### 🛠️ Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test thoroughly
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open a Pull Request

### 📊 Areas for Contribution
- **Data engineering**: New feature engineering ideas
- **UI/UX**: Dashboard improvements and mobile optimization
- **Models**: Alternative algorithms and ensemble methods
- **Documentation**: Tutorials and usage examples

---

## 📚 References & Resources

### 🏛️ Singapore Housing Context
- [HDB Official Portal](https://www.hdb.gov.sg/)
- [Singapore Housing Market Analysis](https://www.channelnewsasia.com/commentary/emphasis-home-ownership-hdb-lease-review-of-public-housing-2071266)
- [Public Housing as Singapore Icon](https://www.hdb.gov.sg/about-us/our-role/public-housing-a-singapore-icon)

### 🔬 Technical References
- [Scikit-learn Pipeline Documentation](https://scikit-learn.org/stable/modules/compose.html)
- [Handling Missing Values with IterativeImputer](https://medium.com/towards-data-science/a-better-way-to-handle-missing-values-in-your-dataset-using-iterativeimputer-9e6e84857d98)
- [Multicollinearity in Regression Analysis](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/)

### 📖 Methodology Papers
- Ridge Regression and Regularization Techniques
- Feature Selection in High-Dimensional Data
- Real Estate Price Prediction: A Survey

---

## 🏆 Acknowledgments

Special thanks to:
- **HDB Singapore** for providing transparent public data
- **Singapore data science community** for feedback and insights
- **Open source contributors** who made this project possible

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Wes Lee**  
📍 Singapore  
🔗 [LinkedIn](https://www.linkedin.com/in/wes-lee)  
📧 [Contact](mailto:weslee.qb@gmail.com)

---

*Built with ❤️ for the Singapore community*

**⭐ Star this repository if it helped you make a better HDB purchase decision!**
