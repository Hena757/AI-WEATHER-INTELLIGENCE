# 🌦️ AI Weather Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Deploy](https://img.shields.io/badge/Deploy-Render%20%7C%20AWS-orange.svg)](DEPLOYMENT.md)
[![PRs](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](CONTRIBUTING.md)

A production-ready **Machine Learning platform** that predicts whether it will rain tomorrow in Australia using advanced **Explainable AI (SHAP)**, serving predictions through both a **REST API** and an **interactive Streamlit dashboard**.

---

## ✨ Features

### 🧠 Machine Learning
- **6 model comparison**: XGBoost, LightGBM, Random Forest, Gradient Boosting, Logistic Regression, Decision Tree
- **Best model**: XGBoost with **ROC-AUC 0.8911** and **85.78% accuracy**
- **Hyperparameter tuning** with GridSearchCV and cross-validation
- **Explainable AI** with SHAP (SHapley Additive exPlanations)

### 🔍 Explainable AI (SHAP)
- **Global explanations**: Beeswarm summary plots, feature importance, dependence plots
- **Local explanations**: Waterfall plots, force plots, per-prediction contribution tables
- **Interactive dashboard** for transparent model interpretation

### 🌐 Live Weather Integration
- **OpenWeather API** integration (with Open-Meteo fallback)
- **Automatic preprocessing** of live data into model-ready features
- **API key management** with environment variable support
- **Smart caching** (5-minute TTL) to reduce API calls

### 🚀 Production-Ready API
- **RESTful endpoints** for predictions, health checks, and model metadata
- **Structured JSON responses** with consistent error handling
- **Gunicorn** production server configuration
- **CORS support** for cross-origin requests

### 📊 Interactive Dashboard
- **5-page Streamlit app** with modern responsive design
- **Live weather predictions** with confidence scores and SHAP explanations
- **Historical analysis** with interactive Plotly charts
- **Multi-city weather comparison**
- **Model insights** with performance metrics

### 🐳 Docker Deployment
- **Multi-stage Docker builds** for production optimization
- **Docker Compose** orchestration with health checks and shared volumes
- **Non-root users** for security
- **Ready for Render.com and AWS** deployment

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [API Documentation](#api-documentation)
- [Dashboard](#dashboard)
- [Explainable AI](#explainable-ai)
- [Performance Metrics](#performance-metrics)
- [Deployment](#deployment)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AI Weather Intelligence                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐    ┌───────────────────────────────────────────┐ │
│  │  Data Sources    │    │  Application Layer                        │ │
│  │                  │    │                                           │ │
│  │  ┌────────────┐  │    │  ┌─────────────┐   ┌──────────────────┐  │ │
│  │  │ weatherAUS │  │    │  │  Flask API  │   │  Streamlit        │  │
│  │  │ .csv       │  │    │  │  (port 8000)│   │  Dashboard         │  │
│  │  └────────────┘  │    │  └──────┬──────┘   │  (port 8501)      │  │
│  │                  │    │         │          └────────┬─────────┘  │ │
│  │  ┌────────────┐  │    │         └──────────┬────────┘            │ │
│  │  │ OpenWeather │  │    │                    │                      │ │
│  │  │ API         │  │    │  ┌─────────────────▼────────────────┐   │ │
│  │  └────────────┘  │    │  │  Shared Services                  │   │ │
│  │  ┌────────────┐  │    │  │  - src/data_preprocessing.py      │   │ │
│  │  │ Open-Meteo │  │    │  │  - src/feature_engineering.py     │   │ │
│  │  │ (fallback) │  │    │  │  - src/explainability.py           │   │ │
│  │  └────────────┘  │    │  │  - src/weather_service.py          │   │ │
│  └──────────────────┘    │  └──────────────────┬────────────────┘   │ │
│                          │                     │                     │ │
│                          │  ┌──────────────────▼────────────────┐   │ │
│                          │  │  Model Layer                      │   │ │
│                          │  │  - models/best_model.joblib       │   │ │
│                          │  │  - models/preprocessing_metadata  │   │ │
│                          │  └───────────────────────────────────┘   │ │
│  ┌──────────────────┐    │                                           │ │
│  │  Docker Compose  │    │                                           │ │
│  │  - api service   │    │                                           │ │
│  │  - dashboard svc │    │                                           │ │
│  └──────────────────┘    │                                           │ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

### Source
The model is trained on the **[Australian Weather Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)** from Kaggle, containing daily weather observations from numerous Australian weather stations.

### Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Records** | 142,193 |
| **Features** | 35 (after feature engineering) |
| **Locations** | 49 Australian cities |
| **Date Range** | Nov 2007 - Jun 2017 |
| **Target Variable** | `RainTomorrow` (Yes/No) |
| **Class Balance** | ~78% No Rain, ~22% Rain |

### Key Features

#### Original Features (21)
- **Temperature**: `MinTemp`, `MaxTemp`, `Temp9am`, `Temp3pm`
- **Rainfall**: `Rainfall`, `RainToday`
- **Humidity**: `Humidity9am`, `Humidity3pm`
- **Pressure**: `Pressure9am`, `Pressure3pm`
- **Wind**: `WindGustDir`, `WindGustSpeed`, `WindDir9am`, `WindDir3pm`, `WindSpeed9am`, `WindSpeed3pm`
- **Clouds**: `Cloud9am`, `Cloud3pm`
- **Other**: `Location`, `Evaporation`, `Sunshine`

#### Engineered Features (14)
- `TemperatureDifference` = MaxTemp - MinTemp
- `AverageTemperature` = (MinTemp + MaxTemp) / 2
- `HumidityIndex` = (Humidity9am + Humidity3pm) / 2
- `PressureDifference` = Pressure9am - Pressure3pm
- `Month`, `Quarter`, `DayOfYear`, `Season`, `IsWeekend`
- `WindIntensityCategory` (Low/Moderate/High/VeryHigh)
- `RainfallIndicator`, `RainTodayBinary`

---

## 📦 Installation

### Prerequisites
- **Python 3.12+**
- **pip** package manager
- **Docker** (optional, for containerized deployment)
- **OpenWeather API key** (optional, free at [openweathermap.org](https://openweathermap.org/api))

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/Hena757/AI-WEATHER-INTELLIGENCE.git
cd AI-WEATHER-INTELLIGENCE

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Docker Installation

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Set your OpenWeather API key in .env

# 3. Build and start all services
docker-compose up --build -d
```

---

## 🚀 Quick Start

### 1. Train the Model

```bash
python -m src.train
```

This trains all 6 models, performs hyperparameter tuning, and saves the best model.

### 2. Generate SHAP Explanations

```bash
# Generate all explanation artifacts
python scripts/generate_explanations.py

# Or generate global/local separately
python scripts/generate_explanations.py --global-only
python scripts/generate_explanations.py --local-only
```

### 3. Run the Flask API

```bash
# Development
python api/app.py

# Production (with gunicorn)
gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 4 wsgi:app
```

### 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Test Everything

```bash
# Test the API
python scripts/test_api.py

# Test the dashboard
python scripts/test_dashboard_v2.py

# Test weather integration
python scripts/test_weather_integration.py

# Verify deployment configuration
python scripts/verify_deployment.py
```

---

## 📁 Project Structure

```
AI-Weather-Intelligence/
├── api/                          # Flask REST API
│   ├── __init__.py               # Package initialization
│   ├── app.py                    # Flask application & endpoints
│   ├── config.py                 # Configuration classes
│   └── services.py               # Model service layer
│
├── dashboard/                    # Streamlit dashboard
│   └── app.py                    # Interactive dashboard application
│
├── data/
│   ├── raw/                      # Raw dataset (weatherAUS.csv)
│   └── processed/                # Cleaned and preprocessed data
│
├── models/                       # Trained model artifacts
│   ├── best_model.joblib         # Best model pipeline (XGBoost)
│   ├── categorical_encoder.joblib
│   ├── numeric_scaler.joblib
│   ├── preprocessing_metadata.joblib
│   └── model_comparison_results.csv
│
├── notebooks/                    # Jupyter notebooks
│   └── 01_data_understanding.ipynb
│
├── reports/                      # Evaluation & explanation artifacts
│   ├── explanations/             # SHAP plots and contribution tables
│   ├── roc_curve_*.png           # ROC curves per model
│   ├── pr_curve_*.png            # Precision-recall curves per model
│   └── model_comparison_summary.csv
│
├── scripts/                      # Utility scripts
│   ├── generate_explanations.py  # SHAP explanation generator
│   ├── test_api.py               # API test suite
│   ├── test_dashboard_v2.py      # Dashboard test suite
│   ├── test_weather_integration.py
│   └── verify_deployment.py      # Deployment verification
│
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data cleaning & preprocessing
│   ├── feature_engineering.py    # Feature engineering
│   ├── train.py                  # Model training pipeline
│   ├── evaluate.py               # Model evaluation
│   ├── explainability.py         # SHAP explainability module
│   ├── weather_service.py        # OpenWeather API integration
│   └── predict.py                # Prediction utilities
│
├── Dockerfile                    # API Docker image
├── Dockerfile.dashboard          # Dashboard Docker image
├── docker-compose.yml            # Orchestration
├── render.yaml                   # Render.com deployment
├── DEPLOYMENT.md                 # Deployment guide
├── requirements.txt              # Python dependencies
├── wsgi.py                       # WSGI entry point
├── .env.example                  # Environment variables template
└── .dockerignore
```

---

## 🔄 Preprocessing Pipeline

The preprocessing pipeline (`src/data_preprocessing.py`) performs:

### 1. Data Cleaning
- **Duplicate removal**: Removes duplicate rows
- **Type conversion**: Converts date strings to datetime, numeric strings to floats
- **Missing value handling**: Drops rows with missing target values

### 2. Outlier Treatment
- **IQR-based clipping**: Uses Interquartile Range rule (1.5x)
- **Quantile clipping**: Bounds clipped to 1st/99th percentiles

### 3. Feature Engineering
- **Temperature metrics**: `TemperatureDifference`, `AverageTemperature`
- **Humidity metrics**: `HumidityIndex`
- **Pressure metrics**: `PressureDifference`
- **Time-based features**: `Month`, `Quarter`, `DayOfYear`, `Season`, `IsWeekend`
- **Wind features**: `WindIntensityCategory`
- **Rainfall features**: `RainfallIndicator`, `RainTodayBinary`

### 4. Data Splitting
- **Training**: 70% (stratified)
- **Validation**: 15% (stratified)
- **Test**: 15% (stratified)

### 5. Preprocessing Transformers
- **Numeric features**: Median imputation + StandardScaler
- **Categorical features**: Most-frequent imputation + OneHotEncoder
- **Total pipeline output**: 133 features

---

## 🤖 Machine Learning Pipeline

### Models Evaluated

| Model | ROC-AUC | Accuracy | F1 | CV Score |
|-------|---------|----------|-----|----------|
| **XGBoost** | **0.8911** | **0.8578** | **0.6295** | **0.7463** |
| LightGBM | 0.8910 | 0.8565 | 0.6251 | 0.7467 |
| Random Forest | 0.8888 | 0.8543 | 0.6031 | 0.7271 |
| Gradient Boosting | 0.8811 | 0.8510 | 0.6064 | 0.7365 |
| Logistic Regression | 0.8696 | 0.8465 | 0.5953 | 0.7316 |
| Decision Tree | 0.8474 | 0.8388 | 0.5737 | 0.7037 |

### Training Process

1. **Feature selection**: 33 original + engineered features
2. **Data splitting**: Stratified 70/15/15 train/val/test
3. **Preprocessing**: ColumnTransformer with imputation, scaling, one-hot encoding
4. **Hyperparameter tuning**: GridSearchCV with 3-fold StratifiedKFold
5. **Model comparison**: All models evaluated on ROC-AUC
6. **Best model selection**: Highest ROC-AUC on test set

### Evaluation Artifacts
- ROC curves for all 6 models
- Precision-Recall curves for all 6 models
- Confusion matrices for all 6 models
- Model comparison summary CSV

---

## 📡 API Documentation

### Endpoints

#### `GET /` - API Documentation
Returns API metadata and all available endpoints.

#### `GET /api/health` - Health Check
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "model_loaded": true,
    "model_type": "XGBClassifier",
    "timestamp": 1785849620.05
  }
}
```

#### `GET /api/v1/model` - Model Metadata
Returns model type, feature columns, target column, and classes.

#### `GET /api/v1/features` - Feature Schema
Returns the 33 feature columns expected by the model.

#### `POST /api/v1/predict` - Single Prediction
**Request:**
```json
{
  "Location": "Sydney",
  "MinTemp": 12.0,
  "MaxTemp": 22.0,
  "Rainfall": 0.0,
  "Humidity9am": 70.0,
  "Humidity3pm": 55.0,
  "Pressure9am": 1015.0,
  "Pressure3pm": 1012.0,
  "WindSpeed3pm": 15.0,
  "WindGustSpeed": 30.0,
  "Cloud3pm": 5.0,
  "Temp3pm": 20.0,
  "...": "..."
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "prediction": 0,
    "prediction_label": "No Rain",
    "probability": 0.1168,
    "confidence": 0.8832,
    "probabilities": {"0": 0.8832, "1": 0.1168}
  }
}
```

#### `POST /api/v1/predict/batch` - Batch Prediction
Accepts an array of weather observations and returns predictions for each.

### Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `INVALID_INPUT` | Missing or invalid request fields |
| 404 | `NOT_FOUND` | Endpoint not found |
| 405 | `METHOD_NOT_ALLOWED` | HTTP method not allowed |
| 413 | `PAYLOAD_TOO_LARGE` | Request body too large |
| 500 | `INTERNAL_ERROR` | Unexpected server error |
| 503 | `MODEL_NOT_FOUND` | Model not loaded |

---

## 📊 Dashboard

The Streamlit dashboard features 5 pages:

### 🏠 Dashboard
- Dataset overview metrics
- Temperature trends by city
- Rainfall distribution
- Monthly rain probability

### 🔮 Predict & Explain
- **Manual input** or **live weather** modes
- SHAP feature contribution charts
- Waterfall and force plots
- Confidence score cards

### 📊 Historical Analysis
- Multi-city analysis with date filters
- Temperature trends comparison
- Rainfall box plots
- Humidity vs temperature correlation
- Summary statistics tables

### 🌍 Multi-City Comparison
- Live weather comparison across cities
- Rain probability comparison charts
- Temperature comparison charts

### 📈 Model Insights
- Model performance metrics (ROC-AUC, Accuracy, F1)
- Global SHAP explanations
- Top features by mean |SHAP|
- Model comparison charts

---

## 🔍 Explainable AI (SHAP)

### Global Explanations
- **Summary (beeswarm) plot**: Shows feature impact across all predictions
- **Global feature importance**: Mean absolute SHAP values
- **Dependence plots**: Feature value vs SHAP value relationships

### Local Explanations
- **Waterfall plots**: Decomposition of individual predictions
- **Force plots**: Visual representation of feature contributions
- **Contribution tables**: Per-feature SHAP values with direction

### Key Insights From SHAP
Top contributing features to rain prediction:
1. **Humidity3pm** - Higher humidity strongly increases rain probability
2. **WindGustSpeed** - Higher wind gusts correlate with rain events
3. **Rainfall** - Existing rainfall is a strong predictor
4. **Pressure3pm** - Lower pressure increases rain probability
5. **Cloud3pm** - Higher cloud cover indicates rain likelihood

---

## 📈 Performance Metrics

### Model Performance

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.8911 |
| **Accuracy** | 85.78% |
| **Precision** | 75.66% |
| **Recall** | 53.90% |
| **F1 Score** | 62.95% |
| **CV Score** | 0.7463 |

### System Metrics
- **API response time**: <100ms per single prediction
- **Model loading time**: ~2 seconds at startup
- **Weather API caching**: 5-minute TTL reduces API calls
- **Dashboard load time**: <2 seconds with cached data

---

## 🐳 Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for complete deployment guide.

### Quick Deploy Options

#### Docker Compose
```bash
cp .env.example .env
docker-compose up --build -d
```

#### Render.com
Push to GitHub → New Blueprint → Connect repo → Auto-deploy

#### AWS ECS
Build images → Push to ECR → Create Fargate tasks

---

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENWEATHER_API_KEY` | OpenWeather API key | - |
| `API_PORT` | Flask API port | 8000 |
| `DASHBOARD_PORT` | Streamlit port | 8501 |
| `LOG_LEVEL` | Logging level | INFO |
| `CORS_ORIGINS` | Allowed origins | * |
| `MODEL_PATH` | Model file path | models/best_model.joblib |
| `PREPROCESSING_METADATA_PATH` | Metadata path | models/preprocessing_metadata.joblib |
| `CLEANED_DATA_PATH` | Cleaned data path | data/processed/cleaned_weather_dataset.csv |

---

## 🧪 Testing

```bash
# API tests (38 tests)
python scripts/test_api.py

# Live API tests (29 tests)
python scripts/test_weather_integration.py

# Dashboard tests
python scripts/test_dashboard_v2.py

# SHAP explanation verification
python scripts/verify_explanations.py

# Deployment configuration verification (71 checks)
python scripts/verify_deployment.py
```

---

## 🚀 Future Enhancements

### Short-term
- [ ] **More weather features**: Add UV index, solar radiation, evapotranspiration
- [ ] **Time series models**: LSTM/GRU for sequential prediction
- [ ] **Model serving**: MLflow model registry integration
- [ ] **Real-time streaming**: WebSocket support for live predictions

### Medium-term
- [ ] **Multi-target prediction**: Predict rainfall amount, not just probability
- [ ] **Geographic visualizations**: Interactive maps for all Australian cities
- [ ] **User accounts**: Save prediction history and custom dashboards
- [ ] **Model retraining pipeline**: Automated retraining with new data

### Long-term
- [ ] **Ensemble predictions**: Combine multiple models with meta-learning
- [ ] **Active learning**: Improve model with user feedback
- [ ] **Federated learning**: District-level models for local accuracy
- [ ] **Mobile app**: React Native / Flutter companion app
- [ ] **International expansion**: Support weather data from other countries

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Write clean, documented Python code
- Add tests for new functionality
- Update documentation for API changes
- Follow the existing code style

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Australian Weather Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) on Kaggle
- **SHAP**: [SHapley Additive exPlanations](https://github.com/shap/shap) library
- **XGBoost**: [XGBoost](https://github.com/dmlc/xgboost) gradient boosting framework
- **OpenWeather**: [OpenWeather API](https://openweathermap.org/api)
- **Streamlit**: [Streamlit](https://streamlit.io/) dashboard framework

---

<div align="center">
  Made with ❤️ by the AI Weather Intelligence Team
</div>