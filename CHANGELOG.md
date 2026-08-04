# Changelog

All notable changes to the AI Weather Intelligence Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-04

### Added

#### Core ML Pipeline
- Initial weather prediction model with 6 model comparison
- XGBoost selected as best model (ROC-AUC: 0.8911, Accuracy: 85.78%)
- Hyperparameter tuning with GridSearchCV and 3-fold StratifiedKFold
- Comprehensive model evaluation with ROC curves, PR curves, confusion matrices
- Feature engineering module with 14 additional weather features

#### Explainable AI (SHAP)
- `src/explainability.py` module with reusable SHAP explanation functions
- Global explanations: summary (beeswarm) plots, feature importance, dependence plots
- Local explanations: waterfall plots, force plots, contribution tables
- CLI script `scripts/generate_explanations.py` for generating explanation artifacts
- Explanation verification script `scripts/verify_explanations.py`

#### REST API
- Production-ready Flask API with structured JSON responses
- Endpoints: health check, model metadata, feature schema, single/batch predictions
- Centralized exception handling with consistent error codes
- CORS support and configurable logging
- WSGI entry point (`wsgi.py`) for production deployment
- API test suites (`scripts/test_api.py`, `scripts/test_live_api.py`)

#### Streamlit Dashboard
- Professional 5-page dashboard with modern responsive design
- Interactive prediction with SHAP explanations
- Live weather data via OpenWeather API integration
- Historical weather analysis with interactive Plotly charts
- Multi-city weather comparison
- Model insights with performance metrics
- Custom CSS styling with gradient headers, metric cards, and prediction cards

#### OpenWeather API Integration
- `src/weather_service.py` modular service class
- API key management (env var or runtime input)
- Automatic fallback to Open-Meteo API
- File-based caching with 5-minute TTL
- Automatic preprocessing of live data into model-ready features
- 15 supported Australian cities

#### Docker & Deployment
- Multi-stage Docker builds for API (`Dockerfile`) and dashboard (`Dockerfile.dashboard`)
- Docker Compose orchestration with health checks and shared volumes
- Render.com deployment configuration (`render.yaml`)
- Environment variable template (`.env.example`)
- Deployment verification script (`scripts/verify_deployment.py`)
- Comprehensive deployment guide (`DEPLOYMENT.md`)

#### Documentation
- Professional README with architecture diagram, features, and usage
- Detailed dataset and preprocessing pipeline documentation
- Complete API documentation with request/response examples
- Performance metrics and model comparison tables
- Contributing guidelines (`CONTRIBUTING.md`)
- Deployment guide for Docker, Render.com, and AWS

### Fixed
- SHAP dependence plot generation with feature data extraction
- Streamlit dashboard `use_container_width` compatibility
- PowerShell compatibility for curl commands

### Security
- Non-root users in Docker containers
- API keys excluded from Docker images
- Input validation for all API endpoints
- Structured error handling without leaking internal details

## [Unreleased]

### Planned
- Time series models (LSTM/GRU) for sequential prediction
- MLflow model registry integration
- WebSocket support for real-time predictions
- Interactive geographic visualizations
- User accounts and prediction history
- Ensemble predictions with meta-learning
- International weather data support

---

## Commit History

### feat: project initialization
- Initial project structure with data preprocessing, feature engineering, and model training

### feat: model training pipeline
- Added 6 model comparison with hyperparameter tuning
- Saved best model and evaluation artifacts

### feat: SHAP explainability
- Added comprehensive SHAP explanation module
- Generated global and local explanation plots

### feat: Flask REST API
- Added production-ready API with health checks and predictions

### feat: Streamlit dashboard
- Built professional interactive dashboard with 5 pages

### feat: OpenWeather integration
- Added live weather fetching with caching and error handling

### feat: Docker deployment
- Added Dockerfiles, docker-compose, and Render deployment config

### docs: comprehensive documentation
- Added README, DEPLOYMENT, and CONTRIBUTING guides