# Deployment Guide for AI Weather Intelligence Platform

This guide covers Docker deployment, Render.com deployment, and AWS deployment options for the AI Weather Intelligence Platform.

## Table of Contents
1. [Docker Deployment](#docker-deployment)
2. [Render.com Deployment](#rendercom-deployment)
3. [AWS Deployment](#aws-deployment)
4. [Environment Variables](#environment-variables)
5. [Production Considerations](#production-considerations)

---

## Docker Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+

### Quick Start

```bash
# 1. Copy environment configuration
cp .env.example .env

# 2. Edit .env and set your OpenWeather API key
# OPENWEATHER_API_KEY=your_key_here

# 3. Build and start all services
docker-compose up --build -d

# 4. Check service status
docker-compose ps

# 5. View logs
docker-compose logs -f
```

### Access the Services
- **Flask API**: http://localhost:8000
  - Health check: http://localhost:8000/api/health
  - API docs: http://localhost:8000/
- **Streamlit Dashboard**: http://localhost:8501

### Docker Commands

```bash
# Build only the API
docker build -t weather-api -f Dockerfile .

# Build only the dashboard
docker build -t weather-dashboard -f Dockerfile.dashboard .

# Run API only
docker run -p 8000:8000 -e OPENWEATHER_API_KEY=your_key weather-api

# Run dashboard only
docker run -p 8501:8501 -e OPENWEATHER_API_KEY=your_key weather-dashboard

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild after code changes
docker-compose up --build -d
```

### Docker Architecture

```
┌─────────────────────────────────────────────────┐
│              Docker Compose                     │
│                                                 │
│  ┌──────────────┐    ┌────────────────────┐    │
│  │  Flask API   │    │  Streamlit Dashboard│    │
│  │  :8000       │    │  :8501             │    │
│  │  gunicorn    │    │  streamlit         │    │
│  └──────┬───────┘    └─────────┬──────────┘    │
│         │                      │               │
│  ┌──────┴──────────────────────┴──────────┐    │
│  │  Shared Volumes                        │    │
│  │  - weather_data (cache)               │    │
│  │  - weather_reports (explanations)     │    │
│  └───────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

---

## Render.com Deployment

### Option 1: Blueprint (Infrastructure as Code)

Render supports deployment via the `render.yaml` file included in this repository.

1. **Push code to GitHub**
2. **In Render Dashboard**: New → Blueprint
3. **Connect your repository**
4. **Render will auto-detect** `render.yaml` and create both services
5. **Set the `OPENWEATHER_API_KEY`** environment variable in each service

### Option 2: Manual Deployment

#### Flask API Service
1. **New Web Service** → Connect repository
2. **Name**: `weather-intelligence-api`
3. **Environment**: `Docker`
4. **Dockerfile Path**: `./Dockerfile`
5. **Health Check Path**: `/api/health`
6. **Environment Variables**:
   - `FLASK_ENV=production`
   - `PORT=8000`
   - `OPENWEATHER_API_KEY=your_key`

#### Streamlit Dashboard Service
1. **New Web Service** → Connect repository
2. **Name**: `weather-intelligence-dashboard`
3. **Environment**: `Docker`
4. **Dockerfile Path**: `./Dockerfile.dashboard`
5. **Health Check Path**: `/_stcore/health`
6. **Environment Variables**:
   - `STREAMLIT_SERVER_PORT=8501`
   - `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
   - `STREAMLIT_SERVER_HEADLESS=true`
   - `OPENWEATHER_API_KEY=your_key`

---

## AWS Deployment

### Option 1: AWS ECS with Fargate

#### Prerequisites
- AWS CLI configured
- ECR repository created

#### Steps

```bash
# 1. Build and push API image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t weather-api .
docker tag weather-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/weather-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/weather-api:latest

# 2. Build and push dashboard image
docker build -t weather-dashboard -f Dockerfile.dashboard .
docker tag weather-dashboard:latest <account>.dkr.ecr.us-east-1.amazonaws.com/weather-dashboard:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/weather-dashboard:latest

# 3. Create ECS task definitions and services
# Use the docker-compose.yml as reference for container configs
```

#### ECS Task Definition (API)
```json
{
  "family": "weather-api",
  "networkMode": "awsvpc",
  "containerDefinitions": [
    {
      "name": "weather-api",
      "image": "<account>.dkr.ecr.us-east-1.amazonaws.com/weather-api:latest",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "environment": [
        {"name": "FLASK_ENV", "value": "production"},
        {"name": "PORT", "value": "8000"},
        {"name": "OPENWEATHER_API_KEY", "value": "your_key"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/weather-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "weather"
        }
      }
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024"
}
```

### Option 2: AWS Elastic Beanstalk

1. **Create application** with Docker platform
2. **Upload source** or connect to GitHub
3. **Configure environment variables** in the EB console
4. **Deploy** - EB will use the Dockerfile

### Option 3: AWS EC2 with Docker

```bash
# 1. Launch EC2 instance with Docker AMI
# 2. SSH into instance
# 3. Clone repository
git clone https://github.com/Hena757/AI-WEATHER-INTELLIGENCE.git
cd AI-WEATHER-INTELLIGENCE

# 4. Set environment variables
export OPENWEATHER_API_KEY=your_key

# 5. Build and run with docker-compose
docker-compose up --build -d

# 6. Configure security group to allow ports 8000 and 8501
```

---

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENWEATHER_API_KEY` | OpenWeather API key | - | No (uses fallback) |
| `API_PORT` | Flask API port | 8000 | No |
| `DASHBOARD_PORT` | Streamlit port | 8501 | No |
| `LOG_LEVEL` | Logging level | INFO | No |
| `CORS_ORIGINS` | Allowed CORS origins | * | No |
| `MODEL_PATH` | Model file path | /app/models/best_model.joblib | No |
| `PREPROCESSING_METADATA_PATH` | Metadata path | /app/models/preprocessing_metadata.joblib | No |
| `CLEANED_DATA_PATH` | Cleaned data path | /app/data/processed/cleaned_weather_dataset.csv | No |

---

## Production Considerations

### Security
- **API keys**: Never commit `.env` files. Use platform secret managers (Render env vars, AWS Secrets Manager)
- **Non-root users**: Docker images run as non-root users
- **CORS**: Configure `CORS_ORIGINS` to specific domains in production
- **HTTPS**: Use platform-provided SSL (Render auto-provisions, AWS requires ALB/CloudFront)

### Performance
- **Gunicorn**: API uses 2 workers with 4 threads each
- **Streamlit**: Runs in headless mode
- **Caching**: Weather data cached for 5 minutes to reduce API calls
- **Model loading**: Model loaded once at startup, not per request

### Monitoring
- **Health checks**: Both services have health check endpoints
- **Logging**: Structured logs to stdout (captured by platform)
- **Metrics**: Use platform monitoring (Render metrics, CloudWatch)

### Scaling
- **API**: Scale horizontally by increasing gunicorn workers or ECS tasks
- **Dashboard**: Scale with multiple Streamlit instances behind a load balancer
- **Database**: No database required - model artifacts are static files

### Data Persistence
- **Weather cache**: Stored in Docker volumes
- **SHAP explanations**: Generated at runtime and cached
- **Model artifacts**: Baked into Docker image for consistency

---

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Change ports in .env
   API_PORT=8001
   DASHBOARD_PORT=8502
   ```

2. **OpenWeather API not working**
   - Check `OPENWEATHER_API_KEY` is set
   - System automatically falls back to Open-Meteo

3. **Model not loading**
   - Ensure `models/best_model.joblib` exists
   - Check `MODEL_PATH` environment variable

4. **Docker build fails**
   - Ensure Docker daemon is running
   - Check disk space
   - Try `docker-compose build --no-cache`

### Logs

```bash
# API logs
docker-compose logs api

# Dashboard logs
docker-compose logs dashboard

# All logs
docker-compose logs -f