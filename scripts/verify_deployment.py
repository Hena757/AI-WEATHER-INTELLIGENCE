"""Verify the Docker deployment configuration files."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

print("=" * 70)
print("DEPLOYMENT CONFIGURATION VERIFICATION")
print("=" * 70)

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] {name}" + (f" - {detail}" if detail and not condition else ""))


# 1. Check required files exist
print("\n[1] Required Files")
required_files = [
    "Dockerfile",
    "Dockerfile.dashboard",
    "docker-compose.yml",
    ".dockerignore",
    ".env.example",
    "render.yaml",
    "DEPLOYMENT.md",
    "wsgi.py",
    "requirements.txt",
]
for f in required_files:
    path = BASE_DIR / f
    check(f"{f} exists", path.exists(), f"missing: {f}")

# 2. Validate Dockerfile
print("\n[2] Dockerfile (Flask API)")
dockerfile = (BASE_DIR / "Dockerfile").read_text(encoding="utf-8")
check("Uses python:3.12-slim", "python:3.12-slim" in dockerfile)
check("Multi-stage build", "AS builder" in dockerfile and "AS production" in dockerfile)
check("Installs gunicorn", "gunicorn==23.0.0" in dockerfile)
check("Non-root user", "USER weatherapi" in dockerfile)
check("Exposes port 8000", "EXPOSE 8000" in dockerfile)
check("Health check curl", "curl" in dockerfile)
check("Copies api/", "COPY api/" in dockerfile)
check("Copies src/", "COPY src/" in dockerfile)
check("Copies models/", "COPY models/" in dockerfile)
check("Copies wsgi.py", "COPY wsgi.py" in dockerfile)
check("Gunicorn CMD", "gunicorn" in dockerfile and "wsgi:app" in dockerfile)

# 3. Validate Dockerfile.dashboard
print("\n[3] Dockerfile.dashboard (Streamlit)")
dash_dockerfile = (BASE_DIR / "Dockerfile.dashboard").read_text(encoding="utf-8")
check("Uses python:3.12-slim", "python:3.12-slim" in dash_dockerfile)
check("Multi-stage build", "AS builder" in dash_dockerfile and "AS production" in dash_dockerfile)
check("Non-root user", "USER weatherdash" in dash_dockerfile)
check("Exposes port 8501", "EXPOSE 8501" in dash_dockerfile)
check("Copies dashboard/", "COPY dashboard/" in dash_dockerfile)
check("Copies src/", "COPY src/" in dash_dockerfile)
check("Copies models/", "COPY models/" in dash_dockerfile)
check("Copies data/", "COPY data/" in dash_dockerfile)
check("Streamlit CMD", "streamlit" in dash_dockerfile and "dashboard/app.py" in dash_dockerfile)
check("Headless mode", "STREAMLIT_SERVER_HEADLESS=true" in dash_dockerfile)

# 4. Validate docker-compose.yml
print("\n[4] docker-compose.yml")
compose = (BASE_DIR / "docker-compose.yml").read_text(encoding="utf-8")
check("Has api service", "api:" in compose)
check("Has dashboard service", "dashboard:" in compose)
check("API port mapping", "${API_PORT:-8000}:8000" in compose or "8000:8000" in compose)
check("Dashboard port mapping", "${DASHBOARD_PORT:-8501}:8501" in compose or "8501:8501" in compose)
check("API healthcheck", "api/health" in compose)
check("Dashboard healthcheck", "_stcore/health" in compose)
check("Named volumes", "weather_data:" in compose and "weather_reports:" in compose)
check("Restart policy", "restart: unless-stopped" in compose)
check("OpenWeather env var", "OPENWEATHER_API_KEY" in compose)

# 5. Validate .dockerignore
print("\n[5] .dockerignore")
dockerignore = (BASE_DIR / ".dockerignore").read_text(encoding="utf-8")
check("Excludes .git", ".git" in dockerignore)
check("Excludes __pycache__", "__pycache__/" in dockerignore)
check("Excludes .venv", ".venv/" in dockerignore)
check("Excludes .env", ".env" in dockerignore)
check("Excludes raw data", "data/raw/" in dockerignore)
check("Excludes notebooks", "notebooks/" in dockerignore)

# 6. Validate .env.example
print("\n[6] .env.example")
env_example = (BASE_DIR / ".env.example").read_text(encoding="utf-8")
check("Has OPENWEATHER_API_KEY", "OPENWEATHER_API_KEY" in env_example)
check("Has API_PORT", "API_PORT" in env_example)
check("Has DASHBOARD_PORT", "DASHBOARD_PORT" in env_example)
check("Has LOG_LEVEL", "LOG_LEVEL" in env_example)
check("Has CORS_ORIGINS", "CORS_ORIGINS" in env_example)
check("Has MODEL_PATH", "MODEL_PATH" in env_example)

# 7. Validate render.yaml
print("\n[7] render.yaml")
render = (BASE_DIR / "render.yaml").read_text(encoding="utf-8")
check("Has API service", "weather-intelligence-api" in render)
check("Has dashboard service", "weather-intelligence-dashboard" in render)
check("API Dockerfile", "Dockerfile" in render)
check("Dashboard Dockerfile", "Dockerfile.dashboard" in render)
check("API healthcheck", "/api/health" in render)
check("Dashboard healthcheck", "/_stcore/health" in render)
check("OpenWeather env var", "OPENWEATHER_API_KEY" in render)

# 8. Validate DEPLOYMENT.md
print("\n[8] DEPLOYMENT.md")
deployment = (BASE_DIR / "DEPLOYMENT.md").read_text(encoding="utf-8")
check("Has Docker section", "## Docker Deployment" in deployment)
check("Has Render section", "## Render.com Deployment" in deployment)
check("Has AWS section", "## AWS Deployment" in deployment)
check("Has env vars section", "## Environment Variables" in deployment)
check("Has production section", "## Production Considerations" in deployment)
check("Has troubleshooting", "## Troubleshooting" in deployment)

# 9. Validate requirements.txt
print("\n[9] requirements.txt")
requirements = (BASE_DIR / "requirements.txt").read_text(encoding="utf-8")
check("Has Flask", "Flask==" in requirements)
check("Has flask-cors", "flask-cors==" in requirements)
check("Has streamlit", "streamlit==" in requirements)
check("Has shap", "shap==" in requirements)
check("Has scikit-learn", "scikit-learn==" in requirements)
check("Has pandas", "pandas==" in requirements)
check("Has requests", "requests==" in requirements)

print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 70)

if failed > 0:
    sys.exit(1)