# Contributing to AI Weather Intelligence

Thank you for considering contributing to the AI Weather Intelligence Platform! This document outlines the contribution guidelines and development workflow.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Git Commit Guidelines](#git-commit-guidelines)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** to your GitHub account
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/AI-WEATHER-INTELLIGENCE.git
   cd AI-WEATHER-INTELLIGENCE
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/Hena757/AI-WEATHER-INTELLIGENCE.git
   ```
4. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
6. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. **Update your main branch**:
   ```bash
   git checkout main
   git pull upstream main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make your changes** and commit them:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

4. **Pull the latest changes** before pushing:
   ```bash
   git pull --rebase upstream main
   ```

5. **Push and open a pull request**:
   ```bash
   git push origin feature/your-feature
   ```

## Code Style

### Python
- Follow **PEP 8** style guide
- Use **type hints** for all function signatures
- Write **docstrings** for all public functions and classes (Google style)
- Keep functions **small and focused**
- Use **meaningful variable names**
- Maximum line length: **120 characters**
- Use `from __future__ import annotations` for modern type hints

### Example Docstring
```python
def load_model_pipeline(model_path: str | Path) -> Any:
    """Load the trained sklearn Pipeline from disk.
    
    Parameters
    ----------
    model_path : str | Path
        Path to the serialized model file.
    
    Returns
    -------
    Any
        The loaded sklearn Pipeline object.
    
    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    ...
```

### Imports
Order imports alphabetically:
1. Standard library
2. Third-party packages
3. Local modules

```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.data_preprocessing import clean_dataset
```

## Testing

### Run All Tests
```bash
# API tests
python scripts/test_api.py

# Dashboard tests
python scripts/test_dashboard_v2.py

# Weather integration tests
python scripts/test_weather_integration.py

# Deployment verification
python scripts/verify_deployment.py
```

### Test Requirements
- All new functionality must have **tests**
- Tests should be **deterministic** (no random failures)
- Tests should **fail fast** with clear error messages
- Cover **edge cases** and **error conditions**

## Documentation

### Documentation Standards
- Update **README.md** for user-facing changes
- Update **DEPLOYMENT.md** for deployment changes
- Add **docstrings** for new functions
- Document **API changes** in API section
- Keep documentation **up to date** with code

### Documentation Structure
Each new feature should include:
1. Usage example
2. API reference (if applicable)
3. Configuration options
4. Deployment considerations

## Git Commit Guidelines

### Commit Message Format
Use **Conventional Commits** format:

```
<type>(<scope>): <subject>

<body body=..body...>

<footer..footer...>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Code style (formatting, whitespace)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding/updating tests
- `chore`: Build/tooling/maintenance
- `ci`: CI configuration changes

### Examples
```
feat(api): add batch prediction endpoint

Add support for predicting multiple weather observations
in a single API request.

Closes #123
```

```
fix(dashboard): handle missing OpenWeather API key

Show a clear warning message when the API key is not
configured instead of silently failing.

Fixes #456
```

### Commit Rules
- Use **imperative mood** ("add" not "added")
- Keep subject **under 50 characters**
- Reference issues/PRs when applicable
- One logical change per commit

## Pull Request Process

### Preparation Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Code is formatted properly
- [ ] No unnecessary files changed
- [ ] Branch is up to date with main

### PR Title Format
```
<type>: <description>
```

Examples:
- `feat: add XGBoost model evaluation`
- `fix: resolve dashboard loading error`
- `docs: update API documentation`

### PR Description Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the testing performed.

## Screenshots
If applicable, add screenshots.

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests passing
- [ ] Documentation updated
```

### Review Process
1. Maintainer reviews the PR
2. Automated checks run
3. Discussion and requested changes
4. Approval and merge

## Issue Reporting

### Bug Reports
When reporting a bug, include:
- **Description** of the issue
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Environment** (OS, Python version)
- **Screenshots** (if applicable)
- **Error logs**

### Feature Requests
When requesting a feature, include:
- **Clear description** of the feature
- **Use case** for the feature
- **Expected behavior**
- **Potential implementation approach**

## Development Setup

### Running the API
```bash
# Development server
python api/app.py

# With live reload
flask --app api/app --debug run
```

### Running the Dashboard
```bash
streamlit run dashboard/app.py
```

### Running Tests
```bash
python -m pytest tests/ -v
```

### Generating Explanations
```bash
python scripts/generate_explanations.py
```

## Additional Resources

- [Project README](README.md)
- [Deployment Guide](DEPLOYMENT.md)
- [OpenWeather API](https://openweathermap.org/api)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [SHAP Documentation](https://shap.readthedocs.io/)