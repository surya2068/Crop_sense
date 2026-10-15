# CropSense — Complete Agricultural Crop Yield & ML Analytics Dashboard

CropSense is a Flask-based machine-learning web application for agricultural crop yield forecasting, input optimization, and exploratory data analysis. Built on comprehensive multi-year Indian agricultural data (1997–2020), CropSense analyzes environmental and management factors, compares regression and ensemble models, and delivers live yield predictions alongside operational underperformance risk alerts.

## Main Features

1. **Dashboard** (`/`) — Comprehensive dataset summary (records, crops, states, seasons), project workflow stages, and live ML performance metrics.
2. **Data** (`/data`) — Processed dataset preview with feature definitions and data types.
3. **EDA** (`/eda`) — Full exploratory data analysis including descriptive statistics, missing value audit, duplicate checks, IQR outlier detection, correlation heatmaps, distribution plots, and automated findings.
4. **Regression** (`/regression`) — Supervised comparison of Linear Regression, Ridge, and Random Forest Regressor on `Yield` with target leakage prevention, evaluation tables, actual vs. predicted plots, residual histograms, feature importance rankings, and an instant AJAX prediction form.
5. **Ensemble** (`/ensemble`) — Comparison of Random Forest, Decision Tree, AdaBoost, and XGBoost regressor architectures with metrics and interactive AJAX inference.
6. **Predict** (`/predict`) — Live prediction interface accepting Crop, State, Season, Area, Annual Rainfall, Fertilizer, and Pesticide inputs to estimate expected Yield ($t/ha$), Total Production ($tons$), and Underperformance Risk with probability.

## Dataset & Leakage Prevention

The project utilizes the `crop_yield.csv` dataset containing **19,689 agricultural records** across **55 crops**, **30 states**, and **6 seasons**:
- `Area` (cultivated hectares)
- `Annual_Rainfall` (precipitation in mm)
- `Fertilizer` & `Pesticide` (applied chemical dosages in kg)
- `Crop_Year` (1997–2020)
- `Yield` (production per unit area — target variable)

### Target Leakage Prevention
In agricultural data, `Yield = Production / Area`. If `Production` were included as an input feature, the models would trivially memorize the arithmetic ratio rather than learning actual environmental and input relationships. CropSense strictly isolates `Production` during training, enabling pre-harvest yield forecasting.

### Underperformance Alert Definition
Analogous to solar fault detection, CropSense derives an operational **Underperformance Alert**:
- `UNDERPERFORMANCE_FLAG = 1` when `Annual_Rainfall > 800 mm`, `Fertilizer_per_Area > 20 kg/ha`, and realized `Yield` falls below 50% of the median yield for that specific crop; otherwise `0`.

## Course Outcomes (CO1 – CO5) Mapping

CropSense fulfills university Course Outcomes CO1 through CO5 across the machine learning lifecycle:

| Outcome | Description | Implemented Project Module | Route & Endpoints |
|---|---|---|---|
| **CO1** | End-to-End ML System Lifecycle | Data Ingestion, Cleaning, Preprocessing, End-to-End Traced Pipeline | `/`, `/data`, `/eda` |
| **CO2** | Linear Supervised Learning & Regularization | Linear Regression, Ridge, Lasso, Elastic Net with Scaling & Bias-Variance Analysis | `/regression` |
| **CO3** | Tree-based Ensembles & Feature Importance | Decision Trees, Random Forests, AdaBoost, XGBoost with Feature Importances | `/ensemble` |
| **CO4** | Unsupervised Learning & Dimensionality Reduction | K-Means (Elbow & Silhouette), Hierarchical Ward Clustering, DBSCAN, PCA Scree, t-SNE | `/unsupervised` |
| **CO5** | Evaluation, Validation & Optimization | 3-Way Split (70/15/15), 5-Fold Cross Validation, ROC-AUC, PR-AUC, Calibration Curve (Brier Score), Grid vs Random vs Bayesian Search | `/evaluation` |

## Installation & Running Locally

```powershell
cd "Crop Sense ml"
python -m pip install -r requirements.txt
python app.py
```

Open in your browser: `http://127.0.0.1:5000`

---

## Docker Deployment & Pushing to Docker Hub

This project is fully containerized with a production-grade [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml).

### 1. Build the Docker Image
```bash
docker build -t cropsense:latest .
```

### 2. Run the Container Locally
```bash
docker run -d -p 5000:5000 --name cropsense-app cropsense:latest
```
Or with Docker Compose:
```bash
docker compose up -d
```
Then open: `http://localhost:5000`

### 3. Push Image to Docker Hub (for Submission / Sharing)
1. Log in to your Docker Hub account:
   ```bash
   docker login
   ```
2. Tag the image with your Docker Hub username:
   ```bash
   docker tag cropsense:latest <your_dockerhub_username>/cropsense:latest
   ```
3. Push the image to Docker Hub:
   ```bash
   docker push <your_dockerhub_username>/cropsense:latest
   ```
4. Verify by pulling on any machine:
   ```bash
   docker run -d -p 5000:5000 <your_dockerhub_username>/cropsense:latest
   ```

